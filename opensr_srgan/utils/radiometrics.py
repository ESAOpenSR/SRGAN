import torch
import numpy as np
import torch.nn.functional as F
from skimage import exposure
import torch

from .tensor_conversions import tensor_to_numpy


# -------------------------------------------------------------------------
# SENTINEL-2 NORMALIZATION HELPERS
# -------------------------------------------------------------------------
def normalise_s2(im: torch.Tensor, stage: str = "norm") -> torch.Tensor:
    """
    Normalize or denormalize Sentinel-2 image values.

    This function applies a symmetric scaling to map reflectance-like values
    to the range [-1, 1] for model input, and reverses it for visualization
    or saving.

    Parameters
    ----------
    im : torch.Tensor
        Input image tensor (any shape), typically reflectance-scaled.
    stage : {"norm", "denorm"}
        - "norm"   → normalize image to [-1, 1]
        - "denorm" → reverse normalization back to [0, 1]

    Returns
    -------
    torch.Tensor
        The normalized or denormalized image tensor.
    """
    assert stage in ["norm", "denorm"]
    value = 3.0  # reference scaling factor

    if stage == "norm":
        # Scale roughly from [0, value/10] → [0, 1] → [-1, 1]
        im = im * (10.0 / value)
        im = (im * 2) - 1
        im = torch.clamp(im, -1, 1)
    else:  # stage == "denorm"
        # Reverse mapping: [-1, 1] → [0, 1] → [0, value/10]
        im = (im + 1) / 2
        im = im * (value / 10.0)
        im = torch.clamp(im, 0, 1)

    return im


def normalise_10k(im: torch.Tensor, stage: str = "norm") -> torch.Tensor:
    """
    Normalize or denormalize Sentinel-2 data scaled in units of 10,000.

    This is the most common scaling for Sentinel-2 L2A reflectance data,
    where reflectance = DN / 10000.

    Parameters
    ----------
    im : torch.Tensor
        Input tensor (any shape), expected to contain DN values ~[0, 10000].
    stage : {"norm", "denorm"}
        - "norm"   → divide by 10,000 to map to [0, 1]
        - "denorm" → multiply by 10,000 to restore original scale

    Returns
    -------
    torch.Tensor
        Scaled tensor.
    """
    assert stage in ["norm", "denorm"]

    if stage == "norm":
        im = im / 10000.0
        im = torch.clamp(im, 0, 1)
    else:  # "denorm"
        im = im * 10000.0
        im = torch.clamp(im, 0, 10000)

    return im


def zero_one_signed(im: torch.Tensor, stage: str = "norm") -> torch.Tensor:
    """Convert values between the ``[0, 1]`` and ``[-1, 1]`` ranges.

    Parameters
    ----------
    im : torch.Tensor
        Tensor containing values already scaled to ``[0, 1]``.
    stage : {"norm", "denorm"}
        - ``"norm"``   → map ``[0, 1]`` → ``[-1, 1]`` using ``im * 2 - 1``.
        - ``"denorm"`` → map ``[-1, 1]`` → ``[0, 1]`` using ``(im + 1) / 2``.

    Returns
    -------
    torch.Tensor
        Range-adjusted tensor.
    """

    assert stage in ["norm", "denorm"]

    if stage == "norm":
        return torch.clamp((im * 2.0) - 1.0, -1.0, 1.0)
    return torch.clamp((im + 1.0) / 2.0, 0.0, 1.0)


def normalise_10k_signed(im: torch.Tensor, stage: str = "norm") -> torch.Tensor:
    """Normalize Sentinel-2 DN values to the symmetric ``[-1, 1]`` range.

    This helper chains :func:`normalise_10k` (``[0, 10000]`` → ``[0, 1]``) with
    :func:`zero_one_signed` (``[0, 1]`` → ``[-1, 1]``) so models that expect
    signed inputs receive the conventional ``[-1, 1]`` distribution.

    Parameters
    ----------
    im : torch.Tensor
        Input tensor containing reflectance-like values expressed as
        ``[0, 10000]``.
    stage : {"norm", "denorm"}
        Normalisation stage. ``"denorm"`` restores the original ``[0, 10000]``
        range.

    Returns
    -------
    torch.Tensor
        Tensor in the requested range.
    """

    assert stage in ["norm", "denorm"]

    if stage == "norm":
        scaled = normalise_10k(im, stage="norm")
        return zero_one_signed(scaled, stage="norm")
    scaled = zero_one_signed(im, stage="denorm")
    return normalise_10k(scaled, stage="denorm")


def sen2_stretch(im: torch.Tensor) -> torch.Tensor:
    """
    Apply a simple contrast stretch to Sentinel-2 data.

    Multiplies reflectance values by (10/3) ≈ 3.33 to increase dynamic range
    for visualization or augmentation purposes.

    Parameters
    ----------
    im : torch.Tensor
        Sentinel-2 tensor (any shape).

    Returns
    -------
    torch.Tensor
        Contrast-stretched image tensor.
    """
    stretched = im * (10 / 3.0)
    return torch.clamp(stretched, 0.0, 1.0)


def minmax_percentile(
    tensor: torch.Tensor, pmin: float = 2, pmax: float = 98
) -> torch.Tensor:
    """
    Perform percentile-based min-max normalization to [0, 1].

    Uses quantiles instead of absolute min/max to reduce outlier influence.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor (any shape).
    pmin : float
        Lower percentile (default 2%).
    pmax : float
        Upper percentile (default 98%).

    Returns
    -------
    torch.Tensor
        Tensor scaled to [0, 1] based on percentile range.
    """
    min_val = torch.quantile(tensor, pmin / 100.0)
    max_val = torch.quantile(tensor, pmax / 100.0)
    tensor = (tensor - min_val) / (max_val - min_val)
    return tensor


# -------------------------------------------------------------------------
# GENERAL UTILITIES
# -------------------------------------------------------------------------
def minmax(img: torch.Tensor) -> torch.Tensor:
    """
    Standard min-max normalization to [0, 1] over the entire tensor.

    Parameters
    ----------
    img : torch.Tensor
        Input tensor (any shape).

    Returns
    -------
    torch.Tensor
        Min-max normalized tensor.
    """
    min_val = torch.min(img)
    max_val = torch.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img


# ---------------------------------------------------------------
# HISTOGRAM MATCHING
# ---------------------------------------------------------------
def histogram(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Perform **per-channel histogram matching** of `target` → `reference`.

    Each channel in the target image is adjusted so that its cumulative
    distribution function (CDF) matches that of the corresponding channel
    in the reference image.
    This preserves overall color/radiometric tone relationships, but aligns
    the pixel intensity distributions more precisely than simple moment matching.

    Supports both single images and batched tensors.

    Parameters
    ----------
    reference : torch.Tensor
        Reference image or batch, shape (C, H, W) or (B, C, H, W).
        Its histogram will be used as the target distribution.
    target : torch.Tensor
        Target image or batch to be adjusted, shape (C, H, W) or (B, C, H, W).
        Must have the same number of channels as `reference`.

    Returns
    -------
    torch.Tensor
        Histogram-matched version of the target, with the same shape and dtype
        as the input. If a single image is given, returns shape (C, H, W).
    """

    # Ensure both inputs have correct dimensionality: either (C,H,W) or (B,C,H,W)
    assert target.ndim in (3, 4) and reference.ndim in (
        3,
        4,
    ), "Expected (C,H,W) or (B,C,H,W) for both reference and target"

    # Save device/dtype for conversion back later
    device, dtype = target.device, target.dtype

    # --- Normalize to batch form (always 4D: B,C,H,W) ---
    # If inputs are 3D (single images), temporarily add batch dimension
    ref = reference.unsqueeze(0) if reference.ndim == 3 else reference
    tgt = target.unsqueeze(0) if target.ndim == 3 else target

    # Extract shapes
    B_ref, C_ref, H_ref, W_ref = ref.shape
    B_tgt, C_tgt, H_tgt, W_tgt = tgt.shape

    # Channel sanity check
    assert C_ref == C_tgt, f"Channel mismatch: reference={C_ref}, target={C_tgt}"

    # --- Resize reference spatially to match target ---
    # Uses bilinear interpolation, no corner alignment, safe for float data
    if (H_ref, W_ref) != (H_tgt, W_tgt):
        ref = F.interpolate(
            ref.to(dtype=torch.float32),
            size=(H_tgt, W_tgt),
            mode="bilinear",
            align_corners=False,
        )

    # Convert to NumPy for histogram matching operations
    ref_np = tensor_to_numpy(ref)
    tgt_np = tensor_to_numpy(tgt)
    out_np = np.empty_like(tgt_np)  # preallocate output array

    # --- Loop over batches and channels ---
    for b in range(B_tgt):
        # If reference has only one batch (B_ref=1), broadcast it to all targets
        rb = b % B_ref

        for c in range(C_tgt):
            ref_ch = ref_np[rb, c]  # reference channel
            tgt_ch = tgt_np[b, c]  # target channel

            # Mask invalid pixels (NaN or Inf)
            mask = np.isfinite(tgt_ch) & np.isfinite(ref_ch)

            if mask.any():
                # Perform per-channel histogram matching
                matched = exposure.match_histograms(tgt_ch[mask], ref_ch[mask])
                out = tgt_ch.copy()
                out[mask] = matched
                out_np[b, c] = out
            else:
                # If no valid pixels, copy target as-is
                out_np[b, c] = tgt_ch

    # Convert back to torch tensor on the original device/dtype
    out = torch.from_numpy(out_np).to(device=device, dtype=dtype)

    # If the original input was 3D, remove the temporary batch dimension
    return out[0] if target.ndim == 3 else out


# ---------------------------------------------------------------
# MOMENT MATCHING
# ---------------------------------------------------------------
def moment(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Perform **moment matching** between two multispectral image tensors.

    Each channel in the `target` image is rescaled to match the mean and
    standard deviation (first and second moments) of the corresponding channel
    in the `reference` image.
    This operation effectively transfers the global radiometric statistics
    (brightness and contrast) from `reference` to `target`.

    Parameters
    ----------
    reference : torch.Tensor
        Reference image whose per-channel statistics will be matched (e.g. Sentinel-2),
        shape (C, H, W).
    target : torch.Tensor
        Target image to be adjusted (e.g. SPOT-6),
        shape (C, H, W).
    reference_amount : float, optional
        Currently unused. Can later be used to control blending strength between
        the original target and the moment-matched output (0–1).

    Returns
    -------
    torch.Tensor
        Moment-matched image with shape (C, H, W),
        where each target channel now has the same mean and standard deviation
        as the corresponding reference channel.
    """

    device, dtype = target.device, target.dtype

    # Convert to NumPy arrays for easier numerical processing
    reference_np = tensor_to_numpy(reference)
    target_np = tensor_to_numpy(target)

    matched_channels = []

    # Iterate channel-wise through reference and target
    for ref_ch, tgt_ch in zip(reference_np, target_np):

        # --- Compute per-channel mean and std ---
        ref_mean = np.mean(ref_ch)
        tgt_mean = np.mean(tgt_ch)
        ref_std = np.std(ref_ch)
        tgt_std = np.std(tgt_ch)

        # --- Apply moment matching formula ---
        # Normalize target → scale by reference std → shift by reference mean
        matched_channel = (((tgt_ch - tgt_mean) / tgt_std) * ref_std) + ref_mean
        matched_channels.append(matched_channel)

    matched_np = np.stack(matched_channels, axis=0)

    # Convert back to PyTorch tensor with channel-first format (C, H, W)
    matched = torch.from_numpy(matched_np).to(device=device, dtype=dtype)

    return matched
