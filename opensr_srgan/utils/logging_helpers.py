import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .radiometrics import minmax_percentile
from .tensor_conversions import tensor_to_numpy


def _tensor_to_plot_data(t: torch.Tensor):
    """Convert a CPU tensor to a NumPy array suitable for matplotlib visualization.

    Ensures contiguous memory layout before conversion and detaches the tensor
    from the computation graph. Typically used for final image preparation
    in validation or inference visualizations.

    Args:
        t (torch.Tensor): Input tensor to convert. Can have arbitrary shape.

    Returns:
        numpy.ndarray: NumPy array representation of the tensor, compatible
        with matplotlib plotting functions.
    """
    return tensor_to_numpy(t.contiguous())


def _to_numpy_img(t: torch.Tensor):
    """Convert a normalized image tensor in (C, H, W) format to a NumPy array.

    Supports grayscale, RGB, and RGBA images, as well as arbitrary
    multichannel tensors (which are converted by permuting to H×W×C).

    Args:
        t (torch.Tensor): Input image tensor in channel-first (C, H, W) format,
            with pixel values expected in the range [0, 1].

    Returns:
        numpy.ndarray: NumPy array in channel-last format (H, W, C) suitable
        for matplotlib visualization.

    Raises:
        ValueError: If the input tensor does not have exactly three dimensions.
    """
    if t.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(t.shape)}")
    C, H, W = t.shape
    t = t.detach().clamp(0, 1)
    if C == 1:
        out = _tensor_to_plot_data(t[0])
        return out  # grayscale

    if C >= 3:
        rgb = t[:3]
        out = _tensor_to_plot_data(rgb.permute(1, 2, 0))
        return out

    return _tensor_to_plot_data(t.permute(1, 2, 0))


def _select_display_bands(t: torch.Tensor) -> torch.Tensor:
    """Keep only bands that can be displayed before contrast stretching."""
    if t.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(t.shape)}")
    if t.shape[1] > 3:
        return t[:, :3]
    return t


def plot_tensors(lr, sr, hr, title="Train", *, matched=None):
    """Render LR–SR–HR rows, optionally including an SR matched column.

    Performs percentile-based min–max stretching and clamping to [0, 1] on each
    tensor, converts channel-first images to numpy for plotting, and arranges up
    to two samples (rows) with three columns (LR, SR, HR) into a matplotlib
    figure. The figure is then rasterized and returned as a `PIL.Image`.

    Args:
        lr (torch.Tensor): Low-resolution batch tensor of shape `(B, C, H, W)`,
            values expected in an arbitrary range (stretched internally).
        sr (torch.Tensor): Super-resolved batch tensor of shape `(B, C, H, W)`.
        hr (torch.Tensor): High-resolution (target) batch tensor of shape `(B, C, H, W)`.
        title (str, optional): Figure title placed above the grid. Defaults to `"Train"`.
        matched (torch.Tensor, optional): Corrected SR shown beside raw SR.
            All columns share a display range when this is supplied.

    Returns:
        PIL.Image.Image: RGB grid with up to two rows and columns
        `[LR | SR | HR]` or `[LR | SR | SR matched | HR]`.

    Notes:
        - Only the first two items of the batch are visualized to avoid large figures.
        - Supports grayscale, RGB, RGBA, and generic multi-channel inputs
          (first 3 channels shown).
        - This function is side-effect free for tensors (uses `.detach()` and
          plots copies), and closes the matplotlib figure after rendering.
    """
    # Drop non-visible bands before stretching. For RGB-NIR data, including NIR in
    # the percentile stretch can crush RGB contrast and produce misleading plots.
    lr = _select_display_bands(lr)
    sr = _select_display_bands(sr)
    hr = _select_display_bands(hr)

    if matched is not None:
        matched = _select_display_bands(matched)
        rows = []
        for index in range(min(2, lr.shape[0])):
            images = [lr[index], sr[index], matched[index], hr[index]]
            rows.append(_panel_image(images, ["LR", "SR", "SR matched", "HR"], title))
        result = Image.new(
            "RGB", (max(row.width for row in rows), sum(row.height for row in rows))
        )
        top = 0
        for row in rows:
            result.paste(row, (0, top))
            top += row.height
        return result

    # --- denorm(?) + stretch  ---
    lr = minmax_percentile(lr)
    sr = minmax_percentile(sr)
    hr = minmax_percentile(hr)

    # clamp in-place-friendly
    lr, sr, hr = lr.clamp(0, 1), sr.clamp(0, 1), hr.clamp(0, 1)

    # shapes
    B, C, H, W = lr.shape  # (B,C,H,W)

    # Determine colormap for grayscale images
    if C == 1:
        cmap = "bone"
    else:
        cmap = None

    # limit to max_n (images to plot ontop of each other)
    max_n = 2
    if B > max_n:
        lr = lr[:max_n]
        sr = sr[:max_n]
        hr = hr[:max_n]
        B = max_n

    # figure/axes: always 2D array even for B==1
    fixed_width = 15
    variable_height = (15 / 3) * B
    fig, axes = plt.subplots(
        B, 3, figsize=(fixed_width, variable_height), squeeze=False
    )

    # loop over batch
    with torch.no_grad():
        for i in range(B):
            img_lr = _to_numpy_img(lr[i].detach().cpu())
            img_sr = _to_numpy_img(sr[i].detach().cpu())
            img_hr = _to_numpy_img(hr[i].detach().cpu())

            axes[i, 0].imshow(img_lr)
            axes[i, 0].set_title("LR")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(img_sr)
            axes[i, 1].set_title("SR")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(img_hr)
            axes[i, 2].set_title("HR")
            axes[i, 2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    # render to PIL
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    pil_image = Image.open(buf).convert("RGB").copy()
    buf.close()
    plt.close(fig)
    return pil_image


def _common_display_range(images: list[torch.Tensor]) -> tuple[float, float]:
    """Return one robust display range shared by all images in a panel."""
    values = torch.cat([image.detach().float().reshape(-1) for image in images])
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 0.0, 1.0
    low, high = torch.quantile(finite, finite.new_tensor([0.02, 0.98]))
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return low, high


def _panel_image(
    images: list[torch.Tensor], labels: list[str], title: str
) -> Image.Image:
    """Render equally scaled channel images as an RGB PIL image."""
    low, high = _common_display_range(images)
    fig, axes = plt.subplots(
        1, len(images), figsize=(4 * len(images), 4), squeeze=False
    )
    for axis, image, label in zip(axes[0], images, labels):
        displayed = image.detach().float().clamp(low, high).sub(low).div(high - low)
        axis.imshow(_to_numpy_img(displayed.cpu()))
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB").copy()
    buffer.close()
    plt.close(fig)
    return result


def save_validation_artifacts(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    output_dir: str | Path,
    epoch: int,
    num_images: int,
    *,
    sample_offset: int = 0,
    local_images: bool = True,
    is_global_zero: bool = True,
    matched: torch.Tensor | None = None,
    live_sr: torch.Tensor | None = None,
) -> list[Path]:
    """Save LR/bicubic/SR/[SR matched]/HR RGB and NIR validation panels.

    Tensors use their supplied model/data values. Each sample and displayed
    band uses one shared percentile range across all columns, so comparisons remain
    meaningful. The caller should pass ``is_global_zero`` from Lightning to
    prevent duplicate files under DDP.
    """
    if not local_images or not is_global_zero or num_images <= 0:
        return []
    if any(t.ndim != 4 for t in (lr, sr, hr)):
        raise ValueError("lr, sr, and hr must have shape (B,C,H,W)")
    if not (lr.shape[0] == sr.shape[0] == hr.shape[0]):
        raise ValueError("lr, sr, and hr must contain the same batch size")
    if sr.shape[-2:] != hr.shape[-2:]:
        raise ValueError("sr and hr must have the same spatial shape")
    if matched is not None and matched.shape != sr.shape:
        raise ValueError("matched and sr must have the same shape")
    if live_sr is not None and live_sr.shape != sr.shape:
        raise ValueError("live_sr and sr must have the same shape")

    destination = Path(output_dir).expanduser() / "validation_images"
    destination.mkdir(parents=True, exist_ok=True)
    count = min(int(num_images), lr.shape[0])
    upsampled_lr = F.interpolate(lr[:count].float(), size=hr.shape[-2:], mode="nearest")
    bicubic = F.interpolate(
        lr[:count].float(), size=hr.shape[-2:], mode="bicubic", align_corners=False
    )
    outputs: list[Path] = []
    for index in range(count):
        sources = [upsampled_lr[index], bicubic[index], sr[index], hr[index]]
        labels = ["LR", "Bicubic", "SR", "HR"]
        if live_sr is not None:
            labels[2] = "SR EMA"
            sources.insert(3, live_sr[index])
            labels.insert(3, "SR live (non-EMA)")
        if matched is not None:
            sources.insert(3, matched[index])
            labels.insert(3, "SR matched")
        for band_name in ("rgb", "nir"):
            if band_name == "rgb" and sr.shape[1] < 3:
                continue
            if band_name == "nir" and sr.shape[1] < 4:
                continue
            selected = [
                source[:3] if band_name == "rgb" else source[3:4] for source in sources
            ]
            path = (
                destination
                / f"epoch-{int(epoch):04d}_sample-{int(sample_offset) + index:03d}_{band_name}.png"
            )
            _panel_image(
                selected,
                labels,
                f"Epoch {epoch} | {band_name.upper()}",
            ).save(path)
            outputs.append(path)
    return outputs
