import io

import matplotlib.pyplot as plt
import torch
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

    if C in (3, 4):
        rgb = t[:3]
        out = _tensor_to_plot_data(rgb.permute(1, 2, 0))
        return out

    return _tensor_to_plot_data(t.permute(1, 2, 0))
    # Multichannel (first 3 shown upstream)


def plot_tensors(lr, sr, hr, title="Train"):
    """Render LR–SR–HR triplets from a batch as a single PIL image.

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

    Returns:
        PIL.Image.Image: RGB image containing a grid with columns `[LR | SR | HR]`
        and up to two rows (first two items of the batch).

    Notes:
        - Only the first two items of the batch are visualized to avoid large figures.
        - Supports grayscale, RGB, RGBA, and generic multi-channel inputs
          (first 3 channels shown).
        - This function is side-effect free for tensors (uses `.detach()` and
          plots copies), and closes the matplotlib figure after rendering.
    """
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
