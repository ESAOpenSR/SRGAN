"""SwinIR-style window-transformer generator for image super-resolution.

This is a clean PyTorch adaptation of the architectural ideas in SwinIR
(`Jingyun Liang et al., 2021 <https://arxiv.org/abs/2108.10257>`_): shallow
convolutional features, residual groups of shifted-window transformer blocks,
and a PixelShuffle reconstruction head.  It is deliberately **not** a copy of
the official implementation and it is not checkpoint-compatible with it.  In
particular, the public API and reconstruction head are adapted for arbitrary
multispectral input/output band counts used by OpenSRGAN.

The implementation is self-contained and requires only PyTorch.  Inputs and
outputs remain in the caller's reflectance convention (normally normalised to
``[0, 1]``): no ImageNet mean/std transform, sigmoid, or output clamp is
applied here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "SwinIRGenerator",
    "window_partition",
    "window_reverse",
    "calculate_mask",
]


_NEGATIVE_ATTENTION_BIAS: Final[float] = -100.0


def _require_positive_int(name: str, value: int) -> int:
    """Return a positive integer configuration value or raise a useful error."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """Split a BHWC tensor into non-overlapping windows.

    Parameters
    ----------
    x:
        Tensor with shape ``(batch, height, width, channels)``.  ``height`` and
        ``width`` must both be divisible by ``window_size``.
    window_size:
        Side length of each square window.

    Returns
    -------
    torch.Tensor
        Windows with shape ``(batch * num_windows, window_size, window_size,
        channels)``.  Channels are deliberately last to make token flattening
        and attention inexpensive.
    """
    if x.ndim != 4:
        raise ValueError(f"window_partition expects BHWC input, got {tuple(x.shape)}.")
    window_size = _require_positive_int("window_size", window_size)
    batch, height, width, channels = x.shape
    if height % window_size or width % window_size:
        raise ValueError(
            "window_partition requires height and width divisible by window_size; "
            f"got {(height, width)} and {window_size}."
        )
    return (
        x.reshape(
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size, window_size, channels)
    )


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    """Reassemble windows returned by :func:`window_partition` into BHWC.

    ``height`` and ``width`` refer to the padded feature map dimensions, not
    the original unpadded input dimensions.
    """
    if windows.ndim != 4:
        raise ValueError(
            "window_reverse expects windows shaped "
            "(batch*num_windows, window, window, channels); "
            f"got {tuple(windows.shape)}."
        )
    window_size = _require_positive_int("window_size", window_size)
    height = _require_positive_int("height", height)
    width = _require_positive_int("width", width)
    if height % window_size or width % window_size:
        raise ValueError("height and width must be divisible by window_size.")
    if windows.shape[1:3] != (window_size, window_size):
        raise ValueError(
            "Window tensor shape does not match window_size: "
            f"got {tuple(windows.shape[1:3])} and {window_size}."
        )
    windows_per_image = (height // window_size) * (width // window_size)
    if windows.shape[0] % windows_per_image:
        raise ValueError(
            "Number of windows is incompatible with requested height/width: "
            f"{windows.shape[0]} windows for {height}x{width}."
        )
    batch = windows.shape[0] // windows_per_image
    channels = windows.shape[-1]
    return (
        windows.reshape(
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
        .contiguous()
    )


def calculate_mask(
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor | None:
    """Build the additive mask needed by shifted-window attention.

    The input spatial dimensions must already have been padded to multiples of
    ``window_size``.  A value of zero permits attention and a large negative
    value suppresses tokens that became neighbours only because of the cyclic
    shift.  ``None`` is returned for unshifted blocks.
    """
    height = _require_positive_int("height", height)
    width = _require_positive_int("width", width)
    window_size = _require_positive_int("window_size", window_size)
    if not isinstance(shift_size, int) or isinstance(shift_size, bool):
        raise ValueError(f"shift_size must be an integer; got {shift_size!r}.")
    if shift_size < 0 or shift_size >= window_size:
        raise ValueError(
            "shift_size must be in [0, window_size); "
            f"got {shift_size} for window_size={window_size}."
        )
    if height % window_size or width % window_size:
        raise ValueError("height and width must be divisible by window_size.")
    if shift_size == 0:
        return None

    mask_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float32
    image_mask = torch.zeros((1, height, width, 1), device=device, dtype=mask_dtype)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = h_slices
    count = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            image_mask[:, h_slice, w_slice, :] = count
            count += 1

    mask_windows = window_partition(image_mask, window_size).reshape(-1, window_size**2)
    difference = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return difference.ne(0).to(mask_dtype).mul_(_NEGATIVE_ATTENTION_BIAS)


class DropPath(nn.Module):
    """Per-sample stochastic depth, implemented without optional dependencies."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must lie in [0, 1), got {drop_prob}.")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        random_tensor = keep_prob + torch.rand(
            (x.shape[0],) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype
        )
        return x * random_tensor.floor_().div(keep_prob)


class Mlp(nn.Module):
    """Pre-norm transformer feed-forward network."""

    def __init__(self, dim: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        if hidden_features < 1:
            raise ValueError("embed_dim * mlp_ratio must produce at least one feature.")
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.drop(self.fc2(x))


class WindowAttention(nn.Module):
    """Multi-head self-attention confined to square windows.

    A learned relative-position table supplies the spatial bias.  PyTorch's
    scaled-dot-product attention is used so CUDA kernels can select an
    efficient implementation when available.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        dim = _require_positive_int("dim", dim)
        window_size = _require_positive_int("window_size", window_size)
        num_heads = _require_positive_int("num_heads", num_heads)
        if dim % num_heads:
            raise ValueError(
                f"embed_dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        relative_size = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size * relative_size, num_heads)
        )
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords[0] += window_size - 1
        relative_coords[1] += window_size - 1
        relative_coords[0] *= relative_size
        self.register_buffer(
            "relative_position_index", relative_coords.sum(0), persistent=False
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _relative_position_bias(self, dtype: torch.dtype) -> Tensor:
        tokens = self.window_size**2
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(tokens, tokens, self.num_heads)
        return bias.permute(2, 0, 1).to(dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Attend tokens shaped ``(batch_windows, tokens, channels)``."""
        if x.ndim != 3:
            raise ValueError(
                "WindowAttention expects (batch_windows, tokens, channels); "
                f"got {tuple(x.shape)}."
            )
        batch_windows, tokens, channels = x.shape
        expected_tokens = self.window_size**2
        if tokens != expected_tokens or channels != self.dim:
            raise ValueError(
                "WindowAttention received incompatible token shape "
                f"{tuple(x.shape)}; expected (*, {expected_tokens}, {self.dim})."
            )
        qkv = (
            self.qkv(x)
            .reshape(batch_windows, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attention_bias = self._relative_position_bias(query.dtype).unsqueeze(0)
        if mask is not None:
            if mask.ndim != 3 or mask.shape[1:] != (tokens, tokens):
                raise ValueError(
                    "Attention mask must have shape (num_windows, tokens, tokens); "
                    f"got {tuple(mask.shape)}."
                )
            num_windows = mask.shape[0]
            if batch_windows % num_windows:
                raise ValueError("Attention batch is not divisible by the number of masks.")
            mask = mask.to(device=x.device, dtype=query.dtype)
            attention_bias = attention_bias + mask.repeat(batch_windows // num_windows, 1, 1).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(attended))


class SwinTransformerBlock(nn.Module):
    """Pre-norm Swin block with optional half-window cyclic shift."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        *,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)

    def forward(self, x: Tensor) -> Tensor:
        """Transform BCHW features whose spatial dimensions are window-aligned."""
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} features, received {channels}.")
        shortcut = x
        normalized = self.norm1(x.permute(0, 2, 3, 1))
        # A cyclic shift has no neighbouring window to connect when either
        # dimension contains just one window.  In that case the standard Swin
        # behaviour is an unshifted block; otherwise its boundary mask would
        # unnecessarily split the only window into disconnected regions.
        shift_size = (
            self.shift_size
            if min(height, width) > self.window_size
            else 0
        )
        if shift_size:
            normalized = torch.roll(
                normalized, shifts=(-shift_size, -shift_size), dims=(1, 2)
            )
        windows = window_partition(normalized, self.window_size)
        windows = windows.reshape(-1, self.window_size**2, channels)
        mask = calculate_mask(
            height,
            width,
            self.window_size,
            shift_size,
            device=x.device,
            dtype=x.dtype,
        )
        attended = self.attn(windows, mask).reshape(
            -1, self.window_size, self.window_size, channels
        )
        attended = window_reverse(attended, self.window_size, height, width)
        if shift_size:
            attended = torch.roll(
                attended, shifts=(shift_size, shift_size), dims=(1, 2)
            )
        x = shortcut + self.drop_path(attended.permute(0, 3, 1, 2))
        mlp_output = self.mlp(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x + self.drop_path(mlp_output)


class ResidualSwinTransformerGroup(nn.Module):
    """Alternating shifted-window blocks followed by convolutional residual fusion."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        *,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: Sequence[float],
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Every residual Swin group must contain at least one block.")
        if len(drop_path) != depth:
            raise ValueError("drop_path must contain one value per transformer block.")
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(
                dim,
                num_heads,
                window_size,
                0 if block_index % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[block_index],
            )
            for block_index in range(depth)
        )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return residual + self.conv(x)


class SwinIRGenerator(nn.Module):
    """Window-transformer super-resolution generator for multispectral imagery.

    The backbone follows the SwinIR design: a shallow convolution, residual
    groups of alternating regular/shifted window attention, global residual
    feature fusion, and a sub-pixel reconstruction head.  Inputs can have any
    spatial size, including dimensions smaller than a window; internally they
    are padded to whole attention windows and outputs are cropped exactly back
    to ``(scale * height, scale * width)``.

    Parameters
    ----------
    in_channels:
        Number of low-resolution input bands.
    out_channels:
        Number of super-resolved output bands.  Defaults to ``in_channels``.
    scale:
        Integer enlargement factor.  Values 2, 3, and 4 are supported.
    embed_dim:
        Width of transformer features.
    depths, num_heads:
        Parallel sequences defining transformer depth and attention heads for
        each residual group.  By default this is the six-group SwinIR-M style
        width/depth, but has no official checkpoint compatibility.
    window_size:
        Side length of local attention windows.
    mlp_ratio:
        Hidden-width multiplier of the feed-forward MLP.
    qkv_bias, drop_rate, attn_drop_rate, drop_path_rate:
        Standard transformer regularisation controls.
    use_checkpoint:
        Recompute transformer blocks during backward to reduce activation
        memory.  This should be enabled only when memory is limiting.

    Notes
    -----
    This module does not apply any range conversion, sigmoid, clipping, or
    ImageNet preprocessing.  It therefore preserves the SRGAN training
    pipeline's normalised reflectance convention.
    """

    _SUPPORTED_SCALES: Final[frozenset[int]] = frozenset({2, 3, 4})

    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int | None = None,
        scale: int = 4,
        embed_dim: int = 180,
        depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        in_channels = _require_positive_int("in_channels", in_channels)
        if out_channels is None:
            out_channels = in_channels
        out_channels = _require_positive_int("out_channels", out_channels)
        scale = _require_positive_int("scale", scale)
        if scale not in self._SUPPORTED_SCALES:
            supported = ", ".join(str(value) for value in sorted(self._SUPPORTED_SCALES))
            raise ValueError(f"scale must be one of {{{supported}}}; got {scale}.")
        embed_dim = _require_positive_int("embed_dim", embed_dim)
        window_size = _require_positive_int("window_size", window_size)
        if window_size < 2:
            raise ValueError("window_size must be at least 2 for shifted-window attention.")
        if not isinstance(depths, Sequence) or isinstance(depths, (str, bytes)):
            raise ValueError("depths must be a non-empty sequence of positive integers.")
        if not isinstance(num_heads, Sequence) or isinstance(num_heads, (str, bytes)):
            raise ValueError("num_heads must be a non-empty sequence of positive integers.")
        if not depths or len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must be non-empty sequences of equal length.")
        depths = tuple(_require_positive_int("depth", depth) for depth in depths)
        num_heads = tuple(_require_positive_int("num_heads", heads) for heads in num_heads)
        for heads in num_heads:
            if embed_dim % heads:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must divide evenly by every num_heads value; "
                    f"received {heads}."
                )
        if (
            isinstance(mlp_ratio, bool)
            or not isinstance(mlp_ratio, (float, int))
            or not math.isfinite(float(mlp_ratio))
            or mlp_ratio <= 0
        ):
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio!r}.")
        for name, value in (
            ("drop_rate", drop_rate),
            ("attn_drop_rate", attn_drop_rate),
            ("drop_path_rate", drop_path_rate),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1), got {value!r}.")
        if not isinstance(qkv_bias, bool):
            raise ValueError(f"qkv_bias must be a bool, got {qkv_bias!r}.")
        if not isinstance(use_checkpoint, bool):
            raise ValueError(f"use_checkpoint must be a bool, got {use_checkpoint!r}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = float(mlp_ratio)
        self.use_checkpoint = bool(use_checkpoint)

        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        all_drop_path = torch.linspace(0, float(drop_path_rate), sum(depths)).tolist()
        offset = 0
        groups: list[nn.Module] = []
        for depth, heads in zip(depths, num_heads):
            groups.append(
                ResidualSwinTransformerGroup(
                    embed_dim,
                    depth,
                    heads,
                    window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=float(drop_rate),
                    attn_drop=float(attn_drop_rate),
                    drop_path=all_drop_path[offset : offset + depth],
                    use_checkpoint=self.use_checkpoint,
                )
            )
            offset += depth
        self.layers = nn.ModuleList(groups)
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use the conventional small truncated-normal transformer initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _pad_input(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Pad BCHW input on bottom/right to a whole number of windows."""
        height, width = x.shape[-2:]
        padded_height = math.ceil(height / self.window_size) * self.window_size
        padded_width = math.ceil(width / self.window_size) * self.window_size
        pad_height = padded_height - height
        pad_width = padded_width - width
        if pad_height == 0 and pad_width == 0:
            return x, height, width
        # Reflection is a good boundary condition for ordinary crops but is not
        # defined when any padding side is as large as its input dimension.  A
        # replicate fallback keeps single-pixel and very narrow inputs valid.
        pad_mode = "reflect" if pad_height < height and pad_width < width else "replicate"
        return F.pad(x, (0, pad_width, 0, pad_height), mode=pad_mode), height, width

    def forward(self, x: Tensor) -> Tensor:
        """Return a super-resolved tensor with no range clamp or colour transform."""
        if x.ndim != 4:
            raise ValueError(f"SwinIRGenerator expects BCHW input, got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SwinIRGenerator expects {self.in_channels} input bands, got {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError("SwinIRGenerator expects a floating-point reflectance tensor.")
        x, original_height, original_width = self._pad_input(x)
        shallow = self.conv_first(x)
        features = shallow
        for group in self.layers:
            features = group(features)
        features = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = self.conv_after_body(features) + shallow
        output = self.upsample(self.conv_before_upsample(features))
        return output[..., : original_height * self.scale, : original_width * self.scale]
