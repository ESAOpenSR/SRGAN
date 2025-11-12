"""Flexible generator variants built from shared model blocks.

Provides a modular SR backbone with pluggable residual blocks:
`res` (ResidualBlockNoBN), `rcab` (attention), `rrdb` (dense-in-residual),
and `lka` (large-kernel). Uses a head → body → tail design with PixelShuffle
upsampling for {×2, ×4, ×8}.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn

from ..model_blocks import (
    ResidualBlockNoBN,
    RCAB,
    RRDB,
    LKAResBlock,
    make_upsampler,
)

__all__ = ["FlexibleGenerator", "flexible_generator"]


_BLOCK_REGISTRY: Dict[str, Callable[[int, int], nn.Module]] = {
    "res": lambda n_channels, kernel: ResidualBlockNoBN(
        n_channels=n_channels,
        kernel_size=kernel,
        res_scale=0.2,
    ),
    "rcab": lambda n_channels, kernel: RCAB(
        n_channels=n_channels,
        kernel_size=kernel,
        reduction=16,
        res_scale=0.2,
    ),
    "rrdb": lambda n_channels, kernel: RRDB(
        n_features=n_channels,
        growth_channels=max(16, n_channels // 3),
        res_scale=0.2,
    ),
    "lka": lambda n_channels, kernel: LKAResBlock(
        n_channels=n_channels,
        kernel_size=kernel,
        res_scale=0.2,
    ),
}


class FlexibleGenerator(nn.Module):
    """Modular super-resolution generator with pluggable residual blocks.

    Provides a single, drop-in generator backbone that can be instantiated with
    different residual block families—**res**, **rcab**, **rrdb**, or **lka**—all
    built from a shared interface. The network follows a head → body → tail
    design with learnable upsampling:

        Head (large-kernel conv) → N × Block(type=block_type) → Body tail conv
        → Skip add → Upsampler (×2/×4/×8) → Output conv

    Use this when you want to compare architectural choices or sweep hyper-
    parameters without changing call sites.

    Args:
        in_channels (int): Number of input channels (e.g., RGB=3, RGB-NIR=4/6).
        n_channels (int): Base feature width used throughout the backbone.
        n_blocks (int): Number of residual blocks in the body.
        small_kernel (int): Kernel size for body/ tail convolutions.
        large_kernel (int): Kernel size for head/ output convolutions.
        scale (int): Upscaling factor; one of {2, 4, 8}.
        block_type (str): Residual block family in {"res","rcab","rrdb","lka"}.

    Attributes:
        head (nn.Sequential): Large-receptive-field stem conv + activation.
        body (nn.Sequential): Sequence of residual blocks of the selected type.
        body_tail (nn.Conv2d): Fusion conv after the residual stack.
        upsampler (nn.Module): PixelShuffle-style learnable upsampling to `scale`.
        tail (nn.Conv2d): Final projection to `in_channels`.

    Raises:
        ValueError: If `scale` is not in {2, 4, 8} or `block_type` is unknown.

    Example:
        >>> g = FlexibleGenerator(in_channels=3, block_type="rcab", scale=4)
        >>> x = torch.randn(1, 3, 64, 64)
        >>> y = g(x)  # (1, 3, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_channels: int = 96,
        n_blocks: int = 32,
        small_kernel: int = 3,
        large_kernel: int = 9,
        scale: int = 8,
        block_type: str = "rcab",
    ) -> None:
        super().__init__()

        if scale not in {2, 4, 8}:
            raise ValueError("scale must be one of {2, 4, 8}")

        block_key = block_type.lower()
        if block_key not in _BLOCK_REGISTRY:
            raise ValueError("block_type must be one of {'res', 'rcab', 'rrdb', 'lka'}")

        self.scale = scale

        padding_large = large_kernel // 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, large_kernel, padding=padding_large),
            nn.PReLU(),
        )

        block_factory = _BLOCK_REGISTRY[block_key]
        self.body = nn.Sequential(
            *[block_factory(n_channels, small_kernel) for _ in range(n_blocks)]
        )
        self.body_tail = nn.Conv2d(
            n_channels, n_channels, small_kernel, padding=small_kernel // 2
        )
        self.upsampler = make_upsampler(n_channels, scale)
        self.tail = nn.Conv2d(
            n_channels, in_channels, large_kernel, padding=padding_large
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the flexible SR generator.

        Parameters
        ----------
        x : torch.Tensor
            Low-resolution input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Super-resolved output tensor of shape (B, C, sH, sW),
            where `s` ∈ {2, 4, 8} is the configured upscaling factor.

        Workflow
        --------
        1) Extract shallow features with the head conv.
        2) Transform via a stack of residual blocks (type = `block_type`).
        3) Fuse with a 3×3 body tail conv and add a long skip.
        4) Upsample via PixelShuffle-based `make_upsampler`.
        5) Project back to image space with the output conv.
        """
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        feat = feat + res
        feat = self.upsampler(feat)
        return self.tail(feat)


flexible_generator = FlexibleGenerator
