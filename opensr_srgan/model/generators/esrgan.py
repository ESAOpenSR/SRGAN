"""ESRGAN generator implementation built around RRDB blocks."""

from __future__ import annotations

from torch import Tensor, nn

from ..model_blocks import RRDB, make_upsampler

__all__ = ["ESRGANGenerator"]


class ESRGANGenerator(nn.Module):
    """Generator network used in ESRGAN.

    The architecture follows the design introduced in "ESRGAN: Enhanced
    Super-Resolution Generative Adversarial Networks" (Wang et al., 2018). It
    stacks multiple Residual-in-Residual Dense Blocks (RRDB) and performs
    PixelShuffle-based upsampling.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int | None = None,
        n_features: int = 64,
        n_blocks: int = 23,
        growth_channels: int = 32,
        res_scale: float = 0.2,
        scale: int = 4,
    ) -> None:
        super().__init__()

        if scale < 1 or scale & (scale - 1) != 0:
            raise ValueError("ESRGANGenerator only supports power-of-two scales (1, 2, 4, 8, ...).")

        if n_blocks < 1:
            raise ValueError("ESRGANGenerator requires at least one RRDB block.")

        if out_channels is None:
            out_channels = in_channels

        self.scale = scale
        self.n_blocks = n_blocks
        self.n_features = n_features
        self.growth_channels = growth_channels

        body_blocks = [RRDB(n_features, growth_channels, res_scale=res_scale) for _ in range(n_blocks)]

        self.conv_first = nn.Conv2d(in_channels, n_features, 3, padding=1)
        self.body = nn.Sequential(*body_blocks)
        self.conv_body = nn.Conv2d(n_features, n_features, 3, padding=1)
        self.upsampler = nn.Identity() if scale == 1 else make_upsampler(n_features, scale)
        self.conv_hr = nn.Conv2d(n_features, n_features, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv_last = nn.Conv2d(n_features, out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        first = self.conv_first(x)
        trunk = self.body(first)
        body_out = self.conv_body(trunk)
        feat = first + body_out
        feat = self.upsampler(feat)
        feat = self.activation(self.conv_hr(feat))
        return self.conv_last(feat)
