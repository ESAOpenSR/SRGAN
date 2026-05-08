"""Reusable building blocks for SRGAN family models."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import init

from .EMA import ExponentialMovingAverage

__all__ = [
    "ConvolutionalBlock",
    "SubPixelConvolutionalBlock",
    "ResidualBlock",
    "ResidualBlockNoBN",
    "RCAB",
    "DenseBlock5",
    "RRDB",
    "LKA",
    "LKAResBlock",
    "make_upsampler",
    "ExponentialMovingAverage",
]


class ConvolutionalBlock(nn.Module):
    """A convolutional block comprised of Conv → (BN) → (Activation)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batch_norm: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__()

        act = activation.lower() if activation is not None else None
        if act is not None:
            if act not in {"prelu", "leakyrelu", "tanh"}:
                raise ValueError(
                    "activation must be one of {'prelu', 'leakyrelu', 'tanh'}"
                )

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        ]

        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if act == "prelu":
            layers.append(nn.PReLU())
        elif act == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2))
        elif act == "tanh":
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class SubPixelConvolutionalBlock(nn.Module):
    """Conv → PixelShuffle → PReLU upsampling block."""

    def __init__(
        self,
        kernel_size: int = 3,
        n_channels: int = 64,
        scaling_factor: int = 2,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * (scaling_factor**2),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class ResidualBlock(nn.Module):
    """BN-enabled residual block used in the original SRResNet."""

    def __init__(self, kernel_size: int = 3, n_channels: int = 64) -> None:
        super().__init__()
        self.conv_block1 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation="PReLu",
        )
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x + residual


class ResidualBlockNoBN(nn.Module):
    """Residual block variant without batch norm and with residual scaling."""

    def __init__(
        self,
        n_channels: int = 64,
        kernel_size: int = 3,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


class RCAB(nn.Module):
    """Residual Channel Attention Block (no BN)."""

    def __init__(
        self,
        n_channels: int = 64,
        kernel_size: int = 3,
        reduction: int = 16,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_channels, n_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // reduction, n_channels, 1),
            nn.Sigmoid(),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        w = self.se(y)
        return x + self.res_scale * (y * w)


class DenseBlock5(nn.Module):
    """ESRGAN-style dense block with five convolutions."""

    def __init__(self, n_features: int = 64, growth_channels: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.c1 = nn.Conv2d(n_features, growth_channels, kernel_size, padding=padding)
        self.c2 = nn.Conv2d(n_features + growth_channels, growth_channels, kernel_size, padding=padding)
        self.c3 = nn.Conv2d(n_features + 2 * growth_channels, growth_channels, kernel_size, padding=padding)
        self.c4 = nn.Conv2d(n_features + 3 * growth_channels, growth_channels, kernel_size, padding=padding)
        self.c5 = nn.Conv2d(n_features + 4 * growth_channels, n_features, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, n_features: int = 64, growth_channels: int = 32, res_scale: float = 0.2) -> None:
        super().__init__()
        self.db1 = DenseBlock5(n_features, growth_channels)
        self.db2 = DenseBlock5(n_features, growth_channels)
        self.db3 = DenseBlock5(n_features, growth_channels)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.db1(x)
        y = self.db2(y)
        y = self.db3(y)
        return x + self.res_scale * y


class LKA(nn.Module):
    """Lightweight Large-Kernel Attention module."""

    def __init__(self, n_channels: int = 64) -> None:
        super().__init__()
        self.dw5 = nn.Conv2d(n_channels, n_channels, 5, padding=2, groups=n_channels)
        self.dw7d = nn.Conv2d(n_channels, n_channels, 7, padding=9, dilation=3, groups=n_channels)
        self.pw = nn.Conv2d(n_channels, n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.dw5(x)
        attn = self.dw7d(attn)
        attn = self.pw(attn)
        return x * torch.sigmoid(attn)


class LKAResBlock(nn.Module):
    """Residual block incorporating Large-Kernel Attention."""

    def __init__(self, n_channels: int = 64, kernel_size: int = 3, res_scale: float = 0.2) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.act = nn.PReLU()
        self.lka = LKA(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.lka(y)
        y = self.conv2(y)
        return x + self.res_scale * y


def _icnr_(weight: torch.Tensor, scale: int = 2) -> None:
    """Apply ICNR initialization to a pre-pixel-shuffle convolution kernel."""

    out_channels, in_channels, k1, k2 = weight.shape
    if out_channels % (scale**2) != 0:
        raise ValueError("ICNR requires out_channels divisible by scale**2.")
    subkernel = torch.empty(
        out_channels // (scale**2),
        in_channels,
        k1,
        k2,
        device=weight.device,
        dtype=weight.dtype,
    )
    init.kaiming_normal_(subkernel)
    subkernel = subkernel.repeat_interleave(scale**2, dim=0)
    with torch.no_grad():
        weight.copy_(subkernel)


def make_upsampler(n_channels: int, scale: int, *, use_icnr: bool = False) -> nn.Sequential:
    """Create a pixel-shuffle upsampler matching the flexible generator implementation."""

    if scale < 1 or (scale & (scale - 1)) != 0:
        raise ValueError("scale must be a positive power of two (1, 2, 4, 8, ...).")

    stages: list[nn.Module] = []
    for _ in range(int(math.log2(scale))):
        conv = nn.Conv2d(n_channels, n_channels * 4, 3, padding=1)
        if use_icnr:
            _icnr_(conv.weight, scale=2)
            if conv.bias is not None:
                init.zeros_(conv.bias)
        stages.extend(
            [
                conv,
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        )
    return nn.Sequential(*stages)
