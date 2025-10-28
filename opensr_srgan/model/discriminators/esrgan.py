"""Discriminator architecture used in ESRGAN."""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["ESRGANDiscriminator"]


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    use_batch_norm: bool = True,
) -> nn.Sequential:
    padding = kernel_size // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class ESRGANDiscriminator(nn.Module):
    """Deep VGG-style discriminator introduced with ESRGAN."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 64,
        linear_size: int = 1024,
    ) -> None:
        super().__init__()

        if base_channels <= 0:
            raise ValueError("base_channels must be a positive integer.")

        if linear_size <= 0:
            raise ValueError("linear_size must be a positive integer.")

        features: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _conv_block(base_channels, base_channels, kernel_size=4, stride=2),
            _conv_block(base_channels, base_channels * 2, kernel_size=3, stride=1),
            _conv_block(base_channels * 2, base_channels * 2, kernel_size=4, stride=2),
            _conv_block(base_channels * 2, base_channels * 4, kernel_size=3, stride=1),
            _conv_block(base_channels * 4, base_channels * 4, kernel_size=4, stride=2),
            _conv_block(base_channels * 4, base_channels * 8, kernel_size=3, stride=1),
            _conv_block(base_channels * 8, base_channels * 8, kernel_size=4, stride=2),
            _conv_block(base_channels * 8, base_channels * 16, kernel_size=3, stride=1),
            _conv_block(base_channels * 16, base_channels * 16, kernel_size=4, stride=2),
        ]

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 16, linear_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(linear_size, 1),
        )

        self.base_channels = base_channels
        self.linear_size = linear_size
        self.n_layers = 1 + 10  # first conv + stacked blocks

    def forward(self, x: Tensor) -> Tensor:
        feats = self.features(x)
        pooled = self.pool(feats).view(x.size(0), -1)
        return self.classifier(pooled)
