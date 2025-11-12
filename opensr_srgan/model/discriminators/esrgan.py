"""ESRGAN discriminator architecture (VGG-style).

Implements the deep convolutional discriminator introduced in
*Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)*
by Wang et al., ECCV 2018.

References
----------
- Wang et al., ESRGAN
"""
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
    """
    Construct a convolutional block with optional batch normalization and LeakyReLU.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    stride : int
        Convolution stride.
    use_batch_norm : bool, default=True
        Whether to include a batch normalization layer.

    Returns
    -------
    nn.Sequential
        A sequential block: Conv2d → (BatchNorm2d) → LeakyReLU(0.2).
    """
    padding = kernel_size // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class ESRGANDiscriminator(nn.Module):
    """
    VGG-style discriminator network used in ESRGAN.

    This discriminator processes high-resolution image patches and predicts
    a scalar realism score. It follows the VGG-style design with progressively
    increasing channel depth and downsampling via strided convolutions.

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (e.g., 3 for RGB, 4 for RGB-NIR).
    base_channels : int, default=64
        Number of channels in the first convolutional layer; subsequent layers
        scale as powers of two.
    linear_size : int, default=1024
        Size of the intermediate fully-connected layer before the output scalar.

    Attributes
    ----------
    features : nn.Sequential
        Convolutional feature extractor backbone.
    pool : nn.AdaptiveAvgPool2d
        Global pooling layer to aggregate spatial features.
    classifier : nn.Sequential
        Fully connected layers producing a single output value.
    n_layers : int
        Total number of convolutional blocks (for metadata/reference only).

    Raises
    ------
    ValueError
        If `base_channels` or `linear_size` is not a positive integer.

    Examples
    --------
    >>> disc = ESRGANDiscriminator(in_channels=3)
    >>> x = torch.randn(8, 3, 128, 128)
    >>> y = disc(x)
    >>> y.shape
    torch.Size([8, 1])
    """
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
        """
        Forward pass through the ESRGAN discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Discriminator logits of shape (B, 1), where higher values
            indicate more realistic images.
        """
        feats = self.features(x)
        pooled = self.pool(feats).view(x.size(0), -1)
        return self.classifier(pooled)
