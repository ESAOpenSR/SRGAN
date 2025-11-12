"""SRGAN discriminator architectures built from shared blocks.

Implements the discriminator proposed in the original
*Super-Resolution Generative Adversarial Network (SRGAN)* paper by
Ledig et al. (CVPR 2017). The network follows a VGG-style layout of
convolutional blocks with alternating stride-1 and stride-2 layers,
progressively downsampling the input and classifying image realism.

References
----------
- Ledig et al., *Photo-Realistic Single Image Super-Resolution Using
  a Generative Adversarial Network*, CVPR 2017.
  https://arxiv.org/abs/1609.04802
"""
from __future__ import annotations

import torch
from torch import nn

from ..model_blocks import ConvolutionalBlock

__all__ = ["Discriminator"]


class Discriminator(nn.Module):
    """
    Standard SRGAN discriminator as defined in the original paper.

    The network alternates between stride-1 and stride-2 convolutional
    blocks, gradually reducing spatial resolution while increasing
    channel depth. The resulting features are globally pooled and
    classified via two fully connected layers to produce a single
    realism score.

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (e.g., 3 for RGB, 4 for multispectral SR).
    n_blocks : int, default=8
        Number of convolutional blocks to stack. Must be >= 1.

    Attributes
    ----------
    conv_blocks : nn.Sequential
        Sequential stack of convolutional feature extraction blocks.
    adaptive_pool : nn.AdaptiveAvgPool2d
        Global pooling layer reducing spatial dimensions to 6×6.
    fc1 : nn.Linear
        Fully connected layer mapping pooled features to hidden representation.
    fc2 : nn.Linear
        Output layer producing a single realism score.
    leaky_relu : nn.LeakyReLU
        Activation used in fully connected layers.
    base_channels : int
        Number of channels in the first convolutional layer (default 64).
    kernel_size : int
        Kernel size used in all convolutional blocks (default 3).
    fc_size : int
        Hidden dimension of the fully connected layer (default 1024).
    n_blocks : int
        Total number of convolutional blocks.

    Raises
    ------
    ValueError
        If `n_blocks` < 1.

    Examples
    --------
    >>> disc = Discriminator(in_channels=3, n_blocks=8)
    >>> x = torch.randn(4, 3, 96, 96)
    >>> y = disc(x)
    >>> y.shape
    torch.Size([4, 1])
    """
    def __init__(
        self,
        in_channels: int = 3,
        n_blocks: int = 8,
    ) -> None:
        super().__init__()

        if n_blocks < 1:
            raise ValueError("The SRGAN discriminator requires at least one block.")

        kernel_size = 3
        base_channels = 64
        fc_size = 1024

        conv_blocks: list[nn.Module] = []
        current_in = in_channels
        out_channels = base_channels
        for i in range(n_blocks):
            if i == 0:
                out_channels = base_channels
            elif i % 2 == 0:
                out_channels = current_in * 2
            else:
                out_channels = current_in

            conv_blocks.append(
                ConvolutionalBlock(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 == 0 else 2,
                    batch_norm=i != 0,
                    activation="LeakyReLu",
                )
            )
            current_in = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)

        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.fc_size = fc_size
        self.n_blocks = n_blocks

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SRGAN discriminator.

        Parameters
        ----------
        imgs : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Realism logits of shape (B, 1), where higher values
            indicate greater likelihood of being a real image.
        """
        batch_size = imgs.size(0)
        feats = self.conv_blocks(imgs)
        pooled = self.adaptive_pool(feats)
        flat = pooled.view(batch_size, -1)
        hidden = self.leaky_relu(self.fc1(flat))
        return self.fc2(hidden)
