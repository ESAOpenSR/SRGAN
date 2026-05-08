"""SRResNet architecture definitions built on top of shared model blocks."""

import math
import torch
from torch import nn

from opensr_srgan.utils.checkpoint_loading import load_checkpoint

from ..model_blocks import (
    ConvolutionalBlock,
    ResidualBlock,
    SubPixelConvolutionalBlock,
)


class SRResNet(nn.Module):
    """Canonical SRResNet generator for single-image super-resolution.

    Implements the SRResNet backbone with:
        1) Large-kernel stem conv (+PReLU)
        2) N × residual blocks (small-kernel, no upsampling)
        3) Conv for global residual fusion
        4) Log2(scaling_factor) × SubPixelConvolutionalBlock (×2 each)
        5) Large-kernel output conv (+Tanh)

    Args:
        in_channels (int): Input channels (e.g., 3 for RGB).
        large_kernel_size (int): Kernel size for head/tail convolutions.
        small_kernel_size (int): Kernel size used inside residual/upsampling blocks.
        n_channels (int): Feature width across the network.
        n_blocks (int): Number of residual blocks in the trunk.
        scaling_factor (int): Upscale factor (must be one of {2, 4, 8}).

    Returns:
        torch.Tensor: Super-resolved image of shape (B, in_channels, H*scale, W*scale).

    Notes:
        - The network uses a global skip connection around the residual stack.
        - Upsampling is performed by PixelShuffle via sub-pixel convolution blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        large_kernel_size: int = 9,
        small_kernel_size: int = 3,
        n_channels: int = 64,
        n_blocks: int = 16,
        scaling_factor: int = 4,
    ) -> None:
        """Build the canonical SRResNet generator network."""
        super().__init__()

        scaling_factor = int(scaling_factor)
        if scaling_factor not in {2, 4, 8}:
            raise AssertionError("The scaling factor must be 2, 4, or 8!")

        self.conv_block1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="PReLu",
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    kernel_size=small_kernel_size,
                    n_channels=n_channels,
                )
                for _ in range(n_blocks)
            ]
        )

        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=True,
            activation=None,
        )

        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[
                SubPixelConvolutionalBlock(
                    kernel_size=small_kernel_size,
                    n_channels=n_channels,
                    scaling_factor=2,
                )
                for _ in range(n_subpixel_convolution_blocks)
            ]
        )

        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=in_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="Tanh",
        )

    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        """Forward propagation through SRResNet."""
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block3(output)
        return sr_imgs


class Generator(nn.Module):
    """SRGAN generator wrapper around :class:`SRResNet`.

    Provides a thin adapter that:
        - Builds an internal :class:`SRResNet` with the given hyperparameters.
        - Optionally initializes weights from a pretrained SRResNet checkpoint.
        - Exposes a unified forward for SRGAN pipelines.

    Args:
        in_channels (int): Input channels (e.g., 3 for RGB).
        large_kernel_size (int): Kernel size for head/tail convolutions.
        small_kernel_size (int): Kernel size used inside residual/upsampling blocks.
        n_channels (int): Feature width across the network.
        n_blocks (int): Number of residual blocks in the trunk.
        scaling_factor (int): Upscale factor (must be one of {2, 4, 8}).

    Returns:
        torch.Tensor: Super-resolved image produced by the wrapped SRResNet.
    """

    def __init__(
        self,
        in_channels: int = 3,
        large_kernel_size: int = 9,
        small_kernel_size: int = 3,
        n_channels: int = 64,
        n_blocks: int = 16,
        scaling_factor: int = 4,
    ) -> None:
        super().__init__()

        self.net = SRResNet(
            in_channels=in_channels,
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )

    def initialize_with_srresnet(self, srresnet_checkpoint: str) -> None:
        """Initialize the generator weights from a pretrained SRResNet checkpoint."""
        srresnet = load_checkpoint(srresnet_checkpoint)["model"]
        self.net.load_state_dict(srresnet.state_dict())
        print("\nLoaded weights from pre-trained SRResNet.\n")

    def forward(self, lr_imgs: torch.Tensor) -> torch.Tensor:
        """Forward propagation via the wrapped SRResNet."""
        return self.net(lr_imgs)


__all__ = ["SRResNet", "Generator"]
