"""Conditional GAN generator with stochastic latent modulation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..model_blocks import make_upsampler

__all__ = ["StochasticGenerator", "ConditionalGANGenerator"]


class NoiseResBlock(nn.Module):
    """Residual convolutional block with latent noise modulation.

    Introduces stochastic variability into intermediate feature maps by conditioning
    each residual block on a latent noise vector. The latent code is transformed
    through a small MLP to produce per-channel affine parameters (γ, β) that modulate
    the convolutional activations in a style-based manner:
    ```
    y = Conv(x) * (1 + γ) + β
    ```
    This enables controlled diversity and more expressive texture synthesis while
    maintaining residual stability via a configurable scaling factor.

    Args:
        n_channels (int): Number of feature channels in the block.
        kernel_size (int): Convolutional kernel size for both layers.
        noise_dim (int): Dimensionality of the latent noise vector.
        res_scale (float, optional): Residual scaling factor (typically 0.2).
            Controls the contribution of the residual path to the output.
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        noise_dim: int,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.res_scale = res_scale
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_dim, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, 2 * n_channels),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the noise-modulated residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C, H, W).
        noise : torch.Tensor
            Latent noise vector of shape (B, D), where D = `noise_dim`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C, H, W) with per-channel affine
            modulation applied as:
            ``y = Conv(x) * (1 + γ) + β``

        Notes
        -----
        - The latent vector is transformed via an internal MLP to produce
          (γ, β) modulation parameters.
        - A residual scaling factor (`res_scale`) stabilizes training
          by controlling the magnitude of the residual branch.
        """
        style = self.noise_mlp(noise)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        out = self.conv1(x)
        out = out * (1 + gamma) + beta
        out = self.act(out)
        out = self.conv2(out)
        return x + self.res_scale * out


class StochasticGenerator(nn.Module):
    """Stochastic generator with latent noise modulation for super-resolution.

    Extends a standard SR generator by injecting stochastic latent noise through
    `NoiseResBlock`s, enabling diverse texture generation conditioned on the same
    low-resolution (LR) input. When no latent vector is provided, one is sampled
    internally from a standard normal distribution, allowing both deterministic
    and stochastic inference modes.

    The architecture follows a residual backbone with:
        - A wide receptive-field head convolution.
        - Multiple noise-modulated residual blocks.
        - A tail convolution with learnable upsampling.
        - Configurable scaling factor (×2, ×4, or ×8).

    Args:
        in_channels (int): Number of input channels (e.g., RGB+NIR = 4 or 6).
        n_channels (int): Base number of feature channels in the generator.
        n_blocks (int): Number of noise-modulated residual blocks.
        small_kernel (int): Kernel size for body convolutions.
        large_kernel (int): Kernel size for head/tail convolutions.
        scale (int): Upscaling factor (must be one of {2, 4, 8}).
        noise_dim (int): Dimensionality of the latent vector z.
        res_scale (float): Residual scaling factor for block stability.

    Attributes:
        noise_dim (int): Dimensionality of latent vector z.
        head (nn.Sequential): Initial convolutional stem.
        body (nn.ModuleList): Sequence of `NoiseResBlock`s.
        upsampler (nn.Module): PixelShuffle-based upsampling module.
        tail (nn.Conv2d): Final convolution projecting to output space.

    Example:
        >>> g = StochasticGenerator(in_channels=3, scale=4)
        >>> lr = torch.randn(1, 3, 64, 64)
        >>> sr, noise = g(lr, return_noise=True)
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_channels: int = 96,
        n_blocks: int = 16,
        small_kernel: int = 3,
        large_kernel: int = 9,
        scale: int = 4,
        noise_dim: int = 128,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()

        if scale not in {2, 4, 8}:
            raise ValueError("scale must be one of {2, 4, 8}")

        self.noise_dim = noise_dim
        self.scale = scale

        padding_large = large_kernel // 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, large_kernel, padding=padding_large),
            nn.PReLU(),
        )

        self.body = nn.ModuleList(
            [
                NoiseResBlock(n_channels, small_kernel, noise_dim, res_scale)
                for _ in range(n_blocks)
            ]
        )
        self.body_tail = nn.Conv2d(
            n_channels,
            n_channels,
            small_kernel,
            padding=small_kernel // 2,
        )
        self.upsampler = make_upsampler(n_channels, scale)
        self.tail = nn.Conv2d(
            n_channels, in_channels, large_kernel, padding=padding_large
        )

    def sample_noise(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Sample a latent noise tensor consistent with the generator configuration.

        Parameters
        ----------
        batch_size : int
            Number of noise vectors to generate.
        device : torch.device, optional
            Device on which to allocate the tensor. Defaults to the model's current device.
        dtype : torch.dtype, optional
            Tensor dtype. Defaults to the model's parameter dtype.

        Returns
        -------
        torch.Tensor
            Random latent tensor sampled from a standard normal distribution,
            shape (B, D) where B = `batch_size` and D = `self.noise_dim`.

        Notes
        -----
        - Used for stochastic generation when no latent vector is provided to ``forward()``.
        - Ensures type and device consistency with the current model parameters.
        - The resulting noise can be reused to reproduce identical stochastic outputs.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.randn(batch_size, self.noise_dim, device=device, dtype=dtype)

    def forward(
        self,
        lr: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the stochastic super-resolution generator.

        Parameters
        ----------
        lr : torch.Tensor
            Low-resolution input tensor of shape (B, C, H, W).
        noise : torch.Tensor, optional
            Latent noise tensor of shape (B, D), where D = `noise_dim`.
            If None, a random vector is sampled internally.
        return_noise : bool, default=False
            If True, returns both the super-resolved image and the
            latent noise used for generation.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            - ``sr``: Super-resolved output image of shape (B, C, sH, sW),
              where s is the upscaling factor.
            - Optionally, ``noise``: The latent vector used (if `return_noise=True`).

        Notes
        -----
        - The latent code is broadcast and applied to all residual blocks.
        - Supports both deterministic (fixed noise) and stochastic (random noise)
          inference modes.
        - The upsampling is performed via sub-pixel convolution (PixelShuffle)
          defined in `make_upsampler()`.
        """
        if noise is None:
            noise = torch.randn(
                lr.size(0),
                self.noise_dim,
                device=lr.device,
                dtype=lr.dtype,
            )

        features = self.head(lr)
        residual = features
        for block in self.body:
            residual = block(residual, noise)
        residual = self.body_tail(residual)
        features = features + residual
        features = self.upsampler(features)
        sr = self.tail(features)

        if return_noise:
            return sr, noise
        return sr


# Backwards compatibility ---------------------------------------------------
ConditionalGANGenerator = StochasticGenerator
