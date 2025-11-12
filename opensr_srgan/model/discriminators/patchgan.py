"""PatchGAN discriminator adapted from the pix2pix/CycleGAN reference implementation.

This module provides a reimplementation of the *PatchGAN* discriminator
introduced by Isola et al. (pix2pix) and extended in Zhu et al. (CycleGAN).

The discriminator classifies overlapping image patches as real or fake rather
than entire images, allowing it to focus on fine-scale realism and local
texture consistency — an approach widely used in image-to-image translation
and super-resolution GANs.
"""
from __future__ import annotations

import functools
from torch import nn

__all__ = ["PatchGANDiscriminator"]


def get_norm_layer(norm_type: str = "instance"):
    """
    Return a normalization layer factory.

    Parameters
    ----------
    norm_type : {"batch", "instance", "none"}, default="instance"
        Normalization type to use. If ``"none"`` is selected, an identity
        layer is returned.

    Returns
    -------
    Callable[[int], nn.Module]
        Factory function producing normalization layers.

    Raises
    ------
    NotImplementedError
        If the requested normalization type is unsupported.

    Examples
    --------
    >>> norm = get_norm_layer("batch")
    >>> layer = norm(64)
    >>> print(layer)
    BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    """

    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    if norm_type == "none":

        def _identity(_channels: int):
            return nn.Identity()

        return _identity
    raise NotImplementedError(f"Normalization layer [{norm_type}] is not supported.")


class NLayerDiscriminator(nn.Module):
    """
    Defines the N-layer PatchGAN discriminator that classifies overlapping patches.

    This network consists of a series of strided convolutional layers that reduce
    spatial resolution while increasing channel depth, followed by a final
    1×1-patch output map representing per-patch realism scores.

    Parameters
    ----------
    input_nc : int
        Number of input channels (e.g., 3 for RGB, 4 for RGB-NIR).
    ndf : int, default=64
        Base number of feature channels.
    n_layers : int, default=3
        Number of downsampling layers.
    norm_layer : Callable[[int], nn.Module], default=nn.BatchNorm2d
        Normalization layer factory.

    Attributes
    ----------
    model : nn.Sequential
        Sequential container implementing the PatchGAN discriminator.

    Notes
    -----
    - Each layer uses a 4×4 convolution kernel with stride 2 (except final layers).
    - The final output map has spatial dimensions proportional to the input size.
    """
    def __init__(
        self, input_nc: int, ndf: int = 64, n_layers: int = 3, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):  # type: ignore[override]
        """
        Forward pass through the PatchGAN discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Realism score map of shape (B, 1, H/2ⁿ, W/2ⁿ),
            where higher values indicate more realistic patches.
        """
        return self.model(input)


class PatchGANDiscriminator(nn.Module):
    """
    High-level convenience wrapper for the N-layer PatchGAN discriminator.

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    n_layers : int, default=3
        Number of convolutional layers.
    norm_type : {"batch", "instance", "none"}, default="instance"
        Normalization strategy.

    Attributes
    ----------
    model : NLayerDiscriminator
        Underlying PatchGAN model.
    base_channels : int
        Number of base feature channels (default 64).
    kernel_size : int
        Kernel size for convolutions (default 4).
    n_layers : int
        Number of downsampling layers.

    Raises
    ------
    ValueError
        If `n_layers` < 1.

    Examples
    --------
    >>> disc = PatchGANDiscriminator(input_nc=3, n_layers=3)
    >>> x = torch.randn(4, 3, 128, 128)
    >>> y = disc(x)
    >>> y.shape
    torch.Size([4, 1, 14, 14])
    """
    def __init__(
        self,
        input_nc: int,
        n_layers: int = 3,
        norm_type: str = "instance",
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("PatchGAN discriminator requires at least one layer.")

        ndf = 64
        norm_layer = get_norm_layer(norm_type)
        self.model = NLayerDiscriminator(
            input_nc, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer
        )

        self.base_channels = ndf
        self.kernel_size = 4
        self.n_layers = n_layers

    def forward(self, input):  # type: ignore[override]
        """
        Forward pass through the PatchGAN discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Patch-level realism scores, shape (B, 1, H/2ⁿ, W/2ⁿ).
        """
        return self.model(input)
