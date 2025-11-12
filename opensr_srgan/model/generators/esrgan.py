"""ESRGAN generator implementation built around RRDB blocks."""

from __future__ import annotations

from torch import Tensor, nn

from ..model_blocks import RRDB, make_upsampler

__all__ = ["ESRGANGenerator"]


class ESRGANGenerator(nn.Module):
    """
    ESRGAN generator network for single-image super-resolution.

    This implementation follows the design proposed in
    *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"*
    (Wang et al., 2018). It replaces traditional residual blocks with
    Residual-in-Residual Dense Blocks (RRDBs), omits batch normalization,
    and applies residual scaling for improved stability and perceptual quality.

    The architecture can be summarized as:
        Input → Conv(3×3) → [RRDB × N] → Conv(3×3)
        → (PixelShuffle upsampling × scale) → Conv(3×3) → LeakyReLU → Conv(3×3) → Output

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (e.g., 3 for RGB or 4 for RGB-NIR).
    out_channels : int or None, default=None
        Number of output channels. If ``None``, defaults to ``in_channels``.
    n_features : int, default=64
        Number of feature maps in the base convolutional layers.
    n_blocks : int, default=23
        Number of Residual-in-Residual Dense Blocks (RRDBs) stacked in the network body.
    growth_channels : int, default=32
        Number of intermediate growth channels used within each RRDB.
    res_scale : float, default=0.2
        Residual scaling factor applied to stabilize deep residual learning.
    scale : int, default=4
        Upscaling factor. Must be a power of two (1, 2, 4, 8, ...).

    Attributes
    ----------
    conv_first : nn.Conv2d
        Initial 3×3 convolutional layer that extracts shallow features from the LR input.
    body : nn.Sequential
        Sequential container of ``RRDB`` blocks performing deep feature extraction.
    conv_body : nn.Conv2d
        3×3 convolution applied after the RRDB stack to merge body features.
    upsampler : nn.Module
        PixelShuffle-based upsampling module (from ``make_upsampler``) for the configured scale.
        If ``scale == 1``, replaced by an identity mapping.
    conv_hr : nn.Conv2d
        3×3 convolution used for high-resolution refinement.
    activation : nn.LeakyReLU
        LeakyReLU activation (slope=0.2) applied after ``conv_hr``.
    conv_last : nn.Conv2d
        Final 3×3 projection layer mapping features back to output space.
    scale : int
        Model’s upscaling factor.
    n_blocks : int
        Number of RRDB blocks in the body.
    n_features : int
        Number of feature maps in the feature extraction layers.
    growth_channels : int
        Growth channels per RRDB.

    Raises
    ------
    ValueError
        If ``scale`` is not a power of two or if ``n_blocks < 1``.

    Examples
    --------
    >>> from opensr_srgan.model.generators.esrgan_generator import ESRGANGenerator
    >>> model = ESRGANGenerator(in_channels=3, scale=4)
    >>> x = torch.randn(1, 3, 64, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([1, 3, 256, 256])

    References
    ----------
    - Wang et al., *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*,
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
        """
        Forward pass of the ESRGAN generator.

        Parameters
        ----------
        x : torch.Tensor
            Low-resolution input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Super-resolved output tensor of shape (B, C, sH, sW),
            where `s` is the upscaling factor defined by ``self.scale``.

        Notes
        -----
        - The generator first encodes input features, processes them through
          a sequence of Residual-in-Residual Dense Blocks (RRDBs), and then
          performs upsampling via sub-pixel convolutions.
        - A long skip connection adds the shallow features from the input
          stem to the deep body output before upsampling.
        - The final activation is a LeakyReLU followed by a 3×3 convolution
          that projects features back to image space.
        - When ``scale == 1``, the upsampling block is replaced by an identity.
        """
        first = self.conv_first(x)
        trunk = self.body(first)
        body_out = self.conv_body(trunk)
        feat = first + body_out
        feat = self.upsampler(feat)
        feat = self.activation(self.conv_hr(feat))
        return self.conv_last(feat)
