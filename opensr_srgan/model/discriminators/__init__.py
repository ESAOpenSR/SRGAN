"""Discriminator architectures for SRGAN."""

from .srgan_discriminator import Discriminator
from .patchgan import PatchGANDiscriminator
from .esrgan import ESRGANDiscriminator

__all__ = [
    "Discriminator",
    "PatchGANDiscriminator",
    "ESRGANDiscriminator",
]
