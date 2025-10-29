"""Basic instantiation tests for discriminator architectures."""

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
_ = pytest.importorskip("pytorch_lightning")
from torch import nn  # noqa: E402

from opensr_srgan.model.SRGAN import SRGAN_model  # noqa: E402
from opensr_srgan.model.discriminators import (  # noqa: E402
    Discriminator,
    ESRGANDiscriminator,
    PatchGANDiscriminator,
)


@pytest.mark.parametrize(
    "discriminator_cls, kwargs",
    [
        (Discriminator, {}),
        (PatchGANDiscriminator, {"input_nc": 3}),
        (ESRGANDiscriminator, {}),
    ],
)
def test_discriminator_can_be_instantiated(discriminator_cls, kwargs):
    """Ensure discriminator classes can be constructed with the provided arguments."""

    instance = discriminator_cls(**kwargs)
    assert isinstance(instance, nn.Module)

