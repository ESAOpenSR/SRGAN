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


def test_esrgan_discriminator_warns_about_n_blocks_override(capsys):
    """SRGAN model should inform users when ESRGAN ignores discriminator n_blocks."""

    config = OmegaConf.create(
        {
            "Model": {"in_bands": 3},
            "Generator": {"model_type": "srresnet", "scaling_factor": 4},
            "Discriminator": {"model_type": "esrgan", "n_blocks": 5},
        }
    )

    model = SRGAN_model.__new__(SRGAN_model)
    model.config = config
    model.mode = "train"
    model.generator = None
    model.discriminator = None

    model.get_models("train")

    captured = capsys.readouterr()
    assert (
        "[Discriminator:esrgan] Ignoring unsupported configuration options: n_blocks." in captured.out
    )
