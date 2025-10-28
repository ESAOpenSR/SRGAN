"""Basic instantiation and factory tests for generator architectures."""

from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402  (import after torch availability check)

from opensr_srgan.model.generators import (  # noqa: E402
    ConditionalGANGenerator,
    FlexibleGenerator,
    Generator,
    SRResNet,
    StochasticGenerator,
    build_generator,
)


@pytest.mark.parametrize(
    "generator_cls, kwargs",
    [
        (SRResNet, {}),
        (Generator, {}),
        (FlexibleGenerator, {}),
        (StochasticGenerator, {}),
    ],
)
def test_generator_can_be_instantiated(generator_cls, kwargs):
    """Ensure generator classes can be constructed with default arguments."""

    instance = generator_cls(**kwargs)
    assert isinstance(instance, nn.Module)


def test_conditional_alias_points_to_stochastic_generator():
    """Legacy alias should reference the stochastic generator class."""

    assert ConditionalGANGenerator is StochasticGenerator


@pytest.mark.parametrize(
    "generator_cfg, expected_cls",
    [
        (
            {
                "model_type": "SRResNet",
                "block_type": "standard",
                "large_kernel_size": 9,
                "small_kernel_size": 3,
                "n_channels": 64,
                "n_blocks": 16,
                "scaling_factor": 4,
            },
            Generator,
        ),
        (
            {
                "model_type": "SRResNet",
                "block_type": "rcab",
                "large_kernel_size": 9,
                "small_kernel_size": 3,
                "n_channels": 96,
                "n_blocks": 32,
                "scaling_factor": 8,
            },
            FlexibleGenerator,
        ),
        (
            {
                "model_type": "stochastic_gan",
                "large_kernel_size": 9,
                "small_kernel_size": 3,
                "n_channels": 96,
                "n_blocks": 16,
                "scaling_factor": 4,
            },
            StochasticGenerator,
        ),
        (
            {
                "model_type": "rrdb",  # legacy direct variant
                "large_kernel_size": 9,
                "small_kernel_size": 3,
                "n_channels": 96,
                "n_blocks": 32,
                "scaling_factor": 8,
            },
            FlexibleGenerator,
        ),
    ],
)
def test_build_generator_from_config(generator_cfg, expected_cls):
    """Factory should create the appropriate generator variant for each config."""

    config = OmegaConf.create(
        {
            "Model": {"in_bands": 4},
            "Generator": generator_cfg,
        }
    )

    generator = build_generator(config)
    assert isinstance(generator, expected_cls)
