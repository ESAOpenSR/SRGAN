"""Tests for loss functions and loss-related utilities."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from opensr_srgan.model.loss import GeneratorContentLoss


@pytest.fixture
def loss_config():
    """Minimal loss configuration."""
    return OmegaConf.create(
        {
            "Training": {
                "Losses": {
                    "pixel_loss_weight": 1.0,
                    "perceptual_loss_weight": 1.0,
                    "content_loss_type": "l1",
                }
            },
            "Model": {"in_bands": 3},
        }
    )


def test_generator_content_loss_initialization(loss_config):
    """Test GeneratorContentLoss can be initialized."""
    loss_fn = GeneratorContentLoss(loss_config)
    assert loss_fn is not None


def test_generator_content_loss_forward(loss_config):
    """Test GeneratorContentLoss forward pass."""
    loss_fn = GeneratorContentLoss(loss_config)
    sr = torch.randn(2, 3, 64, 64)
    hr = torch.randn(2, 3, 64, 64)

    result, metrics = loss_fn.return_loss(sr, hr)
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert isinstance(metrics, dict)
    assert result.shape == torch.Size([])


def test_generator_content_loss_metrics(loss_config):
    """Test GeneratorContentLoss metrics computation."""
    loss_fn = GeneratorContentLoss(loss_config)
    sr = torch.randn(2, 3, 64, 64)
    hr = torch.randn(2, 3, 64, 64)

    metrics = loss_fn.return_metrics(sr, hr, prefix="test_")
    assert isinstance(metrics, dict)
    # Should have at least one metric
    assert len(metrics) > 0
    # All keys should have prefix
    for key in metrics.keys():
        assert key.startswith("test_")


def test_generator_content_loss_with_different_weights(loss_config):
    """Test loss computation with different weight configurations."""
    loss_config.Training.Losses.pixel_loss_weight = 2.0
    loss_config.Training.Losses.perceptual_loss_weight = 0.5

    loss_fn = GeneratorContentLoss(loss_config)
    sr = torch.randn(2, 3, 64, 64, requires_grad=True)
    hr = torch.randn(2, 3, 64, 64)

    result, metrics = loss_fn.return_loss(sr, hr)
    assert result.requires_grad
    assert result.item() > 0


def test_generator_content_loss_l2_variant(loss_config):
    """Test GeneratorContentLoss with L2 pixel loss."""
    loss_config.Training.Losses.content_loss_type = "l2"

    loss_fn = GeneratorContentLoss(loss_config)
    sr = torch.randn(2, 3, 64, 64)
    hr = torch.randn(2, 3, 64, 64)

    result, metrics = loss_fn.return_loss(sr, hr)
    assert result is not None
    assert result.item() > 0
