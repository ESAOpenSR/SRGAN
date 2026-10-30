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


def test_prepare_perceptual_input_uses_fixed_idx():
    cfg = OmegaConf.create(
        {
            "Training": {
                "Losses": {
                    "perceptual_weight": 0.0,
                    "fixed_idx": [3, 1, 0],
                }
            },
            "Data": {"normalization": "identity"},
        }
    )
    loss_fn = GeneratorContentLoss(cfg)
    sr = torch.arange(1 * 5 * 2 * 2, dtype=torch.float32).reshape(1, 5, 2, 2)
    hr = sr + 100

    sr_3, hr_3 = loss_fn._prepare_perceptual_input(sr, hr)

    assert torch.equal(sr_3[:, 0], sr[:, 3])
    assert torch.equal(sr_3[:, 1], sr[:, 1])
    assert torch.equal(sr_3[:, 2], sr[:, 0])
    assert torch.equal(hr_3[:, 0], hr[:, 3])


@pytest.mark.parametrize("channels", [4, 6])
@pytest.mark.parametrize("explicit_null", [False, True])
def test_multispectral_default_shuffles_paired_bands_and_supervises_non_rgb(
    monkeypatch, channels, explicit_null,
):
    losses = {"perceptual_weight": 0.0}
    if explicit_null:
        losses["fixed_idx"] = None
    loss_fn = GeneratorContentLoss(OmegaConf.create({"Training": {"Losses": losses}}))
    permutations = [torch.arange(channels - 1, -1, -1), torch.arange(channels).roll(1)]
    calls = []

    def randperm(count, *, device):
        assert count == channels
        idx = permutations[len(calls)].to(device)
        calls.append(idx)
        return idx

    monkeypatch.setattr(torch, "randperm", randperm)
    sr = torch.arange(channels * 4, dtype=torch.float32).reshape(1, channels, 2, 2).requires_grad_()
    hr = sr.detach() + 100
    selections = []
    for _ in range(2):
        sr_3, hr_3 = loss_fn._prepare_perceptual_input(sr, hr)
        idx = calls[-1][:3]
        assert len(idx.unique()) == 3
        torch.testing.assert_close(sr_3, sr[:, idx])
        torch.testing.assert_close(hr_3, hr[:, idx])
        selections.append(sr_3)
    assert len(calls) == 2 and not torch.equal(selections[0], selections[1])
    selections[0].sum().backward()
    assert torch.all(sr.grad[:, channels - 1] > 0)
    assert torch.all(sr.grad[:, calls[0][3:]] == 0)


def test_three_band_perceptual_input_keeps_original_order(monkeypatch):
    loss_fn = GeneratorContentLoss(OmegaConf.create({
        "Training": {"Losses": {"perceptual_weight": 0.0}},
    }))
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: pytest.fail("RGB must stay in order"))
    sr, hr = torch.randn(1, 3, 2, 2), torch.randn(1, 3, 2, 2)
    selected_sr, selected_hr = loss_fn._prepare_perceptual_input(sr, hr)
    assert selected_sr is sr and selected_hr is hr


def test_prepare_perceptual_input_rejects_out_of_range_fixed_idx():
    cfg = OmegaConf.create(
        {
            "Training": {
                "Losses": {
                    "perceptual_weight": 0.0,
                    "fixed_idx": [0, 1, 5],
                }
            },
            "Data": {"normalization": "identity"},
        }
    )
    loss_fn = GeneratorContentLoss(cfg)
    sr = torch.zeros(1, 5, 2, 2)
    hr = torch.zeros(1, 5, 2, 2)

    with pytest.raises(ValueError, match="fixed_idx"):
        loss_fn._prepare_perceptual_input(sr, hr)


@pytest.mark.parametrize('enabled', [None, False, True])
def test_vgg_input_normalization_is_explicit_and_legacy_defaults_are_preserved(enabled):
    losses = {'perceptual_weight': 1.0, 'l1_weight': 0, 'sam_weight': 0}
    if enabled is not None:
        losses['vgg_input_normalization'] = enabled
    criterion = GeneratorContentLoss(OmegaConf.create({'Training': {'Losses': losses}}), testing=True)
    criterion.perceptual_model = torch.nn.Identity()
    prediction = torch.full((1, 3, 4, 4), .4, requires_grad=True)
    reference = torch.full_like(prediction, .2)
    expected_prediction, expected_reference = prediction, reference
    if enabled:
        mean = prediction.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = prediction.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        expected_prediction = (prediction - mean) / std
        expected_reference = (reference - mean) / std
    actual = criterion._perceptual_distance(prediction, reference, build_graph=True)
    torch.testing.assert_close(actual, torch.nn.functional.mse_loss(expected_prediction, expected_reference))
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
