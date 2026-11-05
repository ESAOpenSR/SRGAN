"""Shuffled supervision must reach NIR while validation stays reproducible."""
import pytest
import torch
from omegaconf import OmegaConf

from opensr_srgan.model.loss import GeneratorContentLoss


def config(**changes):
    losses = {
        "perceptual_weight": 1.0, "texture_weight": 0.2, "texture_grid": 4,
        "l1_weight": 0.0, "sam_weight": 0.0, "fixed_idx": None,
        "validation_fixed_idx": [0, 1, 2], "report_band_metrics": True,
        "vgg_before_activation": True,
    }
    losses.update(changes)
    return OmegaConf.create({"Training": {"Losses": losses},
                            "TruncatedVGG": {"i": 3, "j": 4}})


@pytest.mark.parametrize("perceptual,texture", [(1.0, 0.0), (0.0, 0.2)])
def test_shuffled_perceptual_and_gram_both_supervise_nir(monkeypatch, perceptual, texture):
    criterion = GeneratorContentLoss(config(perceptual_weight=perceptual,
                                             texture_weight=texture), testing=True)
    monkeypatch.setattr(torch, "randperm", lambda count, *, device:
                        torch.tensor([3, 1, 0, 2], device=device))
    prediction = torch.rand(1, 4, 32, 32, requires_grad=True)
    reference = torch.rand_like(prediction)
    loss, _ = criterion.return_loss(prediction, reference)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 3].abs().sum() > 0
    assert prediction.grad[:, 2].abs().sum() == 0
    assert all(parameter.grad is None for parameter in criterion.perceptual_model.parameters())


def test_validation_is_deterministic_and_nir_metrics_are_separate(monkeypatch):
    criterion = GeneratorContentLoss(config(), testing=True)
    reference = torch.rand(1, 4, 32, 32) * 0.5
    prediction = reference.clone()
    prediction[:, 3] += 0.1
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs:
                        pytest.fail("Validation must not sample a random triplet"))
    rng = torch.random.get_rng_state().clone()
    first = criterion.return_metrics(prediction, reference)
    second = criterion.return_metrics(prediction, reference)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        assert not first[key].requires_grad
    assert first["l1_rgb"] == 0
    torch.testing.assert_close(first["l1_nir"], torch.tensor(0.1))
    assert first["perceptual"] == 0
    assert first["texture"] == 0
    assert first["perceptual_nir"] > 0
    assert first["ssim_rgb"] > first["ssim_nir"]


def test_validation_override_does_not_restrict_training_triplets():
    criterion = GeneratorContentLoss(config(perceptual_weight=0, texture_weight=0))
    sr = torch.arange(4.0).reshape(1, 4, 1, 1)
    hr = sr + 10
    torch.manual_seed(12)
    selected = [criterion._prepare_perceptual_input(sr, hr) for _ in range(32)]
    assert any(3 in a.flatten().tolist() for a, _ in selected)
    assert len({tuple(a.flatten().tolist()) for a, _ in selected}) > 1
    for a, b in selected:
        torch.testing.assert_close(b - a, torch.full_like(a, 10))
    a, b = criterion._prepare_perceptual_input(sr, hr, for_metrics=True)
    torch.testing.assert_close(a, sr[:, :3])
    torch.testing.assert_close(b, hr[:, :3])
