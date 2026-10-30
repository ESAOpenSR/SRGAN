import pytest
import torch
from omegaconf import OmegaConf

from opensr_srgan.model.loss.local_gram import local_gram_distance
from opensr_srgan.model.loss import GeneratorContentLoss


def test_gram_matches_within_tile_statistics_and_preserves_spatial_regions():
    target = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
    # Reverse spatial order within each tile: identical local feature statistics.
    shuffled = target.reshape(1, 1, 2, 4, 2, 4).flip((3, 5)).reshape_as(target)
    torch.testing.assert_close(local_gram_distance(shuffled, target, 2), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert local_gram_distance(target.flip(-1), target, 2) > 0


def test_gram_has_only_prediction_gradients_and_safe_bfloat_statistics():
    prediction = torch.rand(2, 4, 8, 8, dtype=torch.bfloat16, requires_grad=True)
    target = torch.rand_like(prediction, requires_grad=True)
    loss = local_gram_distance(prediction, target, 2)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert prediction.grad.abs().sum() > 0 and torch.isfinite(prediction.grad).all()
    assert target.grad is None
    with pytest.raises(ValueError):
        local_gram_distance(prediction, target, 3)


def test_texture_only_content_objective_reaches_rgb_and_freezes_vgg():
    cfg = OmegaConf.create({"Training": {"Losses": {
        "perceptual_weight": 0, "l1_weight": 0, "sam_weight": 0,
        "texture_weight": 0.2, "texture_grid": 4, "fixed_idx": [0, 1, 2],
        "vgg_before_activation": True,
    }}, "TruncatedVGG": {"i": 3, "j": 4}})
    criterion = GeneratorContentLoss(cfg, testing=True)
    prediction = torch.rand(1, 4, 32, 32, requires_grad=True)
    target = torch.rand_like(prediction)
    loss, metrics = criterion.return_loss(prediction, target)
    torch.testing.assert_close(loss.detach(), metrics["texture"] * 0.2)
    loss.backward()
    assert prediction.grad[:, :3].abs().sum() > 0
    assert prediction.grad[:, 3].abs().sum() == 0
    assert all(p.grad is None for p in criterion.perceptual_model.parameters())
