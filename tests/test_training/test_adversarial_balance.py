"""Check global RaGAN means, patch-normalized R1, and ranking diagnostics."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from opensr_srgan.model.adversarial import (
    critic_auc,
    image_logits,
    normalized_patch_r1,
    relativistic_means,
)


def _objective(weight, rank=None, global_batch=False, generator=False):
    real = torch.tensor([[1., -2.], [.5, 3.], [4., -.1], [-1., 2.]])
    fake = torch.tensor([[-2., 1.], [3., .2], [.4, -1.], [2., -.3]])
    if rank is not None:
        real, fake = real.chunk(2)[rank], fake.chunk(2)[rank]
    a = real * (1.3 if generator else weight)
    b = fake * weight
    am, bm = relativistic_means(a, b, global_batch=global_batch)
    return F.binary_cross_entropy_with_logits(a - bm, torch.zeros_like(a) if generator else torch.ones_like(a)) + F.binary_cross_entropy_with_logits(b - am, torch.ones_like(b) if generator else torch.zeros_like(b))


def _distributed_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    results = []
    for generator in (False, True):
        weight = torch.tensor(.7, requires_grad=True)
        loss = _objective(weight, rank, True, generator)
        loss.backward()
        result = torch.stack((loss.detach(), weight.grad))
        dist.all_reduce(result)
        results.append(result / 2)
    if rank == 0:
        torch.save(results, output)
    dist.destroy_process_group()


def test_distributed_ragan_matches_global_batch_values_and_gradients(tmp_path):
    output = str(tmp_path / "results.pt")
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"), output), nprocs=2)
    results = torch.load(output, weights_only=True)
    for generator, result in zip((False, True), results):
        weight = torch.tensor(.7, requires_grad=True)
        loss = _objective(weight, generator=generator)
        loss.backward()
        torch.testing.assert_close(result, torch.stack((loss.detach(), weight.grad)))


class _ToyPatchCritic(torch.nn.Module):
    def __init__(self, repeats):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self.repeats = repeats

    def forward(self, image):
        return (image.flatten(1).mean(1) * self.weight).reshape(-1, 1, 1, 1).expand(-1, 1, self.repeats, self.repeats)


def test_r1_is_patch_count_independent_and_differentiable():
    inputs = torch.randn(2, 4, 8, 8)
    for repeats in (1, 5):
        critic = _ToyPatchCritic(repeats)
        penalty = normalized_patch_r1(critic, inputs)
        penalty.backward()
        torch.testing.assert_close(penalty, torch.tensor(2 / 256))
        torch.testing.assert_close(critic.weight.grad, torch.tensor(2 / 256))
        assert not inputs.requires_grad


def test_local_relativistic_statistics_and_image_scores_keep_gradients_local():
    real = torch.tensor([[[[1.0, 3.0]]]], requires_grad=True)
    fake = torch.tensor([[[[-2.0, 4.0]]]], requires_grad=True)

    real_mean, fake_mean = relativistic_means(real, fake, global_batch=False)
    global_real_mean, global_fake_mean = relativistic_means(
        real, fake, global_batch=True
    )
    objective = (real - fake_mean).square().mean() + (fake - real_mean).square().mean()
    objective.backward()

    torch.testing.assert_close(real_mean, torch.tensor(2.0))
    torch.testing.assert_close(fake_mean, torch.tensor(1.0))
    torch.testing.assert_close(global_real_mean, real_mean)
    torch.testing.assert_close(global_fake_mean, fake_mean)
    assert torch.isfinite(real.grad).all()
    assert torch.isfinite(fake.grad).all()
    torch.testing.assert_close(
        image_logits(torch.tensor([[[[1.0, 3.0]]], [[[2.0, 6.0]]]])),
        torch.tensor([2.0, 4.0]),
    )


@pytest.mark.parametrize("real,fake,expected", [
    ([1000., 2000.], [-1000., -2000.], 1.), ([-2., -1.], [1., 2.], 0.),
    ([2., 2.], [2., 2.], .5), ([1., 2.], [0., 1.], .875),
])
def test_critic_auc_handles_extreme_logits_and_ties(real, fake, expected):
    assert critic_auc(torch.tensor(real), torch.tensor(fake)).item() == expected


@pytest.mark.parametrize("real,fake", [(torch.empty(0), torch.ones(1)), (torch.ones(1), torch.empty(0))])
def test_critic_auc_requires_scores_from_both_classes(real, fake):
    with pytest.raises(ValueError, match="requires real and generated"):
        critic_auc(real, fake)


def test_joint_rgb_nir_selection_and_fidelity_guards():
    from opensr_srgan.model.SRGAN import SRGAN_model
    from opensr_srgan.utils.training_callbacks import texture_checkpoint_eligible
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "Training": {"Losses": {"texture_selection_nir_weight": .5,
            "texture_selection_rgb_reference": 2., "texture_selection_nir_reference": 4.,
            "texture_selection_l1_weight": 0., "texture_selection_max_l1_rgb": .03,
            "texture_selection_max_perceptual_nir": 4.1}},
        "Schedulers": {"texture_selection_metric": "val_metrics/multiband_selection"},
        "Logging": {"local_images": False, "num_val_images": 0, "wandb": {"enabled": False}},
    })
    metrics = {"val_metrics/perceptual": torch.tensor(2.), "val_metrics/perceptual_nir": torch.tensor(3.2),
               "val_metrics/l1": torch.tensor(.02), "val_metrics/l1_rgb": torch.tensor(.01)}
    logs = {}
    model = SimpleNamespace(config=config, forward=lambda x: x, _pretrain_check=lambda: True,
        content_loss_criterion=SimpleNamespace(return_metrics=lambda *a, **kw: metrics),
        log=lambda key, value, **kw: logs.update({key: value}),
        pretrain_g_only=False, g_pretrain_steps=0, adv_loss_ramp_steps=0, global_step=100)
    SRGAN_model.validation_step(model, (torch.ones(1, 4, 4, 4), torch.ones(1, 4, 4, 4)), 0)
    torch.testing.assert_close(logs["val_metrics/multiband_selection"], torch.tensor(.9))
    model._pretrain_check = lambda: False
    assert texture_checkpoint_eligible(model, metrics)
    metrics["val_metrics/perceptual_nir"] = torch.tensor(4.2)
    assert not texture_checkpoint_eligible(model, metrics)
