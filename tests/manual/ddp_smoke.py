"""Opt-in, two-CUDA-device integration check; see docs/tests.md.

Run with torchrun, not pytest. Uses the real model, optimizers and training hooks.
"""

import os
import tempfile
import math

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from opensr_srgan.model.SRGAN import SRGAN_model
from opensr_srgan.utils.build_trainer_kwargs import build_lightning_kwargs

EPOCHS = 5
BATCHES_PER_EPOCH = 20
PRETRAIN_BATCHES = 25
WARMUP_BATCHES = 30
RAMP_STEPS = 40
GENERATOR_LR = 0.001


def parameters(module):
    return torch.cat([p.detach().flatten() for p in module.parameters()])


def assert_synchronized(vector):
    assert torch.isfinite(vector).all(), "Non-finite model/EMA parameters"
    gathered = [torch.empty_like(vector) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, vector.contiguous())
    for other in gathered[1:]:
        torch.testing.assert_close(gathered[0], other, rtol=1e-5, atol=1e-7)


class CheckUpdates(pl.Callback):
    def __init__(self):
        self.phases = []
        self.weights = []
        self.epochs = []
        self.validation_epochs = 0
        self.validation_batches = 0

    def on_train_start(self, trainer, model):
        assert trainer.world_size == 2
        assert model.automatic_optimization is False
        assert model._warmup_scheduler_g is not None, "LR warmup must be enabled"

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        completed = len(self.phases)
        assert completed == trainer.current_epoch * BATCHES_PER_EPOCH + batch_idx
        assert model._warmup_scheduler_g.last_epoch == min(completed, WARMUP_BATCHES)
        expected_lr = GENERATOR_LR * min(1.0, max(0.05, (completed + 1) / WARMUP_BATCHES))
        for group in trainer.optimizers[1].param_groups:
            assert math.isclose(group["lr"], expected_lr, rel_tol=1e-6), "Incorrect warmup LR"
        self.phases.append(model._pretrain_check())
        self.before_g = parameters(model.generator).clone()
        self.before_d = parameters(model.discriminator).clone()
        self.before_step = trainer.global_step

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        assert torch.isfinite(outputs["loss"]).all(), "Non-finite training loss"
        g = parameters(model.generator)
        d = parameters(model.discriminator)
        assert not torch.equal(g, self.before_g), "Generator did not update"
        if self.phases[-1]:
            assert torch.equal(d, self.before_d), "Discriminator updated during pretraining"
            assert trainer.global_step == self.before_step + 1
        else:
            assert not torch.equal(d, self.before_d), "Discriminator did not update"
            assert trainer.global_step == self.before_step + 2
        assert_synchronized(g)
        assert_synchronized(d)
        assert model.ema.num_updates == len(self.phases)
        assert_synchronized(torch.cat([
            p.flatten() for p in model.ema.shadow_params.values()
        ]))
        self.weights.append(float(trainer.callback_metrics["training/adv_loss_weight"]))

    def on_validation_start(self, trainer, model):
        self.before_validation = parameters(model.generator).clone()
        assert not model._ema_applied

    def on_validation_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx=0):
        assert model._ema_applied, "Validation must use EMA weights"
        for name, parameter in model.generator.named_parameters():
            torch.testing.assert_close(parameter, model.ema.shadow_params[name])
        self.validation_batches += 1

    def on_validation_end(self, trainer, model):
        self.validation_epochs += 1
        assert not model._ema_applied, "Live weights must be restored after validation"
        torch.testing.assert_close(parameters(model.generator), self.before_validation, rtol=0, atol=0)
        for metric in ("val_metrics/l1", "validation/DISC_adversarial_loss"):
            assert torch.isfinite(trainer.callback_metrics[metric]).all()
        assert model._plateau_scheduler_g.last_epoch == self.validation_epochs
        assert model._plateau_scheduler_d.last_epoch == self.validation_epochs

    def on_train_epoch_end(self, trainer, model):
        self.epochs.append(trainer.current_epoch)
        assert len(self.phases) == len(self.epochs) * BATCHES_PER_EPOCH
        assert self.validation_epochs == len(self.epochs)
        print(f"rank={trainer.global_rank} epoch={trainer.current_epoch + 1}/{EPOCHS} "
              f"batches={len(self.phases)} optimizer_steps={trainer.global_step} "
              f"g_lr={trainer.optimizers[1].param_groups[0]['lr']:.6f} "
              f"adv_weight={self.weights[-1]:.3f}", flush=True)

    def on_train_end(self, trainer, model):
        total = EPOCHS * BATCHES_PER_EPOCH
        assert self.epochs == list(range(EPOCHS))
        assert self.validation_batches == EPOCHS * 2
        assert self.phases == [True] * PRETRAIN_BATCHES + [False] * (total - PRETRAIN_BATCHES)
        assert self.weights[:PRETRAIN_BATCHES] == [0.0] * PRETRAIN_BATCHES
        assert 0 < self.weights[PRETRAIN_BATCHES] < 0.5
        assert all(a <= b for a, b in zip(self.weights, self.weights[1:]))
        # The ramp must finish well before training ends and stay at its maximum.
        assert self.weights[-BATCHES_PER_EPOCH:] == [0.5] * BATCHES_PER_EPOCH
        assert model._warmup_scheduler_g.last_epoch == WARMUP_BATCHES
        assert trainer.global_step == PRETRAIN_BATCHES + 2 * (total - PRETRAIN_BATCHES)
        print(f"PASS rank={trainer.global_rank}: {EPOCHS} epochs, {total} batches; "
              "pretraining, LR warmup, adversarial ramp, validation, EMA and "
              "synchronized parameters", flush=True)


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or torch.cuda.device_count() < 2:
        raise SystemExit("Requires torchrun --nproc_per_node=2 and two visible CUDA GPUs")
    pl.seed_everything(123, workers=True)
    torch.set_num_threads(1)
    config = OmegaConf.create("""
Model:
  in_bands: 1
Generator:
  model_type: SRResNet
  block_type: standard
  n_channels: 4
  n_blocks: 1
  large_kernel_size: 3
  small_kernel_size: 3
  scaling_factor: 2
Discriminator:
  model_type: patchgan
  n_blocks: 1
  use_spectral_norm: false
Training:
  device: cuda
  gpus: [0, 1]
  max_epochs: 5
  val_check_interval: 1.0
  limit_val_batches: 2
  pretrain_g_only: true
  g_pretrain_steps: 25
  adv_loss_ramp_steps: 40
  label_smoothing: false
  EMA: {enabled: true, decay: 0.9, use_num_updates: true, update_after_step: 0}
  Losses:
    adv_loss_beta: 0.5
    adv_loss_schedule: linear
    adv_loss_type: bce
    relativistic_average_d: false
    r1_gamma: 0.0
    l1_weight: 1.0
    sam_weight: 0.0
    perceptual_weight: 0.0
    tv_weight: 0.0
    ssim_win: 3
Optimizers:
  optim_g_lr: 0.001
  optim_d_lr: 0.0005
  gradient_clip_val: 0.5
Schedulers:
  metric: val_metrics/l1
  metric_d: validation/DISC_adversarial_loss
  g_warmup_steps: 30
  g_warmup_type: linear
  patience_g: 100
  patience_d: 100
Logging:
  wandb: {enabled: false}
  num_val_images: 0
""")
    assert config.Training.max_epochs == EPOCHS
    assert config.Training.g_pretrain_steps == PRETRAIN_BATCHES
    assert config.Training.adv_loss_ramp_steps == RAMP_STEPS
    assert config.Schedulers.g_warmup_steps == WARMUP_BATCHES
    assert config.Optimizers.optim_g_lr == GENERATOR_LR
    model = SRGAN_model(config=config, mode="train")
    # Identical seeded dataset on each rank; Lightning's distributed sampler
    # gives each rank different examples (80 / 2 ranks / batch size 2 = 20 steps).
    lr = torch.rand(BATCHES_PER_EPOCH * 4, 1, 4, 4)
    hr = torch.rand(BATCHES_PER_EPOCH * 4, 1, 8, 8)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2, num_workers=0)
    validation = DataLoader(TensorDataset(torch.rand(8, 1, 4, 4), torch.rand(8, 1, 8, 8)),
                            batch_size=2, num_workers=0)
    kwargs, fit_kwargs = build_lightning_kwargs(config, None, None, None)
    assert kwargs["strategy"] == "ddp_find_unused_parameters_true"
    kwargs.update(
        logger=False, callbacks=[CheckUpdates()], precision="32-true",
        enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, num_sanity_val_steps=0, limit_train_batches=BATCHES_PER_EPOCH,
    )
    with tempfile.TemporaryDirectory(prefix="srgan-ddp-") as output_dir:
        print(f"torch={torch.__version__}, lightning={pl.__version__}, "
              f"GPU={torch.cuda.get_device_name(int(os.environ['LOCAL_RANK']))}", flush=True)
        trainer = pl.Trainer(default_root_dir=output_dir, **kwargs)
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=validation, **fit_kwargs)


if __name__ == "__main__":
    main()
