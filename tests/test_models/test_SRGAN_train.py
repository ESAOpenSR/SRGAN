import types
from contextlib import contextmanager
from pathlib import Path

import pytorch_lightning as pl
import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from opensr_srgan.model import SRGAN
from opensr_srgan.model import training_step_PL
from opensr_srgan.model.SRGAN import SRGAN_model


class LoggerMixin:
    def __init__(self):
        self.logged = {}

    def log(self, key, value, **kwargs):
        self.logged[key] = value

    def _log_generator_content_loss(self, loss):
        self.log("generator/content_loss", loss)

    def _log_adv_loss_weight(self, weight):
        self.log("training/adv_loss_weight", weight)


class DummyContentLoss:
    def __init__(self):
        self.calls = 0

    def return_loss(self, sr, hr):
        self.calls += 1
        l1 = torch.nn.functional.l1_loss(sr, hr)
        return 2.0 * l1, {"l1": l1.detach()}


class DummyConfig:
    class Optimizers:
        gradient_clip_val = 0.0

    class Schedulers:
        pass


class TrainingHarness(LoggerMixin):
    def __init__(self, pretrain=False):
        super().__init__()
        self.pretrain_mode = pretrain
        self.content_loss_criterion = DummyContentLoss()
        self.adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss()
        self.adv_target = 0.9
        self.generator = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.discriminator = torch.nn.Conv2d(1, 1, kernel_size=1)
        self._ema_update_after_step = 0
        self.ema = None
        self.global_step = 0
        self.config = DummyConfig()
        self.trainer = types.SimpleNamespace(precision_plugin=None)

    def forward(self, lr):
        return self.generator(lr)

    def _pretrain_check(self):
        return self.pretrain_mode

    def _compute_adv_loss_weight(self):
        return torch.tensor(0.5)

    def _adv_loss_weight(self):
        return self._compute_adv_loss_weight()

    def manual_backward(self, loss):
        loss.backward()

    def optimizers(self):
        opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=0.1)
        opt_g = torch.optim.SGD(self.generator.parameters(), lr=0.1)
        return opt_d, opt_g

    def toggle_optimizer(self, *args, **kwargs):
        return None

    def untoggle_optimizer(self, *args, **kwargs):
        return None


def _sample_batch():
    lr = torch.ones(1, 1, 2, 2, requires_grad=True)
    hr = torch.ones(1, 1, 2, 2)
    return lr, hr


def test_setup_lightning_configures_manual_step_for_pl2(monkeypatch):
    monkeypatch.setattr(SRGAN.pl, "__version__", "2.2.0")
    model = SRGAN.SRGAN_model.__new__(SRGAN.SRGAN_model)
    model.setup_lightning()
    assert model.automatic_optimization is False
    assert (
        model._training_step_implementation.__func__
        is training_step_PL.training_step_PL2
    )


def test_setup_lightning_rejects_pre_v2(monkeypatch):
    monkeypatch.setattr(SRGAN.pl, "__version__", "1.9.5")
    model = SRGAN.SRGAN_model.__new__(SRGAN.SRGAN_model)
    with pytest.raises(RuntimeError, match="requires PyTorch Lightning >= 2.0"):
        model.setup_lightning()


def test_training_step_pl2_runs_manual_optimization():
    harness = TrainingHarness(pretrain=False)
    harness.automatic_optimization = False

    loss = training_step_PL.training_step_PL2(harness, _sample_batch(), batch_idx=0)

    assert torch.is_tensor(loss)
    assert "discriminator/adversarial_loss" in harness.logged
    assert "generator/total_loss" in harness.logged


def test_training_step_pl2_relativistic_branch_logs_rel_metrics():
    harness = TrainingHarness(pretrain=False)
    harness.pl_version = (2, 0, 0)
    harness.automatic_optimization = False
    harness.relativistic_average_d = True
    harness.adv_loss_type = "bce"

    loss = training_step_PL.training_step_PL2(harness, _sample_batch(), batch_idx=0)

    assert torch.is_tensor(loss)
    assert "train_metrics/discriminator/D(y)_prob_relativistic" in harness.logged
    assert "train_metrics/discriminator/D(G(x))_prob_relativistic" in harness.logged


def test_training_step_pl2_rejects_automatic_optimization():
    harness = TrainingHarness(pretrain=False)
    harness.automatic_optimization = True

    with pytest.raises(RuntimeError, match="manual optimization"):
        training_step_PL.training_step_PL2(harness, _sample_batch(), batch_idx=0)


def test_training_step_pl2_pretrain_branch_updates_generator_only():
    harness = TrainingHarness(pretrain=True)
    harness.automatic_optimization = False

    loss = training_step_PL.training_step_PL2(harness, _sample_batch(), batch_idx=0)

    assert torch.is_tensor(loss)
    assert harness.content_loss_criterion.calls == 1
    assert "discriminator/adversarial_loss" in harness.logged
    assert "train_metrics/l1" in harness.logged


def test_training_step_pl2_wasserstein_with_r1_and_gradient_clipping():
    harness = TrainingHarness(pretrain=False)
    harness.automatic_optimization = False
    harness.adv_loss_type = "wasserstein"
    harness.r1_gamma = 0.1
    harness.config.Optimizers.gradient_clip_val = 0.5

    loss = training_step_PL.training_step_PL2(harness, _sample_batch(), batch_idx=0)

    assert torch.is_tensor(loss)
    assert "discriminator/r1_penalty" in harness.logged
    assert "generator/adversarial_loss" in harness.logged


def _small_srgan_config():
    return OmegaConf.create(
        {
            "Model": {"in_bands": 1},
            "Generator": {
                "model_type": "SRResNet",
                "block_type": "standard",
                "n_channels": 4,
                "n_blocks": 1,
                "large_kernel_size": 3,
                "small_kernel_size": 3,
                "scaling_factor": 2,
            },
            "Discriminator": {
                "model_type": "standard",
                "n_blocks": 1,
                "use_spectral_norm": False,
            },
            "Training": {
                "pretrain_g_only": False,
                "g_pretrain_steps": 0,
                "adv_loss_ramp_steps": 4,
                "label_smoothing": False,
                "Losses": {
                    "adv_loss_beta": 0.5,
                    "adv_loss_schedule": "linear",
                    "adv_loss_type": "bce",
                    "relativistic_average_d": False,
                    "r1_gamma": 0.0,
                },
                "EMA": None,
            },
            "Optimizers": {
                "optim_g_lr": 1e-3,
                "optim_d_lr": 5e-4,
                "weight_decay_g": 0.01,
                "weight_decay_d": 0.02,
                "betas": [0.0, 0.99],
                "eps": 1e-7,
            },
            "Schedulers": {
                "metric": "val_loss",
                "metric_g": "val_g",
                "metric_d": "val_d",
                "factor_g": 0.5,
                "factor_d": 0.25,
                "patience_g": 2,
                "patience_d": 3,
                "threshold": 1e-4,
                "cooldown": 1,
                "min_lr": 1e-6,
                "g_warmup_steps": 3,
                "g_warmup_type": "cosine",
            },
            "Logging": {"wandb": {"enabled": False}, "num_val_images": 1},
        }
    )


def _capture_logs(model, monkeypatch):
    logged = {}

    def fake_log(key, value, **kwargs):
        logged[key] = value

    setattr(model, "log", fake_log)
    return logged


def test_configure_optimizers_builds_param_groups_and_warmup():
    model = SRGAN_model(config=_small_srgan_config(), mode="train")

    optimizers, schedulers = model.configure_optimizers()

    assert len(optimizers) == 2
    assert len(schedulers) == 3
    assert schedulers[0] is model._plateau_scheduler_d
    assert schedulers[1] is model._plateau_scheduler_g
    assert schedulers[2] is model._warmup_scheduler_g
    assert model._plateau_metric_d == "val_d"
    assert model._plateau_metric_g == "val_g"
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(5e-4)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert optimizers[0].param_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert optimizers[1].param_groups[0]["weight_decay"] == pytest.approx(0.01)

    warmup_lambda = schedulers[2].lr_lambdas[0]
    assert warmup_lambda(0) >= 0.05
    assert warmup_lambda(10) == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore:GPU available but not used.*")
@pytest.mark.filterwarnings("ignore:The '.*_dataloader' does not have many workers.*")
@pytest.mark.filterwarnings("ignore:.*isinstance\\(treespec, LeafSpec\\).*deprecated.*")
def test_manual_optimization_steps_warmup_and_plateau_schedulers(monkeypatch, tmp_path):
    config = _small_srgan_config()
    config.Discriminator.model_type = "patchgan"
    config.Training.Losses.update(
        {
            "l1_weight": 1.0,
            "sam_weight": 0.0,
            "perceptual_weight": 0.0,
            "tv_weight": 0.0,
            "ssim_win": 3,
        }
    )
    config.Schedulers.g_warmup_steps = 2
    config.Schedulers.metric_g = "val_metrics/l1"
    config.Schedulers.metric_d = "validation/DISC_adversarial_loss"
    config.Logging.num_val_images = 0
    model = SRGAN_model(config=config, mode="train")
    monkeypatch.setattr(SRGAN, "print_model_summary", lambda *_args, **_kwargs: None)

    lr = torch.rand(4, 1, 4, 4)
    hr = torch.rand(4, 1, 8, 8)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=CSVLogger(save_dir=tmp_path, name="scheduler-test"),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
        default_root_dir=tmp_path,
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    assert model._warmup_scheduler_g.last_epoch == 2
    assert model._plateau_scheduler_d.last_epoch == 1
    assert model._plateau_scheduler_g.last_epoch == 1
    assert trainer.optimizers[1].param_groups[0]["lr"] == pytest.approx(1e-3)


def test_train_batch_hooks_freeze_and_log_learning_rates(monkeypatch):
    model = SRGAN_model(config=_small_srgan_config(), mode="train")
    logged = _capture_logs(model, monkeypatch)
    model.pretrain_g_only = True
    model.g_pretrain_steps = -1

    model.on_train_batch_start(batch=None, batch_idx=0)

    assert all(not p.requires_grad for p in model.discriminator.parameters())

    model._trainer = types.SimpleNamespace(
        optimizers=[
            torch.optim.SGD(model.discriminator.parameters(), lr=0.11),
            torch.optim.SGD(model.generator.parameters(), lr=0.22),
        ]
    )
    model.on_train_batch_end(outputs=None, batch=None, batch_idx=0)

    assert logged["lr_discriminator"] == pytest.approx(0.11)
    assert logged["lr_generator"] == pytest.approx(0.22)


class DummyEma:
    def __init__(self):
        self.decay = 0.9
        self.device = None
        self.num_updates = 2
        self.last_decay = 0.8
        self.calls = []

    def to(self, device):
        self.calls.append(("to", device))
        self.device = device

    def apply_to(self, generator):
        self.calls.append(("apply_to", generator))

    def restore(self, generator):
        self.calls.append(("restore", generator))

    def state_dict(self):
        return {"shadow": 1}

    def load_state_dict(self, state):
        self.calls.append(("load_state_dict", state))

    @contextmanager
    def average_parameters(self, generator):
        self.calls.append(("average_parameters", generator))
        yield


def test_ema_helpers_log_apply_restore_and_checkpoint(monkeypatch):
    model = SRGAN_model(config=_small_srgan_config(), mode="train")
    logged = _capture_logs(model, monkeypatch)
    ema = DummyEma()
    model.ema = ema
    model._ema_update_after_step = 7
    model._ema_applied = False
    model._trainer = types.SimpleNamespace(global_step=3)

    model._log_ema_setup_metrics()
    model._log_ema_step_metrics(updated=False)
    model._log_ema_step_metrics(updated=True)
    model._apply_generator_ema_weights()
    model._apply_generator_ema_weights()
    model._restore_generator_weights()
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    model.on_load_checkpoint({"ema_state": {"shadow": 2}})

    assert logged["EMA/enabled"] == pytest.approx(1.0)
    assert logged["EMA/steps_until_activation"] == pytest.approx(4.0)
    assert logged["EMA/num_updates"] == pytest.approx(2.0)
    assert checkpoint["ema_state"] == {"shadow": 1}
    assert ("load_state_dict", {"shadow": 2}) in ema.calls
    assert model._ema_applied is False


def test_ema_logging_disabled_and_without_trainer(monkeypatch):
    model = types.SimpleNamespace(ema=None)
    model._log_ema_setup_metrics = types.MethodType(
        SRGAN_model._log_ema_setup_metrics, model
    )
    model._log_ema_step_metrics = types.MethodType(
        SRGAN_model._log_ema_step_metrics, model
    )
    logged = _capture_logs(model, monkeypatch)

    model._log_ema_setup_metrics()
    assert logged == {}

    model.trainer = object()
    model.ema = None
    model._log_ema_setup_metrics()
    model._log_ema_step_metrics(updated=True)

    assert logged["EMA/enabled"] == pytest.approx(0.0)


def test_predict_step_requires_eval_and_uses_normalizer_ema(monkeypatch):
    model = SRGAN_model(config=_small_srgan_config(), mode="eval")
    lr = torch.ones(1, 1, 4, 4)

    with pytest.raises(RuntimeError, match="eval mode"):
        model.predict_step(lr)

    model.generator.eval()
    model.ema = DummyEma()
    monkeypatch.setattr(SRGAN, "histogram_match", lambda normal, sr: sr)

    out = model.predict_step(lr)

    assert out.device.type == "cpu"
    assert out.shape[-2:] == (8, 8)


def test_predict_step_accepts_already_normalized_10k_inputs(monkeypatch):
    config = _small_srgan_config()
    config.Data = {"normalization": "normalise_10k"}
    model = SRGAN_model(config=config, mode="eval")
    model.generator = torch.nn.Identity()
    model.generator.eval()
    monkeypatch.setattr(SRGAN, "histogram_match", lambda normal, sr: sr)

    normalized_lr = torch.full((1, 1, 4, 4), 0.5)
    raw_lr = normalized_lr * 10000.0

    normalized_out = model.predict_step(normalized_lr)
    raw_out = model.predict_step(raw_lr)

    assert torch.allclose(normalized_out, normalized_lr)
    assert torch.allclose(raw_out, raw_lr)


class ValidationContentLoss:
    def return_metrics(self, sr, hr, prefix):
        return {f"{prefix}l1": torch.nn.functional.l1_loss(sr, hr)}


class ValidationHarness(LoggerMixin):
    def __init__(self, *, in_bands=1, pretrain=False, wasserstein=False):
        super().__init__()
        self.content_loss_criterion = ValidationContentLoss()
        self.config = OmegaConf.create(
            {
                "Model": {"in_bands": in_bands},
                "Logging": {"num_val_images": 1, "wandb": {"enabled": False}},
            }
        )
        self.discriminator = torch.nn.Conv2d(in_bands, 1, kernel_size=1)
        self.adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss()
        self.adv_loss_type = "wasserstein" if wasserstein else "bce"
        self.pretrain = pretrain

    def forward(self, lr):
        return lr

    def _pretrain_check(self):
        return self.pretrain


def test_validation_step_logs_metrics_and_pretrain_discriminator(monkeypatch):
    harness = ValidationHarness(in_bands=1, pretrain=True)
    monkeypatch.setattr(SRGAN, "plot_tensors", lambda *args, **kwargs: object())
    batch = (torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))

    SRGAN_model.validation_step(harness, batch, batch_idx=0)

    assert "val_metrics/l1" in harness.logged
    assert "validation/DISC_adversarial_loss" in harness.logged


def test_validation_step_logs_wasserstein_discriminator(monkeypatch):
    harness = ValidationHarness(in_bands=4, pretrain=False, wasserstein=True)
    monkeypatch.setattr(SRGAN, "plot_tensors", lambda *args, **kwargs: object())
    batch = (torch.ones(1, 4, 2, 2), torch.zeros(1, 4, 2, 2))

    SRGAN_model.validation_step(harness, batch, batch_idx=0)

    assert "validation/DISC_adversarial_loss" in harness.logged


@pytest.mark.parametrize(
    ("in_bands", "visible_bands"), [(1, 1), (3, 3), (4, 3), (6, 3)]
)
def test_validation_visualization_supports_common_band_counts(
    monkeypatch, in_bands, visible_bands
):
    harness = ValidationHarness(in_bands=in_bands, pretrain=False)
    plotted_shapes = []

    def fake_plot(*tensors, **_kwargs):
        plotted_shapes.extend(tensor.shape for tensor in tensors)
        return object()

    monkeypatch.setattr(SRGAN, "plot_tensors", fake_plot)
    batch = (
        torch.ones(1, in_bands, 2, 2),
        torch.zeros(1, in_bands, 2, 2),
    )

    SRGAN_model.validation_step(harness, batch, batch_idx=0)

    assert len(plotted_shapes) == 3
    assert all(shape[1] == visible_bands for shape in plotted_shapes)


def test_validation_discriminator_metric_is_not_limited_by_image_logging(monkeypatch):
    harness = ValidationHarness(in_bands=4, pretrain=False)

    def fail_if_plotted(*_args, **_kwargs):
        raise AssertionError("visualization should be skipped for this batch")

    monkeypatch.setattr(SRGAN, "plot_tensors", fail_if_plotted)
    batch = (torch.ones(1, 4, 2, 2), torch.zeros(1, 4, 2, 2))

    SRGAN_model.validation_step(harness, batch, batch_idx=2)

    assert "validation/DISC_adversarial_loss" in harness.logged


def test_training_step_dispatches_bound_implementation():
    model = SRGAN_model.__new__(SRGAN_model)
    model._training_step_implementation = lambda batch, batch_idx: (batch, batch_idx)

    assert SRGAN_model.training_step(model, "batch", 3) == ("batch", 3)


def test_adv_loss_weight_logs_computed_value(monkeypatch):
    model = SRGAN_model(config=_small_srgan_config(), mode="train")
    logged = _capture_logs(model, monkeypatch)
    model._trainer = types.SimpleNamespace(global_step=2)

    value = model._adv_loss_weight()

    assert value == pytest.approx(0.25)
    assert logged["training/adv_loss_weight"] == pytest.approx(0.25)


def test_adv_loss_ramp_starts_immediately_when_pretraining_is_disabled():
    config = _small_srgan_config()
    config.Training.pretrain_g_only = False
    config.Training.g_pretrain_steps = 1000
    config.Training.adv_loss_ramp_steps = 100
    config.Training.Losses.adv_loss_beta = 1.0
    model = SRGAN_model(config=config, mode="train")
    model._trainer = types.SimpleNamespace(global_step=50)

    assert model._compute_adv_loss_weight() == pytest.approx(0.5)


def test_load_weights_from_checkpoint_accepts_lightning_state_dict(tmp_path, capsys):
    model = SRGAN_model(config=_small_srgan_config(), mode="eval")
    checkpoint_path = Path(tmp_path) / "model.ckpt"
    state_dict = model.state_dict()
    torch.save({"state_dict": state_dict}, checkpoint_path)

    model.load_weights_from_checkpoint(checkpoint_path, map_location="cpu")

    assert "Loaded weights from checkpoint" in capsys.readouterr().out
