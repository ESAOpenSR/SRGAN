import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

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
    def return_loss(self, sr, hr):
        loss = torch.nn.functional.l1_loss(sr, hr)
        return loss, {"l1": loss.detach()}


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
    assert schedulers[0]["monitor"] == "val_d"
    assert schedulers[1]["monitor"] == "val_g"
    assert schedulers[2]["name"] == "warmup_g"
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(5e-4)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert optimizers[0].param_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert optimizers[1].param_groups[0]["weight_decay"] == pytest.approx(0.01)

    warmup_lambda = schedulers[2]["scheduler"].lr_lambdas[0]
    assert warmup_lambda(0) >= 0.05
    assert warmup_lambda(10) == pytest.approx(1.0)


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
    assert "discriminator/adversarial_loss" in harness.logged


def test_validation_step_logs_wasserstein_discriminator(monkeypatch):
    harness = ValidationHarness(in_bands=4, pretrain=False, wasserstein=True)
    monkeypatch.setattr(SRGAN, "plot_tensors", lambda *args, **kwargs: object())
    batch = (torch.ones(1, 4, 2, 2), torch.zeros(1, 4, 2, 2))

    SRGAN_model.validation_step(harness, batch, batch_idx=0)

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


def test_load_weights_from_checkpoint_accepts_lightning_state_dict(tmp_path, capsys):
    model = SRGAN_model(config=_small_srgan_config(), mode="eval")
    checkpoint_path = Path(tmp_path) / "model.ckpt"
    state_dict = model.state_dict()
    torch.save({"state_dict": state_dict}, checkpoint_path)

    model.load_weights_from_checkpoint(checkpoint_path, map_location="cpu")

    assert "Loaded weights from checkpoint" in capsys.readouterr().out
