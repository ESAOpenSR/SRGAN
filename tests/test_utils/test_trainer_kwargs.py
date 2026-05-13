"""Tests for :mod:`opensr_srgan.utils.build_trainer_kwargs`."""

import sys
import types

import pytest
from omegaconf import OmegaConf

if "pytorch_lightning" not in sys.modules:
    # Provide a lightweight stub so the helper can be imported without the heavy
    # Lightning dependency.  Only the constructor and ``fit`` signatures are
    # required for the tests to exercise the filtering logic.
    pl_stub = types.ModuleType("pytorch_lightning")

    class _Trainer:  # pragma: no cover - simple stub
        def __init__(
            self,
            *,
            accelerator=None,
            strategy=None,
            devices=None,
            val_check_interval=None,
            limit_val_batches=None,
            max_epochs=None,
            log_every_n_steps=None,
            logger=None,
            callbacks=None,
        ) -> None:
            pass

        def fit(self, *args, ckpt_path=None, **kwargs):  # pragma: no cover
            return None

    pl_stub.Trainer = _Trainer
    pl_stub.__version__ = "2.1.0"
    sys.modules["pytorch_lightning"] = pl_stub

from opensr_srgan.utils.build_trainer_kwargs import build_lightning_kwargs


def _make_config(gradient_clip_val=0.0, **training_overrides):
    """Return a minimal OmegaConf training config for the helper."""

    base_training = {
        "device": "cuda",
        "gpus": [0],
        "val_check_interval": 1.0,
        "limit_val_batches": 1,
        "max_epochs": 5,
    }
    base_training.update(training_overrides)

    return OmegaConf.create(
        {
            "Training": base_training,
            # ``build_lightning_kwargs`` expects optimiser settings for gradient
            # clipping, so provide the minimal structure required by the real
            # configuration files.
            "Optimizers": {
                "gradient_clip_val": gradient_clip_val,
            },
        }
    )


def _call_builder(config, resume_ckpt=None):
    """Invoke ``build_lightning_kwargs`` with deterministic sentinels."""

    return build_lightning_kwargs(
        config=config,
        logger="logger",
        checkpoint_callback="checkpoint",
        early_stop_callback="early_stop",
        resume_ckpt=resume_ckpt,
    )


def test_cpu_device_forces_single_device(monkeypatch):
    """A CPU run ignores the GPU list and does not set a DDP strategy."""

    config = _make_config(device="cpu", gpus=[0, 1, 2])
    trainer_kwargs, fit_kwargs = _call_builder(config)

    assert trainer_kwargs["accelerator"] == "cpu"
    assert trainer_kwargs["devices"] == 1
    assert "strategy" not in trainer_kwargs
    assert fit_kwargs == {}


def test_multi_gpu_enables_ddp_with_unused_param_detection(monkeypatch):
    """Multiple GPUs default to DDP with find-unused-parameters enabled."""

    monkeypatch.setattr(
        "opensr_srgan.utils.build_trainer_kwargs.torch.cuda.is_available",
        lambda: True,
    )
    config = _make_config(device="cuda", gpus=[0, 1])
    trainer_kwargs, _ = _call_builder(config)

    assert trainer_kwargs["accelerator"] == "gpu"
    assert trainer_kwargs["devices"] == [0, 1]
    assert trainer_kwargs["strategy"] == "ddp_find_unused_parameters_true"


def test_auto_device_respects_cuda_availability(monkeypatch):
    """The ``auto`` device selects CPU when CUDA is unavailable."""

    monkeypatch.setattr(
        "opensr_srgan.utils.build_trainer_kwargs.torch.cuda.is_available",
        lambda: False,
    )
    config = _make_config(device="auto", gpus=[0, 1])
    trainer_kwargs, _ = _call_builder(config)

    assert trainer_kwargs["accelerator"] == "cpu"
    assert trainer_kwargs["devices"] == 1
    assert "strategy" not in trainer_kwargs


def test_invalid_device_raises():
    """Unexpected device strings surface a clear ``ValueError``."""

    config = _make_config(device="tpu")

    with pytest.raises(ValueError):
        _call_builder(config)


def test_integer_gpu_count_enables_ddp_and_resume_ckpt_path():
    """Integer GPU counts still enable DDP and resume via ``ckpt_path``."""

    config = _make_config(device="cuda", gpus=2)
    trainer_kwargs, fit_kwargs = _call_builder(config, resume_ckpt="resume.ckpt")

    assert trainer_kwargs["devices"] == 2
    assert trainer_kwargs["strategy"] == "ddp_find_unused_parameters_true"
    assert fit_kwargs == {"ckpt_path": "resume.ckpt"}


def test_multi_gpu_can_disable_unused_param_detection(monkeypatch):
    """Config can opt out and use plain DDP strategy."""

    monkeypatch.setattr(
        "opensr_srgan.utils.build_trainer_kwargs.torch.cuda.is_available",
        lambda: True,
    )
    config = _make_config(
        device="cuda",
        gpus=[0, 1],
        find_unused_parameters=False,
    )
    trainer_kwargs, _ = _call_builder(config)

    assert trainer_kwargs["strategy"] == "ddp"


def test_v2_resume_uses_fit_ckpt_path():
    """PL>=2 forwards resume checkpoints via ``Trainer.fit(ckpt_path=...)``."""

    config = _make_config(device="cuda", gpus=[0])
    _, fit_kwargs = _call_builder(config, resume_ckpt="resume.ckpt")

    assert fit_kwargs == {"ckpt_path": "resume.ckpt"}


def test_non_sequence_gpu_config_falls_back_to_single_device():
    """Unexpected ``gpus`` values trigger the safe single-device fallback."""

    config = _make_config(device="cuda", gpus="not-a-device-list")
    trainer_kwargs, _ = _call_builder(config)

    assert trainer_kwargs["devices"] == 1
    assert "strategy" not in trainer_kwargs


def test_gradient_clipping_is_not_forwarded_to_trainer():
    """Manual optimization handles clipping in the training step, not Trainer."""

    config = _make_config(gradient_clip_val=1.0)
    trainer_kwargs, _ = _call_builder(config)

    assert "gradient_clip_val" not in trainer_kwargs
