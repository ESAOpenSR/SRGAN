"""Tests for :mod:`opensr_srgan.utils.build_trainer_kwargs`."""

import sys
import types

import importlib.util
from pathlib import Path

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
            resume_from_checkpoint=None,
        ) -> None:
            pass

        def fit(self, *args, ckpt_path=None, **kwargs):  # pragma: no cover
            return None

    pl_stub.Trainer = _Trainer
    pl_stub.__version__ = "2.1.0"
    sys.modules["pytorch_lightning"] = pl_stub


MODULE_NAME = "opensr_srgan.utils.build_trainer_kwargs"
MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "opensr_srgan"
    / "utils"
    / "build_trainer_kwargs.py"
)


if "opensr_srgan" not in sys.modules:
    pkg_stub = types.ModuleType("opensr_srgan")
    pkg_stub.__path__ = []  # mark as package
    sys.modules["opensr_srgan"] = pkg_stub

if "opensr_srgan.utils" not in sys.modules:
    utils_stub = types.ModuleType("opensr_srgan.utils")
    utils_stub.__path__ = []
    sys.modules["opensr_srgan.utils"] = utils_stub

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
build_module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = build_module
assert spec.loader is not None  # for type checkers
spec.loader.exec_module(build_module)
sys.modules["opensr_srgan.utils"].build_trainer_kwargs = build_module
sys.modules["opensr_srgan"].utils = sys.modules["opensr_srgan.utils"]


build_lightning_kwargs = build_module.build_lightning_kwargs


def _make_config(**training_overrides):
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
                "gradient_clip_val": 0.0,
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


def test_multi_gpu_enables_ddp(monkeypatch):
    """Multiple GPUs trigger the DDP strategy when CUDA is requested."""

    monkeypatch.setattr(
        "opensr_srgan.utils.build_trainer_kwargs.torch.cuda.is_available",
        lambda: True,
    )
    config = _make_config(device="cuda", gpus=[0, 1])
    trainer_kwargs, _ = _call_builder(config)

    assert trainer_kwargs["accelerator"] == "gpu"
    assert trainer_kwargs["devices"] == [0, 1]
    assert trainer_kwargs["strategy"] == "ddp"


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
