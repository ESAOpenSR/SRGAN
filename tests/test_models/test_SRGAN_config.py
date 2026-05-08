"""Comprehensive tests for SRGAN_model covering configuration, discriminators, EMA, and loss scheduling."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
_ = pytest.importorskip("pytorch_lightning")

from opensr_srgan.model.SRGAN import SRGAN_model


@pytest.fixture
def base_config():
    """Minimal valid SRGAN configuration."""
    return OmegaConf.create(
        {
            "Model": {"in_bands": 3},
            "Generator": {"model_type": "SRResNet", "scaling_factor": 4},
            "Discriminator": {"model_type": "standard"},
            "Training": {
                "pretrain_g_only": False,
                "g_pretrain_steps": 0,
                "adv_loss_ramp_steps": 100,
                "label_smoothing": False,
                "Losses": {
                    "adv_loss_beta": 1.0,
                    "adv_loss_schedule": "linear",
                    "adv_loss_type": "bce",
                    "relativistic_average_d": False,
                    "r1_gamma": 0.0,
                },
                "EMA": None,
            },
            "Optimizers": {"optim_g_lr": 1e-4, "optim_d_lr": 1e-4},
            "Schedulers": {},
            "Logging": {"wandb": {"enabled": False}, "num_val_images": 1},
        }
    )


def test_srgan_initialization_train_mode(base_config):
    """Test basic initialization in train mode."""
    model = SRGAN_model(config=base_config, mode="train")
    assert model.generator is not None
    assert model.discriminator is not None
    assert model.mode == "train"
    assert not model.automatic_optimization


def test_srgan_initialization_eval_mode(base_config):
    """Test initialization in eval mode (discriminator not created)."""
    model = SRGAN_model(config=base_config, mode="eval")
    assert model.generator is not None
    assert not hasattr(model, "discriminator") or model.discriminator is None


def test_srgan_config_from_dict():
    """Test model accepts dict config and converts to OmegaConf."""
    config_dict = {
        "Model": {"in_bands": 4},
        "Generator": {"model_type": "SRResNet", "scaling_factor": 4},
        "Discriminator": {"model_type": "standard"},
        "Training": {
            "pretrain_g_only": False,
            "g_pretrain_steps": 0,
            "adv_loss_ramp_steps": 1000,
            "Losses": {
                "adv_loss_beta": 1.0,
                "adv_loss_schedule": "linear",
                "adv_loss_type": "bce",
                "r1_gamma": 0.0,
            },
            "EMA": None,
        },
        "Optimizers": {"optim_g_lr": 1e-4, "optim_d_lr": 1e-4},
        "Schedulers": {},
        "Logging": {"wandb": {"enabled": False}, "num_val_images": 1},
    }
    model = SRGAN_model(config=config_dict, mode="train")
    assert model.config is not None
    assert model.config.Model.in_bands == 4


def test_srgan_config_invalid_mode():
    """Test that invalid mode raises assertion."""
    config = OmegaConf.create(
        {
            "Model": {"in_bands": 3},
            "Generator": {"model_type": "SRResNet", "scaling_factor": 4},
            "Discriminator": {"model_type": "standard"},
            "Training": {
                "pretrain_g_only": False,
                "g_pretrain_steps": 0,
                "adv_loss_ramp_steps": 100,
                "Losses": {
                    "adv_loss_beta": 1.0,
                    "adv_loss_schedule": "linear",
                    "adv_loss_type": "bce",
                    "r1_gamma": 0.0,
                },
                "EMA": None,
            },
            "Optimizers": {"optim_g_lr": 1e-4, "optim_d_lr": 1e-4},
            "Schedulers": {},
            "Logging": {"wandb": {"enabled": False}, "num_val_images": 1},
        }
    )
    with pytest.raises(ValueError, match="train"):
        SRGAN_model(config=config, mode="invalid")


def test_discriminator_patchgan(base_config):
    """Test PatchGAN discriminator initialization."""
    base_config.Discriminator.model_type = "patchgan"
    base_config.Discriminator.n_blocks = 3
    model = SRGAN_model(config=base_config, mode="train")
    assert model.discriminator is not None
    from opensr_srgan.model.discriminators.patchgan import PatchGANDiscriminator

    assert isinstance(model.discriminator, PatchGANDiscriminator)


def test_discriminator_esrgan(base_config):
    """Test ESRGAN discriminator initialization."""
    base_config.Discriminator.model_type = "esrgan"
    base_config.Discriminator.base_channels = 64
    base_config.Discriminator.linear_size = 1024
    model = SRGAN_model(config=base_config, mode="train")
    assert model.discriminator is not None
    from opensr_srgan.model.discriminators.esrgan import ESRGANDiscriminator

    assert isinstance(model.discriminator, ESRGANDiscriminator)


def test_discriminator_standard_with_spectral_norm(base_config):
    """Test standard discriminator with spectral norm option."""
    base_config.Discriminator.use_spectral_norm = True
    model = SRGAN_model(config=base_config, mode="train")
    assert model.discriminator is not None


def test_discriminator_invalid_type(base_config):
    """Test that invalid discriminator type raises ValueError."""
    base_config.Discriminator.model_type = "unknown"
    with pytest.raises(ValueError, match="Unknown discriminator"):
        SRGAN_model(config=base_config, mode="train")


def test_adv_loss_type_invalid(base_config):
    """Test that invalid adversarial loss type raises ValueError."""
    base_config.Training.Losses.adv_loss_type = "invalid"
    with pytest.raises(ValueError, match="adv_loss_type must be either"):
        SRGAN_model(config=base_config, mode="train")


def test_adv_loss_wasserstein(base_config):
    """Test Wasserstein GAN loss type initialization."""
    base_config.Training.Losses.adv_loss_type = "wasserstein"
    model = SRGAN_model(config=base_config, mode="train")
    assert model.adv_loss_type == "wasserstein"
    assert model.adversarial_loss_criterion is None


def test_label_smoothing_configuration(base_config):
    """Test label smoothing adjusts adversarial target."""
    base_config.Training.label_smoothing = True
    model = SRGAN_model(config=base_config, mode="train")
    assert model.adv_target == 0.9

    base_config.Training.label_smoothing = False
    model = SRGAN_model(config=base_config, mode="train")
    assert model.adv_target == 1.0


def test_ema_initialization_disabled(base_config):
    """Test that EMA is None when disabled."""
    base_config.Training.EMA = {"enabled": False}
    model = SRGAN_model(config=base_config, mode="train")
    assert model.ema is None


def test_ema_initialization_enabled(base_config):
    """Test EMA initialization when enabled."""
    base_config.Training.EMA = {
        "enabled": True,
        "decay": 0.999,
        "use_num_updates": True,
        "update_after_step": 1000,
    }
    model = SRGAN_model(config=base_config, mode="train")
    assert model.ema is not None
    assert model.ema.decay == 0.999
    assert model._ema_update_after_step == 1000


def test_forward_pass(base_config):
    """Test forward pass through generator."""
    model = SRGAN_model(config=base_config, mode="train")
    lr = torch.randn(2, 3, 32, 32)
    sr = model.forward(lr)
    assert sr.shape[0] == 2
    assert sr.shape[1] == 3
    # Check spatial dimensions scaled by factor 4
    assert sr.shape[2] == 128
    assert sr.shape[3] == 128


def test_pretrain_check_during_pretraining(base_config):
    """Test _pretrain_check returns True during pretraining phase."""
    base_config.Training.pretrain_g_only = True
    base_config.Training.g_pretrain_steps = 1000
    model = SRGAN_model(config=base_config, mode="train")
    # Use trainer simulation to set global_step
    from unittest.mock import MagicMock
    trainer_mock = MagicMock()
    trainer_mock.global_step = 500
    model.trainer = trainer_mock
    assert model._pretrain_check() is True

    trainer_mock.global_step = 1500
    assert model._pretrain_check() is False


def test_pretrain_check_pretraining_disabled(base_config):
    """Test _pretrain_check returns False when pretraining disabled."""
    base_config.Training.pretrain_g_only = False
    model = SRGAN_model(config=base_config, mode="train")
    from unittest.mock import MagicMock
    trainer_mock = MagicMock()
    trainer_mock.global_step = 100
    model.trainer = trainer_mock
    assert model._pretrain_check() is False


def test_adv_loss_weight_linear_ramp(base_config):
    """Test adversarial loss weight with linear ramp schedule."""
    base_config.Training.adv_loss_ramp_steps = 100
    base_config.Training.Losses.adv_loss_schedule = "linear"
    base_config.Training.Losses.adv_loss_beta = 1.0
    base_config.Training.g_pretrain_steps = 0
    model = SRGAN_model(config=base_config, mode="train")
    from unittest.mock import MagicMock
    trainer_mock = MagicMock()
    model.trainer = trainer_mock

    # At start: weight = 0
    trainer_mock.global_step = 0
    assert model._compute_adv_loss_weight() == pytest.approx(0.0)

    # At 50% through ramp: weight ≈ 0.5
    trainer_mock.global_step = 50
    assert model._compute_adv_loss_weight() == pytest.approx(0.5, abs=0.01)

    # At end of ramp: weight = 1.0
    trainer_mock.global_step = 100
    assert model._compute_adv_loss_weight() == pytest.approx(1.0)

    # After ramp: weight stays at beta
    trainer_mock.global_step = 200
    assert model._compute_adv_loss_weight() == pytest.approx(1.0)


def test_adv_loss_weight_cosine_ramp(base_config):
    """Test adversarial loss weight with cosine ramp schedule."""
    base_config.Training.adv_loss_ramp_steps = 100
    base_config.Training.Losses.adv_loss_schedule = "cosine"
    base_config.Training.Losses.adv_loss_beta = 1.0
    base_config.Training.g_pretrain_steps = 0
    model = SRGAN_model(config=base_config, mode="train")
    from unittest.mock import MagicMock
    trainer_mock = MagicMock()
    model.trainer = trainer_mock

    # At start: weight = 0
    trainer_mock.global_step = 0
    assert model._compute_adv_loss_weight() == pytest.approx(0.0)

    # At 50% through ramp: weight ≈ 0.5 (cosine)
    trainer_mock.global_step = 50
    expected = 0.5 * (1.0 - math.cos(math.pi * 0.5))
    assert model._compute_adv_loss_weight() == pytest.approx(expected, abs=0.01)

    # At end of ramp: weight = 1.0
    trainer_mock.global_step = 100
    assert model._compute_adv_loss_weight() == pytest.approx(1.0)


def test_adv_loss_weight_during_pretrain(base_config):
    """Test that adv loss weight is 0 during pretraining."""
    base_config.Training.pretrain_g_only = True
    base_config.Training.g_pretrain_steps = 100
    base_config.Training.Losses.adv_loss_beta = 1.0
    base_config.Training.adv_loss_ramp_steps = 100
    model = SRGAN_model(config=base_config, mode="train")
    from unittest.mock import MagicMock
    trainer_mock = MagicMock()
    trainer_mock.global_step = 50
    model.trainer = trainer_mock
    assert model._compute_adv_loss_weight() == pytest.approx(0.0)


def test_adv_loss_weight_invalid_schedule(base_config):
    """Test that invalid schedule raises ValueError."""
    base_config.Training.Losses.adv_loss_schedule = "invalid"
    base_config.Training.g_pretrain_steps = 0
    model = SRGAN_model(config=base_config, mode="train")
    from unittest.mock import MagicMock
    trainer_mock = MagicMock()
    trainer_mock.global_step = 10
    model.trainer = trainer_mock

    with pytest.raises(ValueError, match="Unknown adversarial loss schedule"):
        model._compute_adv_loss_weight()


def test_relativistic_average_d_configuration(base_config):
    """Test relativistic_average_d configuration is parsed."""
    base_config.Training.Losses.relativistic_average_d = True
    model = SRGAN_model(config=base_config, mode="train")
    assert model.relativistic_average_d is True


def test_r1_gradient_penalty_configuration(base_config):
    """Test R1 gradient penalty parameter is parsed."""
    base_config.Training.Losses.r1_gamma = 10.0
    model = SRGAN_model(config=base_config, mode="train")
    assert model.r1_gamma == 10.0
