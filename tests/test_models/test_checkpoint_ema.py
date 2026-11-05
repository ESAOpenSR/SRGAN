"""Checkpoint contracts for live training weights and inference EMA weights."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import pytorch_lightning as pl
import torch

from opensr_srgan.model.SRGAN import SRGAN_model


def small_model(*, use_num_updates=True):
    config = OmegaConf.load(Path(__file__).resolve().parents[2] / 'opensr_srgan/configs/config_training_example.yaml')
    config.Generator = {'model_type': 'srresnet', 'block_type': 'res',
                        'n_channels': 4, 'n_blocks': 1, 'scaling_factor': 2}
    config.Training.EMA = {'enabled': True, 'decay': .99, 'device': 'cpu',
                           'use_num_updates': use_num_updates}
    return SRGAN_model(config, mode='eval').eval()


@pytest.mark.parametrize('clone_state', [False, True])
def test_checkpoint_saved_during_ema_validation_contains_live_weights(clone_state):
    model = small_model()
    live = {key: value.clone() for key, value in model.state_dict().items()}
    for value in model.ema.shadow_params.values():
        value.add_(1)
    shadow = {key: value.clone() for key, value in model.ema.shadow_params.items()}
    model._apply_generator_ema_weights()
    state = model.state_dict()
    if clone_state:
        state = {key: value.clone() for key, value in state.items()}
    checkpoint = {'state_dict': state}

    model.on_save_checkpoint(checkpoint)

    assert model._ema_applied
    assert model.ema.collected_params
    for key, expected in live.items():
        torch.testing.assert_close(checkpoint['state_dict'][key], expected)
    for key, expected in shadow.items():
        torch.testing.assert_close(checkpoint['ema_state']['shadow_params'][key], expected)
        torch.testing.assert_close(model.generator.state_dict()[key], expected)
    model._restore_generator_weights()
    for key, expected in live.items():
        torch.testing.assert_close(model.state_dict()[key], expected)


def test_failed_checkpoint_save_preserves_active_ema_weights(monkeypatch):
    model = small_model()
    for value in model.ema.shadow_params.values():
        value.add_(1)
    model._apply_generator_ema_weights()
    active = {key: value.clone() for key, value in model.generator.state_dict().items()}

    def fail_save(_self, _checkpoint):
        raise RuntimeError('save failed')

    monkeypatch.setattr(pl.LightningModule, 'on_save_checkpoint', fail_save)
    with pytest.raises(RuntimeError, match='save failed'):
        model.on_save_checkpoint({'state_dict': model.state_dict()})

    assert model._ema_applied
    for key, expected in active.items():
        torch.testing.assert_close(model.generator.state_dict()[key], expected)


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('use_num_updates', [False, True])
def test_weight_initialization_reseeds_ema_from_imported_generator(tmp_path, wrapped, use_num_updates):
    donor = small_model()
    model = small_model(use_num_updates=use_num_updates)
    model.ema.update(model.generator)
    checkpoint = {'state_dict': donor.state_dict()} if wrapped else donor.state_dict()
    path = tmp_path / 'weights.ckpt'
    torch.save(checkpoint, path)

    model.load_weights_from_checkpoint(path, strict=True, map_location='cpu')

    assert model.ema.decay == .99
    assert model.ema.device == torch.device('cpu')
    assert model.ema.num_updates == (0 if use_num_updates else None)
    assert not model._ema_applied
    for key, value in donor.generator.named_parameters():
        torch.testing.assert_close(model.ema.shadow_params[key], value)
    image = torch.rand(1, 4, 4, 4)
    with model.ema.average_parameters(model.generator), torch.no_grad():
        torch.testing.assert_close(model(image), donor(image))
