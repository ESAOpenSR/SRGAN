"""The promoted recipe is portable and identical across supported interfaces."""
import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[2]
FLAT = ROOT / 'opensr_srgan/configs/config_sen2naip_texture.yaml'


def test_texture_hydra_and_flat_yaml_match_all_sections():
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'opensr_srgan/configs/hydra'), version_base='1.3'):
        cfg = hydra.compose(config_name='train', overrides=['experiment=sen2naip_texture'])
    expected = OmegaConf.load(FLAT)
    assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(expected, resolve=True)
    assert OmegaConf.missing_keys(cfg) == {
        'Data.sen2naip_taco_file', 'Data.sen2naip_split_file', 'Model.continue_training',
    }
    assert cfg.Model.load_checkpoint is False
    assert cfg.Logging.wandb.enabled is False
    assert cfg.Logging.wandb.id is None and cfg.Logging.wandb.resume is None
    assert cfg.Postprocessing.method == 'back_projection'


def test_promoted_experiment_settings_and_absolute_step_budget():
    cfg = OmegaConf.load(FLAT)
    assert cfg.Generator.model_type == 'esrgan' and cfg.Generator.n_blocks == 23
    assert cfg.Discriminator.model_type == 'patchgan'
    assert cfg.Training.max_steps == 300000
    assert cfg.Training.max_steps // 2 == 150000
    assert list(cfg.Training.checkpoint_optimizer_steps) == list(range(210000, 300001, 10000))
    assert cfg.Training.Losses.texture_weight == pytest.approx(1.874061393737793)
    assert cfg.Training.Losses.texture_grid == 4
    assert cfg.Training.Losses.vgg_input_normalization
    assert cfg.Training.Losses.vgg_before_activation
    assert list(cfg.Training.Losses.fixed_idx) == [0, 1, 2]
    assert cfg.Training.EMA.decay == .99
    assert cfg.Optimizers.optim_g_lr == 1e-5 and cfg.Optimizers.optim_d_lr == 1e-6
    assert cfg.Schedulers.d_lr_ratio_to_g == .1
    assert cfg.Schedulers.checkpoint_monitor == 'val_metrics/texture_selection'
    assert cfg.Schedulers.d_plateau_enabled is False
    assert 'convergence' not in cfg.Schedulers and 'stop_at_utc' not in cfg.Training


def test_packaged_split_is_the_original_spatial_holdout():
    path = ROOT / 'opensr_srgan/configs/splits/sen2naip_crosssensor.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'd2261258df5af4cca513f7f0f5172de479bd7d7e2a1c2b60ad41b34584b369d6'
    split = json.loads(path.read_text())
    assert split['seed'] == 2025 and split['group']['cell_degrees'] == .25
    assert {key: len(value) for key, value in split['indices'].items()} == {'train': 6401, 'val': 800, 'test': 799}
    sets = [set(split['indices'][key]) for key in ('train', 'val', 'test')]
    assert len(set.union(*sets)) == sum(map(len, sets)) == split['nrecords'] == 8000
    for key in ('train', 'val', 'test'):
        assert len(split['indices'][key]) == len(split['ids'][key])
