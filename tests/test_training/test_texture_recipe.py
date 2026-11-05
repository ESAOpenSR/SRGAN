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
    assert cfg.Training.max_steps == 328584
    assert cfg.Training.max_steps - 308584 == 20000  # 10,000 additional G updates
    assert list(cfg.Training.checkpoint_optimizer_steps) == [308584, 328584]
    assert cfg.Training.Losses.texture_weight == pytest.approx(0.46851534843444825)
    assert cfg.Training.Losses.texture_grid == 4
    assert cfg.Training.Losses.vgg_input_normalization
    assert cfg.Training.Losses.vgg_before_activation
    assert cfg.Training.Losses.fixed_idx is None
    assert list(cfg.Training.Losses.validation_fixed_idx) == [0, 1, 2]
    assert cfg.Training.Losses.global_relativistic_average is True
    assert cfg.Training.Losses.r1_normalize_patch_mean is True
    assert cfg.Training.Losses.texture_selection_nir_weight == pytest.approx(0.5)
    assert cfg.Training.EMA.decay == .99
    assert cfg.Optimizers.optim_g_lr == 5e-6 and cfg.Optimizers.optim_d_lr == 5e-7
    assert cfg.Schedulers.d_lr_ratio_to_g == .1
    assert cfg.Schedulers.checkpoint_monitor == 'val_metrics/multiband_selection'
    assert cfg.Schedulers.d_plateau_enabled is False
    assert cfg.Schedulers.convergence.enabled is False
    assert cfg.Schedulers.fixed_restart.enabled is False
    assert 'stop_at_utc' not in cfg.Training


def test_packaged_split_is_the_original_spatial_holdout():
    path = ROOT / 'opensr_srgan/configs/splits/sen2naip_crosssensor.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '218d5d73aec72bbb17a3853606759439f70cb28a1cafe298ee505234b2b21c5d'
    split = json.loads(path.read_text())
    assert split['seed'] == 2025 and split['group']['cell_degrees'] == .25
    # The final run moves its 16 displayed scenes first; membership is unchanged.
    assert split['indices']['val'][:16] == [23, 680, 1131, 1616, 2172, 2681, 3368, 3844, 4425, 4938, 5484, 5961, 6374, 6964, 7531, 7998]
    assert {key: len(value) for key, value in split['indices'].items()} == {'train': 6401, 'val': 800, 'test': 799}
    sets = [set(split['indices'][key]) for key in ('train', 'val', 'test')]
    assert len(set.union(*sets)) == sum(map(len, sets)) == split['nrecords'] == 8000
    for key in ('train', 'val', 'test'):
        assert len(split['indices'][key]) == len(split['ids'][key])


def test_sen2_sr_inference_config_constructs_eval_generator():
    from opensr_srgan.model.SRGAN import SRGAN_model

    config = OmegaConf.load(ROOT / 'opensr_srgan/configs/pretrained/sen2_sr.yaml')
    model = SRGAN_model(config, mode='eval')
    model.eval()
    assert model.generator.__class__.__name__ == 'ESRGANGenerator'
    assert model.generator.training is False
    assert not hasattr(model, 'discriminator')
    assert model.ema is None
    assert model.normalizer.method == 'normalise_10k'
    assert config.Postprocessing.method == 'back_projection'
