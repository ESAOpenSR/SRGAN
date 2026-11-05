"""Inference configuration contracts for published Sen2 SR weights."""
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


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
