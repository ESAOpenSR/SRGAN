"""Compatibility and value-range contracts for prediction post-processing."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch
import torch.nn.functional as F

from opensr_srgan.model.SRGAN import SRGAN_model
from opensr_srgan.utils.back_projection import BlurDownsample
from opensr_srgan.utils.radiometrics import histogram

ROOT = Path(__file__).resolve().parents[2]


def model_for(method=None):
    cfg = OmegaConf.load(ROOT / 'opensr_srgan/configs/config_training_example.yaml')
    cfg.Model.in_bands = 1
    cfg.Generator = {'model_type': 'srresnet', 'block_type': 'res',
                     'n_channels': 4, 'n_blocks': 1, 'scaling_factor': 2}
    if method is None:
        del cfg.Postprocessing
    else:
        cfg.Postprocessing.method = method
    return SRGAN_model(cfg, mode='eval').eval()


@pytest.mark.parametrize('method', [None, 'histogram'])
@pytest.mark.parametrize('raw', [False, True])
def test_histogram_retains_historical_prediction_sequence(method, raw):
    model = model_for(method)
    lr = torch.linspace(.1, .8, 12).reshape(1, 1, 3, 4)
    if raw:
        lr = lr * 10000
    normalized = model.normalizer.normalize(lr) if raw else lr
    with torch.no_grad():
        expected = histogram(normalized, model.generator(normalized))
    if raw:
        expected = model.normalizer.denormalize(expected)
    torch.testing.assert_close(model.predict_step(lr), expected)


@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('integer', [False, True])
def test_back_projection_matches_original_measurements_and_preserves_input(raw, integer):
    model = model_for('back_projection')
    lr = torch.linspace(.1, .8, 12).reshape(1, 1, 3, 4)
    if raw or integer:
        lr = lr * 10000
    if integer:
        lr = lr.to(torch.int32)
    before = lr.clone()
    result = model.predict_step(lr)
    torch.testing.assert_close(BlurDownsample(1, scale=2)(result), lr.float(), atol=.001, rtol=1e-6)
    assert torch.equal(lr, before)
    assert result.device.type == 'cpu' and not result.requires_grad


def test_postprocess_is_separate_from_raw_forward():
    model = model_for('back_projection')
    lr = torch.rand(1, 1, 3, 4)
    raw = model(lr)
    before = raw.detach().clone()
    matched = model.postprocess(raw, lr)
    assert raw.requires_grad and not matched.requires_grad
    torch.testing.assert_close(raw, before)
    torch.testing.assert_close(model(lr), raw)
    torch.testing.assert_close(F.avg_pool2d(matched, 2), lr)


def test_back_projection_promotes_half_and_does_not_clip():
    model = model_for('back_projection')
    raw = torch.tensor([[[[-2., 2.], [-2., 2.]]]], dtype=torch.float16)
    lr = torch.tensor([[[[.5]]]], dtype=torch.float16)
    result = model.postprocess(raw, lr)
    assert result.dtype == torch.float32 and result.min() < 0 and result.max() > 1
    torch.testing.assert_close(result.mean(), torch.tensor(.5))


def test_unknown_method_fails_before_building_model():
    with pytest.raises(ValueError, match='Postprocessing.method'):
        model_for('typo')


def test_back_projection_rejects_incompatible_geometry():
    model = model_for('back_projection')
    with pytest.raises(ValueError, match='integer multiple'):
        model.postprocess(torch.zeros(1, 1, 7, 8), torch.zeros(1, 1, 3, 4))
