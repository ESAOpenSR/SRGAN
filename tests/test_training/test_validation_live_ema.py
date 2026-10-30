from types import MethodType, SimpleNamespace

import torch

from opensr_srgan.model.SRGAN import SRGAN_model
from opensr_srgan.model.model_blocks.EMA import ExponentialMovingAverage


def test_live_preview_preserves_ema_context_and_original_training_weights():
    generator = torch.nn.Conv2d(1, 1, 1, bias=False)
    generator.weight.data.fill_(1)
    ema = ExponentialMovingAverage(generator, decay=0.99, device='cpu')
    ema.shadow_params['weight'].fill_(2)
    model = SimpleNamespace(generator=generator, ema=ema, _ema_applied=False)
    for name in ('_apply_generator_ema_weights', '_restore_generator_weights', '_live_validation_prediction'):
        setattr(model, name, MethodType(getattr(SRGAN_model, name), model))
    image = torch.ones(1, 1, 2, 2)
    model._apply_generator_ema_weights()
    live = model._live_validation_prediction(image)
    torch.testing.assert_close(live, image)
    assert model._ema_applied
    torch.testing.assert_close(generator(image), image * 2)
    model._restore_generator_weights()
    torch.testing.assert_close(generator(image), image)
    assert not ema.collected_params


def test_validation_logs_labeled_ema_live_galleries(monkeypatch, tmp_path):
    import importlib
    module = importlib.import_module('opensr_srgan.model.SRGAN')
    captured = []
    monkeypatch.setattr(module.wandb, 'Image', lambda path, caption: {'path': str(path), 'caption': caption})
    config = SimpleNamespace(
        Training=SimpleNamespace(Losses=SimpleNamespace()),
        Schedulers=SimpleNamespace(),
        Logging=SimpleNamespace(local_images=True, log_live_and_ema=True,
            num_val_images=1, output_dir=str(tmp_path), wandb=SimpleNamespace(enabled=True)),
    )
    model = SimpleNamespace(config=config, current_epoch=0, _ema_applied=True,
        forward=lambda image: torch.full((1, 4, 16, 16), 0.2),
        postprocess=lambda sr, lr: sr + 0.01,
        _live_validation_prediction=lambda image: torch.full((1, 4, 16, 16), 0.3),
        content_loss_criterion=SimpleNamespace(return_metrics=lambda *args, **kwargs: {'val_metrics/l1': torch.tensor(0.01)}),
        trainer=SimpleNamespace(sanity_checking=False, is_global_zero=True),
        logger=SimpleNamespace(experiment=SimpleNamespace(log=captured.append)),
        log=lambda *args, **kwargs: None, _pretrain_check=lambda: True,
    )
    SRGAN_model.validation_step(model, (torch.rand(1, 4, 4, 4), torch.rand(1, 4, 16, 16)), 0)
    assert len(captured) == 1
    assert set(captured[0]) == {'validation/epoch', 'validation/ema_live_rgb', 'validation/ema_live_nir'}
    assert captured[0]['validation/ema_live_rgb'][0]['path'].endswith('_rgb.png')
