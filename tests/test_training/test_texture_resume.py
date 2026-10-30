"""Exercise the promoted full-state continuation through the shared trainer."""
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]


def assert_same_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_same_state(value, reference)
    else:
        assert actual == expected


def test_full_texture_state_is_restored_before_first_resumed_update(monkeypatch, tmp_path):
    import importlib
    train_module = importlib.import_module('opensr_srgan.train')
    builder_module = importlib.import_module('opensr_srgan.utils.build_trainer_kwargs')
    loss_module = importlib.import_module('opensr_srgan.model.loss')
    dataset_module = importlib.import_module('opensr_srgan.data.dataset_selector')
    from opensr_srgan.utils.training_callbacks import QualityModelCheckpoint

    # Keep the real VGG/local-Gram objective, without fetching pretrained weights.
    content_loss = loss_module.GeneratorContentLoss
    monkeypatch.setattr(loss_module, 'GeneratorContentLoss', lambda cfg: content_loss(cfg, testing=True))
    lr = torch.linspace(.1, .7, 4 * 4 * 8 * 8).reshape(4, 4, 8, 8)
    hr = torch.nn.functional.interpolate(lr, scale_factor=4, mode='nearest')
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    class TinyData(pl.LightningDataModule):
        def train_dataloader(self):
            return loader
        def val_dataloader(self):
            return loader

    monkeypatch.setattr(dataset_module, 'select_dataset', lambda cfg: TinyData())
    restored = []
    expected = None

    class Audit(Callback):
        def on_train_start(self, trainer, model):
            if expected is not None:
                assert trainer.global_step == expected['global_step'] == 4
                assert_same_state(model.state_dict(), expected['state_dict'])
                ema_state = model.ema.state_dict()
                assert_same_state(ema_state['shadow_params'], expected['ema_state']['shadow_params'])
                assert ema_state['num_updates'] == expected['ema_state']['num_updates']
                for optimizer, state in zip(trainer.optimizers, expected['optimizer_states']):
                    assert_same_state(optimizer.state_dict(), state)
                for scheduler, state in zip(trainer.lr_scheduler_configs, expected['lr_schedulers']):
                    assert_same_state(scheduler.scheduler.state_dict(), state)
                quality = next(c for c in trainer.callbacks if isinstance(c, QualityModelCheckpoint))
                previous = expected['callbacks'][quality.state_key]
                assert_same_state(quality.best_model_score, previous['best_model_score'])
                assert_same_state(quality.best_k_models, previous['best_k_models'])
                restored.append(True)

    original_builder = builder_module.build_lightning_kwargs
    def builder(**kwargs):
        trainer_kwargs, fit_kwargs = original_builder(**kwargs)
        trainer_kwargs.update(enable_progress_bar=False, enable_model_summary=False)
        trainer_kwargs['callbacks'].append(Audit())
        return trainer_kwargs, fit_kwargs
    monkeypatch.setattr(builder_module, 'build_lightning_kwargs', builder)

    cfg = OmegaConf.load(ROOT / 'opensr_srgan/configs/config_sen2naip_texture.yaml')
    cfg.Model.continue_training = False  # Create only the synthetic source checkpoint.
    cfg.Data.sen2naip_taco_file = 'unused-by-test-datamodule'
    cfg.Data.sen2naip_split_file = 'unused-by-test-datamodule'
    cfg.Generator.n_channels = 4
    cfg.Generator.n_blocks = 1
    cfg.Generator.growth_channels = 2
    cfg.Discriminator.n_blocks = 1
    cfg.Training.device = 'cpu'
    cfg.Training.gpus = []
    cfg.Training.precision = '32-true'
    cfg.Training.max_steps = 4
    cfg.Training.max_epochs = 4
    cfg.Training.num_sanity_val_steps = 0
    cfg.Training.val_check_interval = 1.0
    cfg.Training.limit_val_batches = 1
    cfg.Training.adv_loss_ramp_steps = 0
    cfg.Training.checkpoint_optimizer_steps = [4, 6]
    cfg.Training.Losses.texture_selection_max_l1 = 2.0
    cfg.Schedulers.g_warmup_steps = 2
    cfg.Logging.local_images = False
    cfg.Logging.num_val_images = 0
    cfg.Logging.output_dir = str(tmp_path / 'source')
    train_module.train(cfg)
    source = tmp_path / 'source' / 'final.ckpt'
    expected = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    assert expected['global_step'] == 4
    assert len(expected['optimizer_states']) == 2 and len(expected['lr_schedulers']) == 3
    assert any(not torch.equal(expected['state_dict']['generator.' + key], value)
               for key, value in expected['ema_state']['shadow_params'].items())
    before = source.read_bytes()
    cfg.Model.continue_training = str(source)
    cfg.Training.max_steps = 6
    cfg.Logging.output_dir = str(tmp_path / 'continued')
    train_module.train(cfg)
    assert restored == [True]
    assert source.read_bytes() == before
    final = torch.load(tmp_path / 'continued' / 'final.ckpt', map_location='cpu', weights_only=False, mmap=True)
    assert final['global_step'] == 6
    assert final['ema_state']['num_updates'] == expected['ema_state']['num_updates'] + 1
    assert (tmp_path / 'continued' / 'optimizer-step-00000006.ckpt').exists()
    for group_d, group_g in zip(final['optimizer_states'][0]['param_groups'], final['optimizer_states'][1]['param_groups']):
        assert group_d['lr'] == group_g['lr'] * .1
