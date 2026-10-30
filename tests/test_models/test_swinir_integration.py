"""Exercise transformer registration through config, GAN training and inference."""
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from opensr_srgan.model.generators import build_generator


ROOT = Path(__file__).resolve().parents[2]


def small_generator(model_type="swinir"):
    return {"model_type": model_type, "embed_dim": 12, "depths": [2],
            "num_heads": [3], "window_size": 4, "scaling_factor": 4,
            "use_checkpoint": False, "qkv_bias": True}


@pytest.mark.parametrize("alias", ["swinir", "SwinIR", "swin-ir", "swin_ir", "transformer"])
def test_swinir_factory_aliases_and_public_export(alias):
    from opensr_srgan.model.generators import SwinIRGenerator

    config = OmegaConf.create({"Model": {"in_bands": 4}, "Generator": small_generator(alias)})
    generator = build_generator(config)
    assert isinstance(generator, SwinIRGenerator)
    with torch.no_grad():
        assert generator(torch.rand(1, 4, 7, 9)).shape == (1, 4, 28, 36)


def test_swinir_factory_reports_inherited_cnn_options(capsys):
    config = OmegaConf.create({"Model": {"in_bands": 4}, "Generator": {
        **small_generator(), "n_channels": 64, "n_blocks": 23, "growth_channels": 32,
    }})
    generator = build_generator(config)
    assert generator.embed_dim == 12
    notice = capsys.readouterr().out
    assert "[Generator:swinir]" in notice
    assert "growth_channels, n_blocks, n_channels" in notice


def test_swinir_hydra_presets_compose_and_build():
    config_dir = ROOT / "opensr_srgan/configs/hydra"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = hydra.compose(config_name="train", overrides=["experiment=swinir_sen2naip"])
    assert config.Generator.model_type == "swinir"
    assert config.Model.in_bands == 4
    assert config.Generator.scaling_factor == 4
    assert config.Training.pretrain_g_only
    assert config.Training.g_pretrain_steps == -1
    assert config.Training.Losses.perceptual_weight == 0
    assert config.Optimizers.optim_g_lr == pytest.approx(2e-4)
    assert list(config.Optimizers.betas) == [.9, .99]
    # Build the real preset, but use a tiny image for a CPU-only forward check.
    generator = build_generator(config).eval()
    assert sum(p.numel() for p in generator.parameters()) > 1_000_000
    with torch.no_grad():
        result = generator(torch.rand(1, 4, 8, 8))
    assert result.shape == (1, 4, 32, 32)
    assert torch.isfinite(result).all()


@pytest.mark.filterwarnings("ignore:.*does not have many workers.*")
@pytest.mark.filterwarnings("ignore:.*isinstance\\(treespec, LeafSpec\\).*deprecated.*")
def test_swinir_manual_gan_training_ema_checkpoint_and_inference(tmp_path, capsys):
    from opensr_srgan.model.SRGAN import SRGAN_model
    from opensr_srgan.inference import load_model

    config = OmegaConf.load(ROOT / "opensr_srgan/configs/config_training_example.yaml")
    config.Generator = small_generator()
    config.Discriminator.model_type = "patchgan"
    config.Discriminator.n_blocks = 1
    config.Training.EMA.enabled = True
    config.Training.Losses.perceptual_weight = 0.0
    config.Training.device = "cpu"
    config.Training.gpus = []
    config.Training.pretrain_g_only = False
    config.Training.g_pretrain_steps = 0
    config.Training.adv_loss_ramp_steps = 0
    config.Training.Losses.adv_loss_beta = .005
    config.Training.Losses.ssim_win = 3
    config.Schedulers.g_warmup_steps = 0
    config.Logging.num_val_images = 0
    config.Logging.local_images = False
    config.Logging.output_dir = str(tmp_path)
    model = SRGAN_model(config=config)
    initial_g = {k: v.clone() for k, v in model.generator.state_dict().items()}
    initial_d = {k: v.clone() for k, v in model.discriminator.state_dict().items()}
    loader = DataLoader(TensorDataset(torch.rand(4, 4, 8, 8), torch.rand(4, 4, 32, 32)), batch_size=2)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=4, max_epochs=1,
                         limit_val_batches=1, num_sanity_val_steps=0, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 4
    assert any(not torch.equal(v, model.generator.state_dict()[k]) for k, v in initial_g.items())
    assert any(not torch.equal(v, model.discriminator.state_dict()[k]) for k, v in initial_d.items())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert "SwinIR (Residual Shifted-Window Transformer)" in capsys.readouterr().out
    checkpoint_path = tmp_path / "trained.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "ema_state" in checkpoint and len(checkpoint["optimizer_states"]) == 2
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    restored, device = load_model(config_path, checkpoint_path, device="cpu")
    assert str(device) == "cpu" and restored.ema is not None
    lr = torch.rand(1, 4, 7, 9)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.generator(lr), restored.generator(lr))
        prediction = restored.predict_step(lr)
    assert prediction.shape == (1, 4, 28, 36) and torch.isfinite(prediction).all()
