from types import SimpleNamespace

import pytorch_lightning as pl
import pytest
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import DataLoader, TensorDataset

from opensr_srgan.utils.training_callbacks import (
    LatestCheckpoint, QualityModelCheckpoint, texture_checkpoint_eligible,
)


def test_texture_gate_uses_epoch_average_without_penalties():
    model = SimpleNamespace(
        _pretrain_check=lambda: False, pretrain_g_only=False,
        g_pretrain_steps=0, adv_loss_ramp_steps=10, global_step=9,
        config=SimpleNamespace(Training=SimpleNamespace(
            Losses=SimpleNamespace(texture_selection_max_l1=0.05))),
    )
    assert not texture_checkpoint_eligible(model, {"val_metrics/l1": torch.tensor(0.02)})
    model.global_step = 10
    # A high-error individual batch does not invalidate a sound epoch average.
    assert texture_checkpoint_eligible(model, {"val_metrics/l1": torch.tensor(0.02)})
    assert not texture_checkpoint_eligible(model, {"val_metrics/l1": torch.tensor(0.06)})
    assert not texture_checkpoint_eligible(model, {"val_metrics/l1": torch.tensor(float("nan"))})
    assert not texture_checkpoint_eligible(model, {})


def test_plateau_scheduler_waits_for_ramp_and_uses_unpenalized_metric():
    from opensr_srgan.model.SRGAN import SRGAN_model

    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(Schedulers=SimpleNamespace(
            g_plateau_after_ramp=True, d_plateau_enabled=False)),
        trainer=SimpleNamespace(sanity_checking=False,
                                callback_metrics={"val_metrics/perceptual": torch.tensor(1.05)}),
        _pretrain_check=lambda: False, pretrain_g_only=False,
        g_pretrain_steps=0, adv_loss_ramp_steps=10, global_step=5,
        _plateau_scheduler_d=SimpleNamespace(step=lambda value: calls.append(("d", value))),
        _plateau_scheduler_g=SimpleNamespace(step=lambda value: calls.append(("g", value))),
        _plateau_metric_d="irrelevant", _plateau_metric_g="val_metrics/perceptual",
    )
    SRGAN_model._step_plateau_schedulers(model)
    assert calls == []
    model.global_step = 10
    SRGAN_model._step_plateau_schedulers(model)
    assert len(calls) == 1 and calls[0][0] == "g"
    torch.testing.assert_close(torch.tensor(calls[0][1]), torch.tensor(1.05))


def test_quality_callback_skips_invalid_epochs_without_changing_metric(monkeypatch):
    from pytorch_lightning.callbacks import ModelCheckpoint

    selected = []
    logged = []
    monkeypatch.setattr(ModelCheckpoint, "on_validation_end", lambda self, trainer, model: selected.append(model.global_step))
    model = SimpleNamespace(
        _pretrain_check=lambda: False, pretrain_g_only=False,
        g_pretrain_steps=0, adv_loss_ramp_steps=10, global_step=5,
        config=SimpleNamespace(
            Schedulers=SimpleNamespace(),
            Training=SimpleNamespace(Losses=SimpleNamespace(texture_selection_max_l1=0.05))),
    )
    trainer = SimpleNamespace(sanity_checking=False, is_global_zero=True, global_step=5, current_epoch=0,
                              callback_metrics={"val_metrics/l1": torch.tensor(0.02),
                                                "val_metrics/texture_selection": torch.tensor(1.05)},
                              loggers=[SimpleNamespace(log_metrics=lambda metrics, step: logged.append(metrics))])
    callback = QualityModelCheckpoint(monitor="val_metrics/texture_selection", save_top_k=1, save_last=False)
    callback.on_validation_end(trainer, model)
    assert selected == []
    model.global_step = trainer.global_step = 10
    callback.on_validation_end(trainer, model)
    assert selected == [10]
    trainer.callback_metrics["val_metrics/l1"] = torch.tensor(0.06)
    callback.on_validation_end(trainer, model)
    assert selected == [10]
    assert logged[-1]["val_metrics/texture_selection_eligible"] == 0
    torch.testing.assert_close(trainer.callback_metrics["val_metrics/texture_selection"], torch.tensor(1.05))


def test_coupled_lr_tracks_warmup_plateau_and_restored_optimizer_rates():
    from opensr_srgan.model.SRGAN import SRGAN_model

    d = torch.optim.Adam([torch.nn.Parameter(torch.ones(1)),
                          ], lr=5e-5)
    g = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(g, patience=0, factor=0.5)
    model = SimpleNamespace(
        config=SimpleNamespace(Schedulers=SimpleNamespace(d_lr_ratio_to_g=0.1)),
        trainer=SimpleNamespace(optimizers=[d, g], sanity_checking=False,
                                callback_metrics={"quality": torch.tensor(1.0)}),
        _pretrain_check=lambda: False, pretrain_g_only=False,
        g_pretrain_steps=0, adv_loss_ramp_steps=0, global_step=100,
        _plateau_scheduler_g=plateau, _plateau_metric_g="quality",
        _plateau_scheduler_d=SimpleNamespace(step=lambda value: pytest.fail("D must follow G")),
        _plateau_metric_d="quality",
    )
    model._sync_discriminator_lr = lambda: SRGAN_model._sync_discriminator_lr(model)
    # Includes warmup and values restored from an older checkpoint with stale D LR.
    for lr in (5e-6, 1e-4, 1e-6):
        g.param_groups[0]["lr"] = lr
        model._sync_discriminator_lr()
        assert d.param_groups[0]["lr"] == pytest.approx(lr * 0.1)
    g.param_groups[0]["lr"] = 1e-4
    SRGAN_model._step_plateau_schedulers(model)
    SRGAN_model._step_plateau_schedulers(model)
    assert g.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert d.param_groups[0]["lr"] == pytest.approx(5e-6)


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.config = SimpleNamespace(Schedulers=SimpleNamespace())

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        # Each later validation is deliberately worse, so best must stay old.
        self.log("quality", float(self.current_epoch))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def test_latest_advances_without_best_and_final_saves_between_validations(tmp_path):
    best = QualityModelCheckpoint(dirpath=tmp_path, monitor="quality", save_top_k=1,
                                  save_last=False, save_on_train_epoch_end=False)
    validation_steps = []

    class InspectLatest(LatestCheckpoint):
        def on_validation_end(self, trainer, pl_module):
            super().on_validation_end(trainer, pl_module)
            checkpoint = torch.load(tmp_path / "last.ckpt", weights_only=False)
            validation_steps.append(checkpoint["global_step"])
            assert checkpoint["callbacks"][best.state_key]["best_model_path"] == best.best_model_path

    data = DataLoader(TensorDataset(torch.ones(3, 1)), batch_size=1)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=14, max_epochs=10,
                         logger=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, check_val_every_n_epoch=2,
                         limit_val_batches=1, callbacks=[best, InspectLatest(tmp_path)])
    model = TinyModel()
    trainer.fit(model, train_dataloaders=data, val_dataloaders=data)
    assert validation_steps == [6, 12]
    assert torch.load(best.best_model_path, weights_only=False)["global_step"] == 6
    for name in ("last.ckpt", "final.ckpt"):
        checkpoint = torch.load(tmp_path / name, weights_only=False)
        assert checkpoint["global_step"] == 14
        assert len(checkpoint["optimizer_states"]) == 1
        for key, value in model.state_dict().items():
            torch.testing.assert_close(checkpoint["state_dict"][key], value)

    # Prove the unconditional final checkpoint is a full resumable checkpoint.
    resumed = pl.Trainer(accelerator="cpu", devices=1, max_steps=16, max_epochs=10,
                         logger=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, enable_checkpointing=False)
    resumed.fit(TinyModel(), train_dataloaders=data, val_dataloaders=data,
                ckpt_path=str(tmp_path / "final.ckpt"))
    assert resumed.global_step == 16


def test_quality_history_survives_resume_in_a_fresh_output_directory(tmp_path):
    old = QualityModelCheckpoint(dirpath=tmp_path / 'old', monitor='quality', save_top_k=2)
    old.best_model_score = torch.tensor(.4)
    old.best_model_path = str(tmp_path / 'old' / 'best.ckpt')
    old.best_k_models = {old.best_model_path: old.best_model_score}
    old.kth_best_model_path = old.best_model_path
    old.kth_value = torch.tensor(.4)
    restored = QualityModelCheckpoint(dirpath=tmp_path / 'new', monitor='quality', save_top_k=2)
    restored.load_state_dict(old.state_dict())
    torch.testing.assert_close(restored.best_model_score, old.best_model_score)
    assert restored.best_k_models == old.best_k_models
    removed = []
    trainer = SimpleNamespace(strategy=SimpleNamespace(remove_checkpoint=removed.append))
    restored._remove_checkpoint(trainer, old.best_model_path)
    assert removed == []
