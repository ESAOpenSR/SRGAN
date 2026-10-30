"""Keep quality selection separate from latest/full-state checkpoint storage."""

from pathlib import Path
import math

from pytorch_lightning.callbacks import Callback, ModelCheckpoint


def texture_checkpoint_eligible(model, metrics) -> bool:
    """Gate texture selection using phase and the global epoch-average L1."""
    if model._pretrain_check():
        return False
    start = max(0, model.g_pretrain_steps) if model.pretrain_g_only else 0
    if model.global_step < start + max(0, model.adv_loss_ramp_steps):
        return False
    losses = model.config.Training.Losses
    maximum = getattr(losses, "texture_selection_max_l1", None)
    if maximum is not None:
        l1 = metrics.get("val_metrics/l1")
        if l1 is None or not math.isfinite(float(l1)) or float(l1) > float(maximum):
            return False
    return True


class QualityModelCheckpoint(ModelCheckpoint):
    """Select top-k models only when the texture run is eligible.

    The monitored metric stays on its natural scale. Rejected epochs are
    skipped, rather than inserting artificial penalties into averaged metrics.
    Latest/final storage is handled independently by LatestCheckpoint.
    """

    def load_state_dict(self, state_dict):
        # Lightning discards selection history when the output directory changes.
        # A continuation in a fresh run directory must retain its quality history,
        # while treating historical checkpoint files as read-only references.
        state = dict(state_dict)
        paths = set(state.get("best_k_models", {})) | {state.get("best_model_path", "")}
        self._historical_paths = {
            path for path in paths if path and self.dirpath is not None
            and Path(path).resolve().parent != Path(self.dirpath).resolve()
        }
        if state.get("dirpath", self.dirpath) != self.dirpath:
            state["dirpath"] = self.dirpath
        super().load_state_dict(state)

    def _remove_checkpoint(self, trainer, filepath):
        if filepath in getattr(self, "_historical_paths", set()):
            return
        super()._remove_checkpoint(trainer, filepath)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        schedulers = pl_module.config.Schedulers
        texture_metric = getattr(
            schedulers, "texture_selection_metric", "val_metrics/texture_selection"
        )
        if self.monitor == texture_metric:
            eligible = texture_checkpoint_eligible(pl_module, trainer.callback_metrics)
            if trainer.is_global_zero:
                for logger in trainer.loggers:
                    logger.log_metrics(
                        {"val_metrics/texture_selection_eligible": float(eligible),
                         "training/optimizer_step": trainer.global_step,
                         "epoch": trainer.current_epoch},
                        # Let the logger keep its native step axis. In manual
                        # GAN optimization Lightning's logged batch step can
                        # differ from the D+G optimizer counter recorded above.
                        step=None,
                    )
            if not eligible:
                return
        super().on_validation_end(trainer, pl_module)


class LatestCheckpoint(ModelCheckpoint):
    """Always save latest after validation and final state at successful end.

    Trainer.save_checkpoint is called on every DDP rank so strategy-specific
    state collection remains correct. The strategy handles rank-zero writing.
    This does not depend on a top-k improvement or a monitored metric.
    """

    def __init__(self, directory):
        # Lightning places ModelCheckpoint callbacks last, preserving their
        # relative order. Appending this after quality selection ensures that
        # its full checkpoint contains the newly updated top-k callback state.
        super().__init__(dirpath=directory, monitor=None, save_top_k=0,
                         save_last=False, every_n_epochs=0,
                         save_on_train_epoch_end=False)
        self.directory = Path(directory)

    def _save(self, trainer, name):
        if trainer.is_global_zero:
            self.directory.mkdir(parents=True, exist_ok=True)
        if name == "last.ckpt":
            self.last_model_path = str(self.directory / name)
        trainer.save_checkpoint(str(self.directory / name), weights_only=False)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._save(trainer, "last.ckpt")

    def on_train_end(self, trainer, pl_module):
        self._save(trainer, "last.ckpt")
        self._save(trainer, "final.ckpt")


class MilestoneCheckpoint(Callback):
    """Preserve named full-state snapshots at explicit optimizer update counts."""

    def __init__(self, directory, optimizer_steps):
        self.directory = Path(directory)
        self.steps = {int(step) for step in optimizer_steps}
        if any(step <= 0 for step in self.steps):
            raise ValueError("Milestone optimizer steps must be positive")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.steps:
            if trainer.is_global_zero:
                self.directory.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(self.directory / f"optimizer-step-{trainer.global_step:08d}.ckpt"))
