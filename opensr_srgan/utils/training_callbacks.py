"""Keep quality selection separate from latest/full-state checkpoint storage."""

from pathlib import Path
import math
import json
import time
from datetime import datetime, timezone

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
    for option, metric, is_minimum in (
        ("texture_selection_max_l1_rgb", "l1_rgb", False),
        ("texture_selection_max_l1_nir", "l1_nir", False),
        ("texture_selection_max_sam", "sam", False),
        ("texture_selection_max_perceptual_rgb", "perceptual", False),
        ("texture_selection_max_perceptual_nir", "perceptual_nir", False),
        ("texture_selection_min_ssim_rgb", "ssim_rgb", True),
        ("texture_selection_min_ssim_nir", "ssim_nir", True),
    ):
        bound = getattr(losses, option, None)
        if bound is not None:
            value = metrics.get("val_metrics/" + metric)
            if value is None or not math.isfinite(float(value)):
                return False
            if (is_minimum and float(value) < float(bound)) or (
                not is_minimum and float(value) > float(bound)
            ):
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


class FixedRestartSchedule(Callback):
    """Set D/G rates from additional G updates, independent of validation scores."""

    def __init__(self, *, start_optimizer_step, peak_g_lr, floor_g_lr,
                 warmup_updates=500, hold_until_update=10000,
                 total_updates=30000, d_ratio=0.1):
        self.origin = int(start_optimizer_step)
        self.peak = float(peak_g_lr)
        self.floor = float(floor_g_lr)
        self.warmup = int(warmup_updates)
        self.hold = int(hold_until_update)
        self.total = int(total_updates)
        self.ratio = float(d_ratio)
        if self.origin < 0 or self.origin % 2:
            raise ValueError("Restart origin must be an even D+G optimizer counter")
        if not all(math.isfinite(x) and x > 0 for x in (self.peak, self.floor, self.ratio)):
            raise ValueError("Restart learning rates and ratio must be finite and positive")
        if self.peak < self.floor or not 0 <= self.warmup <= self.hold < self.total:
            raise ValueError("Invalid restart warmup/hold/decay bounds")

    def generator_lr(self, optimizer_step):
        if optimizer_step < self.origin:
            raise ValueError("Checkpoint precedes the restart origin")
        update = min(self.total, (int(optimizer_step) - self.origin) // 2)
        if self.warmup and update < self.warmup:
            return self.floor + (self.peak - self.floor) * update / self.warmup
        if update <= self.hold:
            return self.peak
        progress = (update - self.hold) / (self.total - self.hold)
        return self.floor + 0.5 * (self.peak - self.floor) * (1 + math.cos(math.pi * progress))

    def _apply(self, trainer, pl_module):
        rate = self.generator_lr(trainer.global_step)
        optimizer_d, optimizer_g = trainer.optimizers
        for optimizer, target in ((optimizer_d, rate * self.ratio), (optimizer_g, rate)):
            for group in optimizer.param_groups:
                group["lr"] = target
        for name, optimizer in (("_plateau_scheduler_d", optimizer_d),
                                ("_plateau_scheduler_g", optimizer_g),
                                ("_warmup_scheduler_g", optimizer_g)):
            scheduler = getattr(pl_module, name, None)
            if scheduler is not None:
                scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        self.last_rate = rate

    def on_train_start(self, trainer, pl_module):
        warmup = getattr(pl_module, "_warmup_scheduler_g", None)
        if warmup is not None and warmup.last_epoch < getattr(pl_module, "_generator_warmup_steps", 0):
            raise ValueError("Fixed restart requires the original G warmup to be complete")
        self._apply(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._apply(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.log("restart/generator_lr_used", self.last_rate, on_step=True, on_epoch=False)
        pl_module.log("restart/new_generator_updates", (trainer.global_step - self.origin) // 2,
                      on_step=True, on_epoch=False)

    def on_train_end(self, trainer, pl_module):
        self._apply(trainer, pl_module)


class DeadlineStop(Callback):
    """Stop at an absolute UTC deadline, reserving time for final full saves."""

    def __init__(self, stop_at_utc, save_margin_seconds=120):
        parsed = datetime.fromisoformat(str(stop_at_utc).replace("Z", "+00:00"))
        if parsed.tzinfo is None or save_margin_seconds < 0:
            raise ValueError("Deadline must include a timezone and margin must be nonnegative")
        self.deadline = parsed.timestamp()
        self.margin = float(save_margin_seconds)
        self.triggered = False

    def _check(self, trainer):
        reached = time.time() >= self.deadline - self.margin
        reached = trainer.strategy.broadcast(reached)
        if reached:
            self.triggered = True
            trainer.should_stop = True
            trainer.limit_val_batches = 0

    def on_fit_start(self, trainer, pl_module):
        self._check(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._check(trainer)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            path = Path(pl_module.config.Logging.output_dir) / "deadline-stop.json"
            path.write_text(json.dumps({
                "reason": "deadline" if self.triggered else "update_budget",
                "deadline_utc": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "optimizer_steps": trainer.global_step,
                "generator_updates": trainer.global_step // 2,
                "save_margin_seconds": self.margin,
            }, indent=2) + "\n")


class ConvergenceStop(Callback):
    """Stop only after adequate training and sustained validation plateau at LR floor."""

    def __init__(self, *, monitor, min_steps, lr_floor, patience=12, min_delta=0.001):
        self.monitor = monitor
        self.min_steps = int(min_steps)
        self.lr_floor = float(lr_floor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.floor_wait = 0
        self.last_step = -1
        self.converged = False

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ("best", "floor_wait", "last_step", "converged")}

    def load_state_dict(self, state_dict):
        for key in ("best", "floor_wait", "last_step", "converged"):
            if key in state_dict:
                setattr(self, key, state_dict[key])

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_step == self.last_step:
            return
        self.last_step = trainer.global_step
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            raise RuntimeError(f"Missing convergence metric: {self.monitor}")
        value = float(metric)
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite convergence metric: {self.monitor}")
        eligible = texture_checkpoint_eligible(pl_module, trainer.callback_metrics)
        if not eligible:
            self.floor_wait = 0
            return
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
        at_floor = all(group["lr"] <= self.lr_floor * (1 + 1e-6)
                       for group in trainer.optimizers[1].param_groups)
        if improved or not at_floor or trainer.global_step < self.min_steps:
            self.floor_wait = 0
        else:
            self.floor_wait += 1
        if self.floor_wait >= self.patience:
            self.converged = True
            trainer.should_stop = True
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.log_metrics({"convergence/floor_wait": self.floor_wait,
                                    "convergence/at_lr_floor": int(at_floor),
                                    "convergence/quality_plateau": int(self.converged),
                                    "training/optimizer_step": trainer.global_step}, step=None)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            path = Path(pl_module.config.Logging.output_dir) / "stopping.json"
            path.write_text(json.dumps({
                **self.state_dict(), "monitor": self.monitor,
                "reason": "validation_plateau_at_lr_floor" if self.converged else "compute_cap",
                "optimizer_steps": trainer.global_step,
                "generator_updates_approx": trainer.global_step // 2,
                "min_steps": self.min_steps, "max_steps": trainer.max_steps,
                "min_delta": self.min_delta, "floor_patience": self.patience,
            }, indent=2) + "\n")
