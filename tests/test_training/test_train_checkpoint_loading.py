import importlib
import runpy
import sys
import types

import pytest
from omegaconf import OmegaConf


def _make_config(load_checkpoint=False, continue_training=False):
    return OmegaConf.create(
        {
            "Model": {
                "in_bands": 4,
                "load_checkpoint": load_checkpoint,
                "continue_training": continue_training,
            },
            "Training": {
                "gpus": [0],
            },
            "Schedulers": {
                "metric_g": "val_metrics/l1",
            },
            "Optimizers": {
                "gradient_clip_val": 0.0,
            },
            "Logging": {
                "wandb": {
                    "enabled": False,
                    "project": "unit-tests",
                    "entity": "unit",
                }
            },
        }
    )


@pytest.fixture()
def train_module(monkeypatch):
    original_module = importlib.import_module("opensr_srgan.train")
    original_namespace = original_module.__dict__.copy()
    state = {}

    # ------------------------------------------------------------------
    # Third-party stubs imported by opensr_srgan.train
    # ------------------------------------------------------------------
    wandb_mod = types.ModuleType("wandb")

    def _wandb_finish():
        state["wandb_finish_calls"] = state.get("wandb_finish_calls", 0) + 1

    wandb_mod.finish = _wandb_finish
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)

    pl_mod = types.ModuleType("pytorch_lightning")

    class DummyTrainer:
        def __init__(self, **kwargs):
            state["trainer_kwargs"] = kwargs

        def fit(self, model, datamodule=None, **kwargs):
            state["fit_call"] = {
                "model": model,
                "datamodule": datamodule,
                "kwargs": kwargs,
            }

    pl_mod.Trainer = DummyTrainer
    pl_mod.__version__ = "2.2.0"
    monkeypatch.setitem(sys.modules, "pytorch_lightning", pl_mod)

    loggers_mod = types.ModuleType("pytorch_lightning.loggers")

    class CSVLogger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class WandbLogger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def log_hyperparams(self, params):
            self.hyperparams = params

    loggers_mod.CSVLogger = CSVLogger
    loggers_mod.WandbLogger = WandbLogger
    monkeypatch.setitem(sys.modules, "pytorch_lightning.loggers", loggers_mod)

    callbacks_mod = types.ModuleType("pytorch_lightning.callbacks")

    class ModelCheckpoint:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    callbacks_mod.ModelCheckpoint = ModelCheckpoint
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", callbacks_mod)
    training_callbacks_mod = types.ModuleType("opensr_srgan.utils.training_callbacks")
    training_callbacks_mod.QualityModelCheckpoint = callbacks_mod.ModelCheckpoint
    training_callbacks_mod.LatestCheckpoint = lambda directory: ("latest", directory)
    training_callbacks_mod.MilestoneCheckpoint = (
        lambda directory, steps: ("milestone", directory, tuple(steps))
    )
    monkeypatch.setitem(sys.modules, "opensr_srgan.utils.training_callbacks", training_callbacks_mod)

    early_stopping_mod = types.ModuleType("pytorch_lightning.callbacks.early_stopping")

    class EarlyStopping:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    early_stopping_mod.EarlyStopping = EarlyStopping
    monkeypatch.setitem(
        sys.modules, "pytorch_lightning.callbacks.early_stopping", early_stopping_mod
    )

    # ------------------------------------------------------------------
    # Internal module stubs imported by train()
    # ------------------------------------------------------------------
    srgan_mod = types.ModuleType("opensr_srgan.model.SRGAN")

    class DummySRGANModel:
        instances = []

        def __init__(self, config=None, **kwargs):
            self.config = config
            self.kwargs = kwargs
            self.loaded_weights_calls = []
            DummySRGANModel.instances.append(self)

        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):  # pragma: no cover
            raise AssertionError(
                "train() must not use class-level load_from_checkpoint"
            )

        def load_weights_from_checkpoint(
            self, ckpt_path, strict=False, map_location=None
        ):
            self.loaded_weights_calls.append((ckpt_path, strict, map_location))

    srgan_mod.SRGAN_model = DummySRGANModel
    monkeypatch.setitem(sys.modules, "opensr_srgan.model.SRGAN", srgan_mod)
    state["model_class"] = DummySRGANModel

    dataset_mod = types.ModuleType("opensr_srgan.data.dataset_selector")

    def select_dataset(config):
        state["selected_dataset_config"] = config
        return "DUMMY_DATAMODULE"

    dataset_mod.select_dataset = select_dataset
    monkeypatch.setitem(sys.modules, "opensr_srgan.data.dataset_selector", dataset_mod)

    trainer_kwargs_mod = types.ModuleType("opensr_srgan.utils.build_trainer_kwargs")

    def build_lightning_kwargs(
        config, logger, checkpoint_callback, early_stop_callback, resume_ckpt=None
    ):
        state["resume_ckpt"] = resume_ckpt
        state["builder_logger"] = logger
        state["builder_checkpoint_callback"] = checkpoint_callback
        state["builder_early_stop_callback"] = early_stop_callback
        return {"max_epochs": 1}, (
            {"ckpt_path": resume_ckpt} if resume_ckpt is not None else {}
        )

    trainer_kwargs_mod.build_lightning_kwargs = build_lightning_kwargs
    monkeypatch.setitem(
        sys.modules, "opensr_srgan.utils.build_trainer_kwargs", trainer_kwargs_mod
    )

    gpu_rank_mod = types.ModuleType("opensr_srgan.utils.gpu_rank")
    gpu_rank_mod._is_global_zero = lambda: False
    monkeypatch.setitem(sys.modules, "opensr_srgan.utils.gpu_rank", gpu_rank_mod)

    train_mod = importlib.reload(importlib.import_module("opensr_srgan.train"))
    try:
        yield train_mod, state
    finally:
        # Restoring sys.modules alone leaves this cached module holding stubs.
        # Restore its namespace so subsequent integration tests use real training.
        train_mod.__dict__.clear()
        train_mod.__dict__.update(original_namespace)


def test_train_uses_instance_weight_loader_for_load_checkpoint(train_module):
    train_mod, state = train_module
    config = _make_config(load_checkpoint="weights.ckpt", continue_training=False)

    train_mod.train(config)

    model = state["model_class"].instances[0]
    assert model.loaded_weights_calls == [("weights.ckpt", False, None)]
    assert state["resume_ckpt"] is None
    assert state["fit_call"]["model"] is model
    assert state["fit_call"]["datamodule"] == "DUMMY_DATAMODULE"
    assert state["fit_call"]["kwargs"] == {}


def test_train_passes_continue_training_to_fit_checkpoint_path(train_module):
    train_mod, state = train_module
    config = _make_config(load_checkpoint=False, continue_training="resume.ckpt")

    train_mod.train(config)

    model = state["model_class"].instances[0]
    assert model.loaded_weights_calls == []
    assert state["resume_ckpt"] == "resume.ckpt"
    assert state["fit_call"]["kwargs"] == {"ckpt_path": "resume.ckpt"}


def test_train_rejects_load_and_continue_combination(train_module):
    train_mod, state = train_module
    config = _make_config(
        load_checkpoint="weights.ckpt", continue_training="resume.ckpt"
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        train_mod.train(config)

    assert state["model_class"].instances == []


def test_train_accepts_plain_dict_config(train_module):
    train_mod, state = train_module
    config = {
        "Model": {
            "in_bands": 4,
            "load_checkpoint": False,
            "continue_training": False,
        },
        "Training": {"gpus": [0]},
        "Schedulers": {"metric_g": "val_metrics/l1"},
        "Optimizers": {"gradient_clip_val": 0.0},
        "Logging": {
            "wandb": {"enabled": False, "project": "unit-tests", "entity": "unit"}
        },
    }

    train_mod.train(config)

    model = state["model_class"].instances[0]
    assert model.loaded_weights_calls == []
    assert state["resume_ckpt"] is None


def test_train_accepts_yaml_path_config(train_module, tmp_path):
    train_mod, state = train_module

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("""
Model:
  in_bands: 4
  load_checkpoint: false
  continue_training: false
Training:
  gpus: [0]
Schedulers:
  metric_g: val_metrics/l1
Optimizers:
  gradient_clip_val: 0.0
Logging:
  wandb:
    enabled: false
    project: unit-tests
    entity: unit
""")

    train_mod.train(str(cfg_path))

    assert state["fit_call"]["datamodule"] == "DUMMY_DATAMODULE"
    assert state["resume_ckpt"] is None


def test_train_rejects_invalid_config_type(train_module):
    train_mod, _ = train_module

    with pytest.raises(TypeError, match="Config must be"):
        train_mod.train(12345)


def test_train_uses_wandb_logger_and_saves_config_when_global_zero(
    train_module, monkeypatch, tmp_path
):
    train_mod, state = train_module
    monkeypatch.chdir(tmp_path)

    # Exercise the global-zero save branch.
    gpu_rank_module = sys.modules["opensr_srgan.utils.gpu_rank"]
    monkeypatch.setattr(gpu_rank_module, "_is_global_zero", lambda: True)

    config = _make_config(load_checkpoint=False, continue_training=False)
    config.Logging.wandb.enabled = True

    train_mod.train(config)

    assert state["builder_logger"].__class__.__name__ == "WandbLogger"
    assert state["builder_logger"].hyperparams["Logging"]["wandb"]["enabled"] is True
    config_files = list((tmp_path / "logs" / "unit-tests").glob("*/config.yaml"))
    assert (
        config_files
    ), "expected config.yaml to be written in logs/unit-tests/<timestamp>/"


def test_train_respects_logging_output_dir(train_module, monkeypatch, tmp_path):
    train_mod, state = train_module

    gpu_rank_module = sys.modules["opensr_srgan.utils.gpu_rank"]
    monkeypatch.setattr(gpu_rank_module, "_is_global_zero", lambda: True)

    output_dir = tmp_path / "hydra-run"
    config = _make_config(load_checkpoint=False, continue_training=False)
    config.Logging.output_dir = str(output_dir)

    train_mod.train(config)

    assert (output_dir / "config.yaml").is_file()
    assert state["builder_checkpoint_callback"].kwargs["dirpath"] == str(output_dir)
    assert state["builder_checkpoint_callback"].kwargs["save_on_train_epoch_end"] is False
    assert state["builder_early_stop_callback"].kwargs["check_on_train_epoch_end"] is False
    assert state["builder_logger"].kwargs["save_dir"] == str(output_dir)


def test_train_registers_texture_latest_and_milestone_callbacks(train_module, tmp_path):
    train_mod, state = train_module
    config = _make_config(load_checkpoint=False, continue_training=False)
    config.Logging.output_dir = str(tmp_path / "texture-run")
    config.Schedulers.texture_selection_metric = "val_metrics/texture_selection"
    config.Training.checkpoint_optimizer_steps = [10, 20]

    train_mod.train(config)

    checkpoint = state["builder_checkpoint_callback"]
    assert checkpoint.kwargs["save_last"] is False
    assert state["trainer_kwargs"]["callbacks"] == [
        ("latest", str(tmp_path / "texture-run")),
        ("milestone", str(tmp_path / "texture-run"), (10, 20)),
    ]


def test_train_seeds_workers_and_finishes_enabled_wandb_run(train_module, tmp_path):
    train_mod, state = train_module
    seeds = []
    train_mod.pl.seed_everything = lambda seed, workers: seeds.append((seed, workers))
    config = _make_config(load_checkpoint=False, continue_training=False)
    config.Training.seed = 17
    config.Logging.wandb.enabled = True
    config.Logging.wandb.name = "run-name"
    config.Logging.wandb.group = "release"
    config.Logging.output_dir = str(tmp_path / "wandb-run")

    train_mod.train(config)

    assert seeds == [(17, True)]
    assert state["builder_logger"].kwargs["name"] == "run-name"
    assert state["builder_logger"].kwargs["group"] == "release"
    assert state["builder_logger"].kwargs["save_dir"] == str(tmp_path / "wandb-run")
    assert state["wandb_finish_calls"] == 1


def test_train_module_main_guard(monkeypatch):
    state = {}

    wandb_mod = types.ModuleType("wandb")
    wandb_mod.finish = lambda: state.setdefault("wandb_finished", True)
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)

    pl_mod = types.ModuleType("pytorch_lightning")

    class DummyTrainer:
        def __init__(self, **kwargs):
            state["trainer_kwargs"] = kwargs

        def fit(self, model, datamodule=None, **kwargs):
            state["fit_call"] = {
                "model": model,
                "datamodule": datamodule,
                "kwargs": kwargs,
            }

    pl_mod.Trainer = DummyTrainer
    pl_mod.__version__ = "2.2.0"
    monkeypatch.setitem(sys.modules, "pytorch_lightning", pl_mod)

    loggers_mod = types.ModuleType("pytorch_lightning.loggers")

    class CSVLogger:
        def __init__(self, *args, **kwargs):
            pass

    class WandbLogger:
        def __init__(self, *args, **kwargs):
            pass

    loggers_mod.CSVLogger = CSVLogger
    loggers_mod.WandbLogger = WandbLogger
    monkeypatch.setitem(sys.modules, "pytorch_lightning.loggers", loggers_mod)

    callbacks_mod = types.ModuleType("pytorch_lightning.callbacks")
    callbacks_mod.ModelCheckpoint = lambda *args, **kwargs: "checkpoint"
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", callbacks_mod)
    training_callbacks_mod = types.ModuleType("opensr_srgan.utils.training_callbacks")
    training_callbacks_mod.QualityModelCheckpoint = callbacks_mod.ModelCheckpoint
    training_callbacks_mod.LatestCheckpoint = lambda directory: ("latest", directory)
    monkeypatch.setitem(sys.modules, "opensr_srgan.utils.training_callbacks", training_callbacks_mod)

    early_stopping_mod = types.ModuleType("pytorch_lightning.callbacks.early_stopping")
    early_stopping_mod.EarlyStopping = lambda *args, **kwargs: "early_stop"
    monkeypatch.setitem(
        sys.modules, "pytorch_lightning.callbacks.early_stopping", early_stopping_mod
    )

    srgan_mod = types.ModuleType("opensr_srgan.model.SRGAN")

    class DummySRGANModel:
        def __init__(self, config=None, **kwargs):
            self.config = config

        def load_weights_from_checkpoint(self, *args, **kwargs):
            return None

    srgan_mod.SRGAN_model = DummySRGANModel
    monkeypatch.setitem(sys.modules, "opensr_srgan.model.SRGAN", srgan_mod)

    dataset_mod = types.ModuleType("opensr_srgan.data.dataset_selector")
    dataset_mod.select_dataset = lambda config: "DM"
    monkeypatch.setitem(sys.modules, "opensr_srgan.data.dataset_selector", dataset_mod)

    trainer_kwargs_mod = types.ModuleType("opensr_srgan.utils.build_trainer_kwargs")
    trainer_kwargs_mod.build_lightning_kwargs = lambda **kwargs: ({"max_epochs": 1}, {})
    monkeypatch.setitem(
        sys.modules, "opensr_srgan.utils.build_trainer_kwargs", trainer_kwargs_mod
    )

    gpu_rank_mod = types.ModuleType("opensr_srgan.utils.gpu_rank")
    gpu_rank_mod._is_global_zero = lambda: False
    monkeypatch.setitem(sys.modules, "opensr_srgan.utils.gpu_rank", gpu_rank_mod)

    monkeypatch.setattr(
        "omegaconf.OmegaConf.load",
        lambda _path: _make_config(load_checkpoint=False, continue_training=False),
    )
    monkeypatch.setattr(sys, "argv", ["train.py"])

    runpy.run_module("opensr_srgan.train", run_name="__main__")

    assert state["fit_call"]["datamodule"] == "DM"
