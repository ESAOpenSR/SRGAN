"""
Training entry point for SRGAN using PyTorch Lightning.

- Accepts a YAML path, dict, or OmegaConf for configuration.
- Builds/loads the model, constructs the DataModule, configures logging,
  checkpointing, early stopping, and launches training.
"""

import datetime
import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
import pytorch_lightning as pl


def _is_path_set(value) -> bool:
    """Return True when an optional path-like config value should be used."""

    return value not in (False, None, "")


def _resolve_experiment_dir(config) -> str:
    """Resolve the directory used for checkpoints and the saved config."""

    configured_output_dir = getattr(config.Logging, "output_dir", None)
    if _is_path_set(configured_output_dir):
        return os.path.normpath(os.path.expanduser(str(configured_output_dir)))

    return os.path.join(
        os.path.normpath("logs/"),
        config.Logging.wandb.project,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )


def train(config):
    """
    Train SRGAN from a configuration.

    Parameters
    ----------
    config : str | Path | dict | OmegaConf
        Path to a YAML file, a Python dict, or an OmegaConf object describing
        Model, Data, Training, Logging, and Schedulers sections.

    Notes
    -----
    - Requires PyTorch Lightning >= 2.0.
    - If `Model.load_checkpoint` is set, weights are loaded before training.
    - If `Model.continue_training` is set, training resumes from that checkpoint.
    - Setting both `Model.load_checkpoint` and `Model.continue_training` is invalid.
    """
    # either path to config file or omegaconf object

    if isinstance(config, str) or isinstance(config, Path):
        config = OmegaConf.load(config)
    elif isinstance(config, dict):
        config = OmegaConf.create(config)
    elif OmegaConf.is_config(config):
        pass
    else:
        raise TypeError(
            "Config must be a filepath (str or Path), dict, or OmegaConf object."
        )

    seed = getattr(config.Training, "seed", None)
    if seed is not None:
        seed_everything = getattr(pl, "seed_everything", None)
        if seed_everything is not None:
            seed_everything(int(seed), workers=True)
    #############################################################################################################

    " LOAD MODEL "
    #############################################################################################################
    model_load_checkpoint = getattr(config.Model, "load_checkpoint", False)
    resume_checkpoint = getattr(config.Model, "continue_training", False)

    def _checkpoint_is_set(value) -> bool:
        return value not in (False, None, "")

    if _checkpoint_is_set(model_load_checkpoint) and _checkpoint_is_set(
        resume_checkpoint
    ):
        raise ValueError(
            "Model.load_checkpoint and Model.continue_training are mutually exclusive. "
            "Use Model.load_checkpoint for weight initialization only, or "
            "Model.continue_training to fully resume optimizer/scheduler/training state."
        )

    # load pretrained weights or instantiate new
    from opensr_srgan.model.SRGAN import SRGAN_model

    model = SRGAN_model(config=config)
    if _checkpoint_is_set(model_load_checkpoint):
        model.load_weights_from_checkpoint(model_load_checkpoint, strict=False)

    resume_ckpt = resume_checkpoint if _checkpoint_is_set(resume_checkpoint) else None

    #############################################################################################################
    """ GET DATA """
    #############################################################################################################
    # create dataloaders via dataset_selector -> config -> class selection -> convert to pl_module
    from opensr_srgan.data.dataset_selector import select_dataset

    pl_datamodule = select_dataset(config)

    #############################################################################################################
    """ Configure Trainer """
    #############################################################################################################

    output_dir_is_set = _is_path_set(getattr(config.Logging, "output_dir", None))
    experiment_dir = _resolve_experiment_dir(config)

    # Configure Logger
    if config.Logging.wandb.enabled:
        # set up logging
        from pytorch_lightning.loggers import WandbLogger

        wandb_project = config.Logging.wandb.project  # whatever you want
        wandb_kwargs = {
            "project": wandb_project,
            "entity": config.Logging.wandb.entity,
            "log_model": False,
        }
        for key in ("name", "group", "id", "resume"):
            value = getattr(config.Logging.wandb, key, None)
            if value:
                wandb_kwargs[key] = str(value)
        if output_dir_is_set:
            wandb_kwargs["save_dir"] = experiment_dir
        wandb_logger = WandbLogger(**wandb_kwargs)
        wandb_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    else:
        print("Not using Weights & Biases logging, reduced CSV logs written locally.")
        from pytorch_lightning.loggers import CSVLogger

        csv_save_dir = experiment_dir if output_dir_is_set else "logs/"
        wandb_logger = CSVLogger(
            save_dir=csv_save_dir,
        )

    # Configure Saving Checkpoints
    from opensr_srgan.utils.training_callbacks import (
        LatestCheckpoint,
        QualityModelCheckpoint,
    )

    dir_save_checkpoints = experiment_dir
    from opensr_srgan.utils.gpu_rank import (
        _is_global_zero,
    )  # make dir only on main process

    if _is_global_zero():  # only on main process
        os.makedirs(dir_save_checkpoints, exist_ok=True)
        print("Experiment Path:", dir_save_checkpoints)
        with open(
            os.path.join(dir_save_checkpoints, "config.yaml"), "w"
        ) as f:  # save config to experiment folder
            OmegaConf.save(config, f)
    from pytorch_lightning.callbacks import ModelCheckpoint
    texture_selection = bool(getattr(config.Schedulers, "texture_selection_metric", None)) or (
        getattr(config.Schedulers, "checkpoint_monitor", None) == "val_metrics/texture_selection"
    )
    checkpoint_type = QualityModelCheckpoint if texture_selection else ModelCheckpoint
    checkpoint_callback = checkpoint_type(
        dirpath=dir_save_checkpoints,
        monitor=getattr(
            config.Schedulers,
            "checkpoint_monitor",
            getattr(config.Schedulers, "metric_g", "val_metrics/l1"),
        ),
        mode=str(getattr(config.Schedulers, "checkpoint_mode", "min")),
        save_last=not texture_selection,  # Texture latest/final is stored independently.
        save_top_k=int(getattr(config.Schedulers, "save_top_k", 2)),
        # Validation metrics are produced at validation end. Saving at
        # train-epoch end can race the final validation callback on resume.
        save_on_train_epoch_end=False,
    )

    # callback to set up early stopping
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    early_stop_callback = EarlyStopping(
        monitor=getattr(
            config.Schedulers,
            "early_stopping_monitor",
            getattr(config.Schedulers, "metric_g", "val_metrics/l1"),
        ),
        min_delta=0.00,
        patience=int(getattr(config.Schedulers, "early_stopping_patience", 250)),
        verbose=True,
        mode=str(
            getattr(
                config.Schedulers,
                "early_stopping_mode",
                getattr(config.Schedulers, "checkpoint_mode", "min"),
            )
        ),
        check_finite=True,
        # The monitored metric is validation-only; checking at train epoch end
        # sees an empty callback_metrics mapping in full-state resumes.
        check_on_train_epoch_end=False,
    )  # patience in epochs

    #############################################################################################################
    """ Set Args for Training and Start Training """
    """ Build trainer kwargs and launch training """
    #############################################################################################################
    from opensr_srgan.utils.build_trainer_kwargs import build_lightning_kwargs

    trainer_kwargs, fit_kwargs = build_lightning_kwargs(
        config=config,
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        resume_ckpt=resume_ckpt,
    )
    if texture_selection:
        trainer_kwargs["callbacks"] = list(trainer_kwargs.get("callbacks", [])) + [
            LatestCheckpoint(experiment_dir)
        ]
    milestone_steps = getattr(config.Training, "checkpoint_optimizer_steps", None)
    if milestone_steps:
        from opensr_srgan.utils.training_callbacks import MilestoneCheckpoint
        trainer_kwargs["callbacks"].append(MilestoneCheckpoint(experiment_dir, milestone_steps))
    # Start training
    trainer = pl.Trainer(**trainer_kwargs)
    try:
        trainer.fit(model, datamodule=pl_datamodule, **fit_kwargs)
    finally:
        if config.Logging.wandb.enabled:
            wandb.finish()


# Run training if called from command line
if __name__ == "__main__":
    """
    CLI entry point.

    Usage
    -----
    python -m opensr_srgan.train --config path/to/config.yaml

    Arguments
    ---------
    --config, -c : str
        Path to YAML config. Defaults to `opensr_srgan/configs/config_10m.yaml`.
    """

    import argparse
    from multiprocessing import freeze_support

    # required for Multiprocessing on Windows
    freeze_support()

    # ---- CLI: single positional argument (config path) ----
    parser = argparse.ArgumentParser(description="Train SRGAN with a YAML config.")
    try:
        default_config = Path(__file__).resolve().parent / "configs" / "config_10m.yaml"
    except NameError:
        default_config = Path.cwd() / "opensr_srgan" / "configs" / "config_10m.yaml"
    parser.add_argument(
        "--config",
        "-c",
        default=str(default_config),
        help=f"Path to YAML config file (default: {default_config})",
    )
    args = parser.parse_args()

    # General
    torch.set_float32_matmul_precision("medium")
    # load config
    cfg_filepath = args.config

    # Run training
    train(cfg_filepath)
