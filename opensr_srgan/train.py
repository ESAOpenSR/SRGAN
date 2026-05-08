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

    # Configure Logger
    if config.Logging.wandb.enabled:
        # set up logging
        from pytorch_lightning.loggers import WandbLogger

        wandb_project = config.Logging.wandb.project  # whatever you want
        wandb_logger = WandbLogger(
            project=wandb_project, entity=config.Logging.wandb.entity, log_model=False
        )
    else:
        print("Not using Weights & Biases logging, reduced CSV logs written locally.")
        from pytorch_lightning.loggers import CSVLogger

        wandb_logger = CSVLogger(
            save_dir="logs/",
        )

    # Configure Saving Checkpoints
    from pytorch_lightning.callbacks import ModelCheckpoint

    dir_save_checkpoints = os.path.join(
        os.path.normpath("logs/"),
        config.Logging.wandb.project,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_save_checkpoints,
        monitor=config.Schedulers.metric_g,
        mode="min",
        save_last=True,
        save_top_k=2,
    )

    # callback to set up early stopping
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    early_stop_callback = EarlyStopping(
        monitor=config.Schedulers.metric_g,
        min_delta=0.00,
        patience=250,
        verbose=True,
        mode="min",
        check_finite=True,
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
