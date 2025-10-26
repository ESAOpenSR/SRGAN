# Package Imports
import datetime
import os
from pathlib import Path
from multiprocessing import freeze_support

import torch
import wandb
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer


def train(config):

    #############################################################################################################
    """LOAD CONFIG"""
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
    #############################################################################################################

    # Get devices
    cuda_devices = config.Training.gpus
    cuda_strategy = "ddp" if len(cuda_devices) > 1 else None

    #############################################################################################################
    " LOAD MODEL "
    #############################################################################################################
    # load pretrained or instanciate new
    from opensr_srgan.model.SRGAN import SRGAN_model

    if config.Model.load_checkpoint != False:
        model = SRGAN_model.load_from_checkpoint(
            config.Model.load_checkpoint, strict=False
        )
    else:
        model = SRGAN_model(config=config)
    if config.Model.continue_training != False:
        resume_from_checkpoint_variable = config.Model.continue_training
    else:
        resume_from_checkpoint_variable = None

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
    """ make it robust for both PL<2.0 and PL>=2.0 """
    #############################################################################################################
    from opensr_srgan.utils.build_trainer_kwargs import build_lightning_kwargs

    trainer_kwargs, fit_kwargs = (
        build_lightning_kwargs(  # get kwargs depending on PL version
            config=config,
            logger=wandb_logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            resume_ckpt=resume_from_checkpoint_variable,
        )
    )

    # Start training
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=pl_datamodule, **fit_kwargs)
    wandb.finish()


# Run training if called from command line
if __name__ == "__main__":
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
