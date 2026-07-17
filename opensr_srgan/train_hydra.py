"""Hydra entry point for SRGAN training.

This module composes grouped Hydra configs and forwards the resolved OmegaConf
object to the existing :func:`opensr_srgan.train.train` function.
"""

from __future__ import annotations

import argparse
import sys

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


def _patch_python314_argparse_help_validation() -> None:
    """Allow Hydra lazy completion help objects on Python 3.14+."""

    if sys.version_info < (3, 14):
        return

    original_check_help = argparse.ArgumentParser._check_help
    if getattr(original_check_help, "_srgan_hydra_compat", False):
        return

    def _check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            return
        return original_check_help(self, action)

    _check_help._srgan_hydra_compat = True
    argparse.ArgumentParser._check_help = _check_help


_patch_python314_argparse_help_validation()


def _has_value(value) -> bool:
    return value not in (False, None, "")


def _attach_hydra_output_dir(cfg: DictConfig) -> DictConfig:
    """Populate ``Logging.output_dir`` from Hydra runtime when it is unset."""

    output_dir = OmegaConf.select(cfg, "Logging.output_dir", default=None)
    if _has_value(output_dir):
        return cfg

    try:
        hydra_output_dir = HydraConfig.get().runtime.output_dir
    except ValueError:
        return cfg

    with open_dict(cfg):
        if "Logging" not in cfg:
            cfg.Logging = {}
        cfg.Logging.output_dir = hydra_output_dir
    return cfg


@hydra.main(version_base="1.3", config_path="configs/hydra", config_name="train")
def main(cfg: DictConfig) -> None:
    """Compose Hydra config and launch the existing SRGAN trainer."""

    torch.set_float32_matmul_precision("medium")
    cfg = _attach_hydra_output_dir(cfg)

    from opensr_srgan.train import train

    train(cfg)


if __name__ == "__main__":
    main()
