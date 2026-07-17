"""Hydra entry point for SRGAN training.

This module composes grouped Hydra configs and forwards the resolved OmegaConf
object to the existing :func:`opensr_srgan.train.train` function.
"""

from __future__ import annotations

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


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
