# SPDX-License-Identifier: MIT
"""Prepare, inspect, and optionally launch a finite Sen2 SR full-state resume."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf


REPOSITORY = "simon-donike/SR-GAN"
TRAINING_CONFIG = "models/sen2-sr/v1/training/config.yaml"
TRAINING_CHECKPOINT = "models/sen2-sr/v1/weights/training.ckpt"
TRAINING_SPLIT = "models/sen2-sr/v1/training/splits/crosssensor.json"
CHECKPOINT_GLOBAL_STEP = 308584


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Local SEN2NAIP/TACO dataset index.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory for this run.")
    parser.add_argument(
        "--max-steps",
        required=True,
        type=int,
        help="Absolute finite Lightning optimizer-step limit; must exceed the checkpoint step.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "gpu"), default="auto")
    parser.add_argument("--revision", help="Optional Hugging Face commit, tag, or branch to resolve.")
    parser.add_argument("--run", action="store_true", help="Launch training after writing and printing the config.")
    return parser.parse_args()


def _resolve_revision(requested_revision: str | None) -> str:
    """Resolve a branch, tag, or implicit default to one immutable Hub SHA."""

    from huggingface_hub import HfApi

    info = HfApi().model_info(REPOSITORY, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Hub did not return a commit SHA for {REPOSITORY}.")
    return info.sha


def _download(filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=REPOSITORY, filename=filename, revision=revision)
    )


def main() -> None:
    args = _parse_args()
    if args.max_steps <= CHECKPOINT_GLOBAL_STEP:
        raise ValueError(
            "--max-steps must exceed the released checkpoint global step "
            f"({CHECKPOINT_GLOBAL_STEP})."
        )
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset index does not exist: {args.dataset}")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    resolved_revision = _resolve_revision(args.revision)
    config_source = _download(TRAINING_CONFIG, resolved_revision)
    checkpoint_source = _download(TRAINING_CHECKPOINT, resolved_revision)
    split_source = _download(TRAINING_SPLIT, resolved_revision)
    args.output_dir.mkdir(parents=True)
    inputs_dir = args.output_dir / "release-inputs"
    inputs_dir.mkdir()
    checkpoint = inputs_dir / "training.ckpt"
    split = inputs_dir / "crosssensor.json"
    shutil.copy2(checkpoint_source, checkpoint)
    shutil.copy2(split_source, split)

    config = OmegaConf.load(config_source)
    config.Data.sen2naip_taco_file = str(args.dataset.resolve())
    config.Data.sen2naip_split_file = str(split.resolve())
    config.Model.continue_training = str(checkpoint.resolve())
    config.Model.load_checkpoint = False
    config.Training.max_steps = args.max_steps
    config.Training.max_epochs = max(int(getattr(config.Training, "max_epochs", 1)), 1)
    config.Training.device = args.device
    config.Logging.output_dir = str(args.output_dir.resolve())
    if "wandb" in config.Logging:
        config.Logging.wandb.enabled = False
    prepared_config = args.output_dir / "config.yaml"
    OmegaConf.save(config, prepared_config)
    provenance = args.output_dir / "release-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "repo_id": REPOSITORY,
                "requested_revision": args.revision,
                "resolved_revision": resolved_revision,
                "training_config": TRAINING_CONFIG,
                "training_checkpoint": TRAINING_CHECKPOINT,
                "training_split": TRAINING_SPLIT,
                "checkpoint_global_step": CHECKPOINT_GLOBAL_STEP,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Prepared full-state resume in {args.output_dir}")
    print(f"  config: {prepared_config}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  split: {split}")
    print(f"  Hub commit: {resolved_revision}")
    print(f"  checkpoint global step: {CHECKPOINT_GLOBAL_STEP}")
    print(f"  max_steps (absolute): {args.max_steps}")
    if not args.run:
        print(
            "Not launching training. Review config.yaml, then run "
            f"'python -m opensr_srgan.train --config {prepared_config}'."
        )
        return

    from opensr_srgan.train import train

    train(prepared_config)


if __name__ == "__main__":
    main()
