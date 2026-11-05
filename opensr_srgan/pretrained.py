"""Build minimal, reproducible inference checkpoints from trusted training runs.

The release format deliberately contains only EMA generator tensors.  It is a
weights-only artifact for :func:`opensr_srgan.load_inference_model`, rather
than a resumable Lightning checkpoint.  In particular, it never carries a
discriminator, VGG loss network, optimizer, scheduler, or CUDA EMA device.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch

from opensr_srgan.model.SRGAN import SRGAN_model
from opensr_srgan.utils.checkpoint_loading import load_checkpoint

__all__ = [
    "export_ema_generator_checkpoint",
    "export_inference_checkpoint",
    "validate_ema_generator_state",
]

_FORMAT_VERSION = "opensr-srgan-ema-generator-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Return a stable digest for an ordered, CPU generator state dictionary."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _discover_config(
    source_checkpoint: Path, destination: Path, config_path: Optional[Union[str, Path]]
) -> Path:
    if config_path is not None:
        candidate = Path(config_path)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Inference config '{candidate}' could not be located."
            )
        return candidate

    # The two-argument CLI is useful for a conventional release directory
    # containing config.yaml and model_ema.ckpt, while --config stays explicit
    # for arbitrary training/checkpoint layouts.
    candidates = (
        destination.parent / "config.yaml",
        source_checkpoint.parent / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        "A generator config is required to validate the EMA export. Pass "
        "--config CONFIG, or place config.yaml beside SOURCE or DEST."
    )


def _ema_generator_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    ema_state = checkpoint.get("ema_state")
    if not isinstance(ema_state, Mapping):
        raise ValueError(
            "Source checkpoint has no ema_state; refusing to export live weights."
        )

    shadow_params = ema_state.get("shadow_params")
    shadow_buffers = ema_state.get("shadow_buffers")
    if not isinstance(shadow_params, Mapping) or not isinstance(
        shadow_buffers, Mapping
    ):
        raise ValueError(
            "Source checkpoint ema_state must contain shadow_params and shadow_buffers."
        )

    generator_state: dict[str, torch.Tensor] = {}
    for collection_name, values in (
        ("shadow_params", shadow_params),
        ("shadow_buffers", shadow_buffers),
    ):
        for name, tensor in values.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"EMA {collection_name} contains a non-tensor entry: {name!r}."
                )
            key = f"generator.{name}"
            if key in generator_state:
                raise ValueError(f"EMA state duplicates generator tensor '{key}'.")
            generator_state[key] = tensor.detach().cpu().clone().contiguous()
    return generator_state


def validate_ema_generator_state(
    state_dict: Mapping[str, torch.Tensor], model: SRGAN_model
) -> None:
    """Ensure an export is a complete, finite generator state for ``model``.

    This strict validation turns architecture/configuration mismatches into a
    release-time error rather than a silently degraded inference model.
    """

    expected = {
        f"generator.{name}": tensor
        for name, tensor in model.generator.state_dict().items()
    }
    actual_keys = set(state_dict)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(
            "EMA generator keys do not match the inference generator ("
            + "; ".join(details)
            + ")."
        )

    for name, expected_tensor in expected.items():
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"EMA generator entry '{name}' is not a tensor.")
        if tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"EMA generator tensor '{name}' has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_tensor.shape)}."
            )
        if tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"EMA generator tensor '{name}' has dtype {tensor.dtype}, "
                f"expected {expected_tensor.dtype}."
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                f"EMA generator tensor '{name}' contains non-finite values."
            )


def export_ema_generator_checkpoint(
    source_checkpoint: Union[str, Path],
    destination: Union[str, Path],
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Export EMA generator weights from a trusted Lightning checkpoint.

    ``source_checkpoint`` must be a full checkpoint with an ``ema_state`` as
    written by :class:`SRGAN_model`.  The destination is a small inference-only
    checkpoint with a strict ``generator.*`` state dictionary and deterministic
    provenance.  The function intentionally does not accept raw/live model
    checkpoints, so a released EMA artifact cannot be produced accidentally
    from non-EMA weights.
    """

    source = Path(source_checkpoint)
    target = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(
            f"Source checkpoint '{source}' could not be located."
        )
    if source.resolve() == target.resolve():
        raise ValueError(
            "Destination must differ from the trusted source checkpoint; refusing to "
            "replace a resumable checkpoint with an inference artifact."
        )
    config = _discover_config(source, target, config_path)

    checkpoint = load_checkpoint(source, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise ValueError("Source must be a full Lightning checkpoint with a state_dict.")

    # Construct in eval mode: validating must not build the discriminator or
    # loss networks, and EMA's serialized CUDA device is never restored.
    model = SRGAN_model(config=config, mode="eval")
    state_dict = _ema_generator_state(checkpoint)
    validate_ema_generator_state(state_dict, model)

    ema_state = checkpoint["ema_state"]
    metadata = {
        "format": _FORMAT_VERSION,
        "format_version": _FORMAT_VERSION,
        # Keep the short, tool-friendly name and the explicit legacy-free
        # spelling for downstream manifests.
        "source_sha256": _sha256_file(source),
        "source_global_step": checkpoint.get("global_step"),
        "source_epoch": checkpoint.get("epoch"),
        "ema_decay": ema_state.get("decay"),
        "ema_num_updates": ema_state.get("num_updates"),
        "config_sha256": _sha256_file(config),
        "generator_state_sha256": _sha256_state_dict(state_dict),
    }
    artifact = {"state_dict": state_dict, "metadata": metadata}

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, target)
    return metadata


# A concise public name for deployment scripts.
export_inference_checkpoint = export_ema_generator_checkpoint


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export EMA-only OpenSR SRGAN weights from a trusted Lightning checkpoint."
        )
    )
    parser.add_argument("source", type=Path, help="trusted full Lightning checkpoint")
    parser.add_argument("destination", type=Path, help="EMA-only inference checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        help="inference YAML used to validate generator keys and shapes",
    )
    args = parser.parse_args(argv)
    metadata = export_ema_generator_checkpoint(
        args.source, args.destination, config_path=args.config
    )
    print(f"Exported {args.destination} ({metadata['generator_state_sha256']})")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(_main())
