"""Utility helpers to instantiate pretrained SRGAN models."""

from __future__ import annotations

import dataclasses
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

import torch
from pytorch_lightning import LightningModule

from opensr_srgan.model.SRGAN import SRGAN_model

__all__ = ["load_from_config", "load_inference_model"]


@dataclasses.dataclass(frozen=True)
class _Preset:
    """Metadata describing an inference-ready configuration."""

    repo_id: str
    config_filename: str
    checkpoint_filename: str


_PRESETS = {
    "RGB-NIR": _Preset(
        repo_id="simon-donike/SR-GAN",
        config_filename="config_RGB-NIR.yaml",
        checkpoint_filename="RGB-NIR_4band_inference.ckpt",
    ),
    "SWIR": _Preset(
        repo_id="simon-donike/SR-GAN",
        config_filename="config_SWIR.yaml",
        checkpoint_filename="SWIR_6band_inference.ckpt",
    ),
}


@contextmanager
def _maybe_download(checkpoint_uri: Union[str, Path]) -> Iterator[Path]:
    """Resolve a checkpoint URI to a local file path."""

    uri_str = str(checkpoint_uri)
    candidate = Path(checkpoint_uri)
    if candidate.is_file():
        yield candidate
        return

    if uri_str.startswith(("http://", "https://")):
        suffix = candidate.suffix if candidate.suffix else ".ckpt"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            torch.hub.download_url_to_file(uri_str, tmp.name, progress=False)
            tmp.flush()
            yield Path(tmp.name)
        return

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint_uri}' does not exist. Provide a local path or HTTP(S) URL."
    )


def load_from_config(
    config_path: Union[str, Path],
    checkpoint_uri: Optional[Union[str, Path]] = None,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
    mode: str = "train",
) -> LightningModule:
    """Instantiate ``SRGAN_model`` from a YAML config and optional checkpoint.

    Parameters
    ----------
    config_path:
        Filesystem path to the YAML configuration that describes the generator
        and discriminator architecture. This should match the configuration the
        checkpoint was trained with.
    checkpoint_uri:
        Optional path or HTTP(S) URL pointing to a Lightning checkpoint. When
        omitted, the factory returns an untrained model initialised from the
        supplied config.
    map_location:
        Optional argument forwarded to :func:`torch.load` during checkpoint
        deserialisation.
    mode:
        Mode in which to instantiate the model. Either "train" or "eval".
    """

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file '{config_path}' could not be located.")

    model = SRGAN_model(config=config_path, mode=mode)

    if checkpoint_uri is not None:
        with _maybe_download(checkpoint_uri) as resolved_path:
            checkpoint = torch.load(str(resolved_path), map_location=map_location)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

        if model.ema is not None and "ema_state" in checkpoint:
            model.ema.load_state_dict(checkpoint["ema_state"])

    model.eval()
    return model


def load_inference_model(
    preset: str,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> LightningModule:
    """Instantiate an off-the-shelf pretrained SRGAN.

    The function downloads a known-good configuration + checkpoint pair from
    the Hugging Face Hub (unless it is already cached) and restores the
    packaged Lightning module.
    """

    key = preset.strip().replace("_", "-").upper()
    try:
        preset_meta = _PRESETS[key]
    except KeyError as err:
        valid = ", ".join(sorted(_PRESETS))
        raise ValueError(
            f"Unknown preset '{preset}'. Available options: {valid}."
        ) from err

    try:  # pragma: no cover - import guard only used at runtime
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "huggingface_hub is required for load_inference_model. "
            "Install the project extras or run 'pip install huggingface-hub'."
        ) from exc

    config_path = hf_hub_download(
        repo_id=preset_meta.repo_id,
        filename=preset_meta.config_filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    checkpoint_path = hf_hub_download(
        repo_id=preset_meta.repo_id,
        filename=preset_meta.checkpoint_filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )

    return load_from_config(
        config_path,
        checkpoint_path,
        map_location=map_location,
        mode="eval",
    )


"""
if __name__ == "__main__":
    # simple test
    # Create Model
    model = load_from_config("opensr_srgan/configs/config_playgound.yaml")
    #model = load_inference_model("RGB-NIR")
    
    # Print Model Summary
    from opensr_srgan.utils.model_descriptions import print_model_summary
    print_model_summary(model)
    
    # Simple test for funcionality
    import torch
    lr = torch.randn(1, 4, 64, 64)
    sr = model.forward(lr)
    print(lr.shape, "->", sr.shape)
"""