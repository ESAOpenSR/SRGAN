"""Utility helpers to instantiate pretrained SRGAN models.

This module provides convenience functions to:
  - Load ``SRGAN_model`` instances from local YAML configuration files and optional checkpoints.
  - Download and instantiate predefined pretrained models from the Hugging Face Hub
    (e.g., RGB-NIR and SWIR variants).
  - Transparently handle local or remote checkpoints, temporary file storage, and EMA restoration.

Typical usage
-------------
>>> from opensr_srgan.model.loading import load_inference_model
>>> model = load_inference_model("RGB-NIR")
>>> model.eval()
>>> sr = model(torch.randn(1, 4, 64, 64))
"""
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
    """Metadata describing a pretrained SRGAN configuration.

    Attributes
    ----------
    repo_id : str
        Repository ID on the Hugging Face Hub containing the config and checkpoint.
    config_filename : str
        Name of the YAML configuration file inside the repository.
    checkpoint_filename : str
        Name of the model checkpoint file inside the repository.
    """
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
    """Resolve a checkpoint URI to a local file path.

    This helper transparently supports:
      - Existing local file paths.
      - Remote HTTP(S) URLs (downloaded temporarily into a secure temp file).

    Parameters
    ----------
    checkpoint_uri : str or Path
        Path or URL of the checkpoint to resolve.

    Yields
    ------
    Path
        A valid local path to a checkpoint file, guaranteed to exist within the context.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist or the URI cannot be resolved.
    """
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
    """Instantiate an :class:`SRGAN_model` from a configuration and optional checkpoint.

    Parameters
    ----------
    config_path : str or Path
        Filesystem path to the YAML configuration describing generator/discriminator
        architecture and training parameters.
    checkpoint_uri : str or Path, optional
        Path or URL to a pretrained model checkpoint. If omitted, returns an untrained model.
    map_location : str or torch.device, optional
        Device mapping passed to :func:`torch.load` when deserializing the checkpoint.
    mode : {"train", "eval"}, default="train"
        Desired mode in which to initialize the model.

    Returns
    -------
    LightningModule
        A fully initialized SRGAN Lightning module, optionally loaded with pretrained weights.

    Raises
    ------
    FileNotFoundError
        If the provided configuration file does not exist.
    RuntimeError
        If checkpoint deserialization or state restoration fails.

    Notes
    -----
    - If the checkpoint contains an exponential moving average (EMA) state, it is restored as well.
    - The model is returned in evaluation mode regardless of `mode` to prevent accidental training
      until explicitly switched with ``model.train()``.
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
    """Load a pretrained SRGAN model from the Hugging Face Hub.

    Downloads both the configuration and checkpoint associated with the requested
    preset, instantiates the model, restores weights, and returns it ready for inference.

    Parameters
    ----------
    preset : {"RGB-NIR", "SWIR"}
        Name of the pretrained model variant to load.
    cache_dir : str or Path, optional
        Directory to cache the downloaded files. Uses the default HF cache if omitted.
    map_location : str or torch.device, optional
        Device mapping for checkpoint deserialization.

    Returns
    -------
    LightningModule
        Pretrained SRGAN model ready for inference.

    Raises
    ------
    ValueError
        If an unknown preset name is provided.
    ImportError
        If ``huggingface_hub`` is not installed.
    FileNotFoundError
        If download or local resolution fails.

    Examples
    --------
    >>> model = load_inference_model("RGB-NIR")
    >>> model.eval()
    >>> x = torch.randn(1, 4, 64, 64)
    >>> y = model(x)
    >>> print(y.shape)
    torch.Size([1, 4, 256, 256])
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
# -------------------------------------------------------------------------
# Example test block (optional)
# -------------------------------------------------------------------------
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