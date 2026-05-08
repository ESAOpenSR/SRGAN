"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch


def load_checkpoint(path: str | Path, map_location: Any = None) -> Any:
    """Load a PyTorch checkpoint with safer defaults when supported.

    Newer PyTorch releases support ``weights_only=True`` to avoid unpickling
    arbitrary Python objects from checkpoints. Older releases do not expose the
    argument, and some legacy checkpoints may still require the old behavior, so
    this helper keeps those paths compatible while making the safer mode the
    first attempt.
    """

    load_kwargs = {"map_location": map_location}
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)
    except Exception as exc:
        warnings.warn(
            "Falling back to torch.load(..., weights_only=False) for checkpoint "
            f"'{path}'. Only load checkpoints from trusted sources. Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, **load_kwargs)
