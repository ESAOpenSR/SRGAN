from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path


def get_version() -> str:
    try:
        return importlib_metadata.version("opensr-srgan")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def bundled_slurm_entrypoint() -> Path:
    return Path(__file__).resolve().parent / "slurm" / "slurm_task_entrypoint.sh"


__version__ = get_version()
