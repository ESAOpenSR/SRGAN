"""Top-level package for OpenSR SRGAN deployment helpers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - import shim for Python <3.8
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ModuleNotFoundError:  # pragma: no cover - fallback for very old Python
    from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore

__all__ = [
    "SRGAN_model",
    "train",
    "load_from_config",
    "load_inference_model",
    "export_ema_generator_checkpoint",
    "export_inference_checkpoint",
    "__version__",
]

try:  # pragma: no cover - depends on installation metadata
    __version__ = _pkg_version("opensr-srgan")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.0.0"


def _silence_known_lightning_deprecations() -> None:
    """Suppress noisy third-party deprecations we cannot patch locally."""
    warnings.filterwarnings(
        "ignore",
        message=r".*isinstance\(treespec,\s*LeafSpec\)\s*is deprecated.*",
        category=DeprecationWarning,
        module=r"pytorch_lightning\.utilities\._pytree",
    )


_silence_known_lightning_deprecations()

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from .model.SRGAN import SRGAN_model as SRGANModel
    from .train import train as _train


def __getattr__(name: str) -> Any:  # pragma: no cover - simple import proxy
    if name == "SRGAN_model":
        from .model.SRGAN import SRGAN_model as _cls

        globals()[name] = _cls
        return _cls
    if name == "train":
        from .train import train as _train_fn

        globals()[name] = _train_fn
        return _train_fn
    if name in {"load_from_config", "load_inference_model"}:
        from . import _factory

        attr = getattr(_factory, name)
        globals()[name] = attr
        return attr
    if name in {"export_ema_generator_checkpoint", "export_inference_checkpoint"}:
        from . import pretrained

        attr = getattr(pretrained, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'opensr_srgan' has no attribute '{name}'")
