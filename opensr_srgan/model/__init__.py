"""Model package exposing the SRGAN Lightning module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["SRGAN_model"]

if TYPE_CHECKING:  # pragma: no cover
    from .SRGAN import SRGAN_model as SRGANModel


def __getattr__(name: str) -> Any:  # pragma: no cover - import proxy
    if name == "SRGAN_model":
        from .SRGAN import SRGAN_model as _cls

        globals()[name] = _cls
        return _cls
    raise AttributeError(f"module 'opensr_srgan.model' has no attribute '{name}'")
