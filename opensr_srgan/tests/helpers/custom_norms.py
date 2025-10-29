"""Custom normalisation helpers used in the test-suite."""

from __future__ import annotations

import torch


def halve(tensor: torch.Tensor) -> torch.Tensor:
    """Scale the input by 0.5."""

    return tensor * 0.5


def double(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse :func:`halve` by scaling the input by 2."""

    return tensor * 2.0
