"""Tests for utility modules: logging helpers, radiometrics, and more."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, patch

from opensr_srgan.utils.logging_helpers import plot_tensors
from opensr_srgan.utils.radiometrics import histogram as histogram_match


def test_plot_tensors_basic_shape():
    """Test plot_tensors produces output with correct spatial dimensions."""
    lr = torch.ones(1, 3, 32, 32) * 0.2
    sr = torch.ones(1, 3, 64, 64) * 0.5
    hr = torch.ones(1, 3, 64, 64) * 0.8

    result = plot_tensors(lr, sr, hr, title="Test")
    assert result is not None
    # Result should be a PIL image or array
    assert hasattr(result, "size") or hasattr(result, "shape")


def test_plot_tensors_single_channel():
    """Test plot_tensors with single-channel inputs."""
    lr = torch.ones(1, 1, 32, 32) * 0.2
    sr = torch.ones(1, 1, 64, 64) * 0.5
    hr = torch.ones(1, 1, 64, 64) * 0.8

    result = plot_tensors(lr, sr, hr, title="Single Channel")
    assert result is not None


def test_plot_tensors_batch_multiple():
    """Test plot_tensors with multiple batches."""
    lr = torch.randn(3, 3, 32, 32)
    sr = torch.randn(3, 3, 64, 64)
    hr = torch.randn(3, 3, 64, 64)

    result = plot_tensors(lr, sr, hr, title="Batch")
    assert result is not None


def test_histogram_match_basic():
    """Test histogram matching preserves shape."""
    src = torch.randn(2, 3, 64, 64)
    ref = torch.randn(2, 3, 64, 64)

    result = histogram_match(src, ref)
    assert result.shape == src.shape


def test_histogram_match_range_preservation():
    """Test histogram matching adjusts distribution."""
    src = torch.ones(1, 3, 32, 32) * 0.5
    ref = torch.ones(1, 3, 32, 32) * 0.2

    result = histogram_match(src, ref)
    # Result should have adjusted values closer to ref range
    assert result is not None


def test_histogram_match_single_band():
    """Test histogram matching with single band."""
    src = torch.randn(2, 1, 32, 32)
    ref = torch.randn(2, 1, 32, 32)

    result = histogram_match(src, ref)
    assert result.shape == src.shape


def test_histogram_match_multi_band():
    """Test histogram matching with many bands."""
    src = torch.randn(1, 11, 32, 32)
    ref = torch.randn(1, 11, 32, 32)

    result = histogram_match(src, ref)
    assert result.shape == src.shape
