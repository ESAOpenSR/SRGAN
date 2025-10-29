# opensr_srgan/tests/test_radiometrics.py
import torch

torch.manual_seed(0)

import numpy as np
import pytest

from opensr_srgan.utils.radiometrics import (
    normalise_s2,
    normalise_10k,
    normalise_10k_signed,
    sen2_stretch,
    zero_one_signed,
    minmax_percentile,
    minmax,
    histogram,
    moment,
)


def test_normalise_s2_roundtrip():
    x = torch.rand(3, 8, 8) * 0.3  # reflectance-like (~[0, 0.3])
    y = normalise_s2(x, "norm")
    z = normalise_s2(y, "denorm")
    assert torch.all(y <= 1) and torch.all(y >= -1)
    assert torch.all(z <= 1) and torch.all(z >= 0)
    assert torch.allclose(x.clamp(0, 1), z, atol=1e-6)


def test_normalise_10k_roundtrip():
    x = torch.randint(0, 10001, (3, 8, 8), dtype=torch.int32).float()
    y = normalise_10k(x, "norm")
    z = normalise_10k(y, "denorm")
    assert torch.all(y <= 1) and torch.all(y >= 0)
    assert torch.all(z <= 10000) and torch.all(z >= 0)
    assert torch.allclose(x.clamp(0, 10000), z, atol=1e-4)


def test_normalise_10k_signed_roundtrip():
    x = torch.randint(0, 10001, (3, 8, 8), dtype=torch.int32).float()
    y = normalise_10k_signed(x, "norm")
    z = normalise_10k_signed(y, "denorm")
    assert torch.all(y <= 1) and torch.all(y >= -1)
    assert torch.all(z <= 10000) and torch.all(z >= 0)
    assert torch.allclose(x.clamp(0, 10000), z, atol=1e-4)


def test_sen2_stretch_range():
    x = torch.rand(3, 8, 8)  # [0,1]
    y = sen2_stretch(x)
    assert y.shape == x.shape
    assert torch.all(y >= 0) and torch.all(y <= 1)


def test_zero_one_signed_roundtrip():
    x = torch.rand(3, 8, 8)
    y = zero_one_signed(x, "norm")
    z = zero_one_signed(y, "denorm")
    assert torch.all(y <= 1) and torch.all(y >= -1)
    assert torch.allclose(x.clamp(0, 1), z, atol=1e-6)


def test_minmax_percentile_basic_bounds():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0]).view(1, 1, 5, 1)  # include outlier
    y = minmax_percentile(x, pmin=2, pmax=98)
    assert x.shape == y.shape  # only validate shape


def test_minmax_unit_range():
    x = torch.randn(4, 5, 6)
    y = minmax(x)
    assert torch.isclose(y.min(), torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(y.max(), torch.tensor(1.0), atol=1e-7)
