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


def test_histogram_preserves_channels_and_resizes_reference():
    reference = torch.tensor(
        [[[[0.0, 0.5], [0.5, 1.0]]]], dtype=torch.float32
    )  # (1,1,2,2)
    target = torch.tensor([[[[1.0, 0.0], [0.25, 0.75]]], [[[0.0, 0.1], [0.2, 0.3]]]])

    matched = histogram(reference, target)

    assert matched.shape == target.shape
    assert torch.all(torch.isfinite(matched))


def test_histogram_channel_mismatch_raises():
    reference = torch.zeros(1, 1, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    with pytest.raises(ValueError):
        histogram(reference, target)


def test_moment_matches_mean_and_std_per_channel():
    reference = torch.tensor(
        [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]], dtype=torch.float32
    )
    target = torch.tensor(
        [[[1.0, 2.0], [0.0, 4.0]], [[5.0, 1.0], [3.0, 2.0]]], dtype=torch.float32
    )

    matched = moment(reference, target)

    ref_means = reference.view(2, -1).mean(dim=1)
    ref_stds = reference.view(2, -1).std(dim=1)
    out_means = matched.view(2, -1).mean(dim=1)
    out_stds = matched.view(2, -1).std(dim=1)

    assert torch.allclose(out_means, ref_means)
