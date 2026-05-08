from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from opensr_srgan.model.model_blocks import (
    ConvolutionalBlock,
    DenseBlock5,
    LKA,
    LKAResBlock,
    RCAB,
    RRDB,
    ResidualBlock,
    ResidualBlockNoBN,
    SubPixelConvolutionalBlock,
    make_upsampler,
)
from opensr_srgan.model.model_blocks import _icnr_


def test_convolutional_block_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation must be one of"):
        ConvolutionalBlock(16, 16, 3, activation="relu")


def test_convolutional_block_tanh_path_keeps_spatial_shape() -> None:
    block = ConvolutionalBlock(16, 16, 3, batch_norm=True, activation="tanh")
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == x.shape


def test_subpixel_block_upsamples_by_scaling_factor() -> None:
    block = SubPixelConvolutionalBlock(n_channels=16, scaling_factor=2)
    x = torch.randn(1, 16, 8, 8)
    y = block(x)
    assert y.shape == (1, 16, 16, 16)


def test_residual_and_attention_blocks_preserve_shape() -> None:
    x = torch.randn(1, 16, 8, 8)

    assert ResidualBlock(n_channels=16)(x).shape == x.shape
    assert ResidualBlockNoBN(n_channels=16)(x).shape == x.shape
    assert RCAB(n_channels=16)(x).shape == x.shape
    assert DenseBlock5(n_features=16, growth_channels=8)(x).shape == x.shape
    assert RRDB(n_features=16, growth_channels=8)(x).shape == x.shape
    assert LKA(n_channels=16)(x).shape == x.shape
    assert LKAResBlock(n_channels=16)(x).shape == x.shape


def test_icnr_requires_divisible_output_channels() -> None:
    weight = torch.empty(10, 4, 3, 3)
    with pytest.raises(ValueError, match=r"divisible by scale\*\*2"):
        _icnr_(weight, scale=2)


def test_make_upsampler_scale_4_with_icnr_produces_expected_shape() -> None:
    upsampler = make_upsampler(16, scale=4, use_icnr=True)
    x = torch.randn(1, 16, 8, 8)
    y = upsampler(x)
    assert y.shape == (1, 16, 32, 32)


def test_make_upsampler_rejects_non_power_of_two_scale() -> None:
    with pytest.raises(ValueError, match="power of two"):
        make_upsampler(16, scale=3)
