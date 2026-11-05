"""CPU contract and correctness tests for the SwinIR generator.

These tests deliberately use a small four-band model.  They exercise the
parts of the implementation that are easy to get subtly wrong: padding and
cropping around window boundaries, shifted-window masking, and preserving the
generator's parameters through checkpoint round-trips.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from opensr_srgan.model.generators.swinir import (  # noqa: E402
    DropPath,
    Mlp,
    ResidualSwinTransformerGroup,
    SwinIRGenerator,
    SwinTransformerBlock,
    WindowAttention,
    calculate_mask,
    window_partition,
    window_reverse,
)


def _generator(**overrides):
    kwargs = dict(
        in_channels=4,
        out_channels=4,
        embed_dim=24,
        depths=(2, 2),
        num_heads=(3, 3),
        window_size=4,
        scale=4,
        use_checkpoint=False,
        drop_path_rate=0.0,
    )
    kwargs.update(overrides)
    return SwinIRGenerator(**kwargs)


@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("hw", [(1, 2), (3, 5), (7, 9)])
def test_forward_returns_exact_scaled_shape_for_odd_and_tiny_inputs(scale, hw):
    model = _generator(scale=scale).eval()
    x = torch.randn(1, 4, *hw)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, hw[0] * scale, hw[1] * scale)
    assert torch.isfinite(y).all()


def test_window_partition_and_reverse_are_exact_for_multiple_images_and_windows():
    x = torch.arange(2 * 8 * 12 * 9, dtype=torch.float32).reshape(2, 8, 12, 9)
    # The public window helpers intentionally operate on canonical BHWC data;
    # padding for odd feature maps belongs to the generator's forward path.
    windows = window_partition(x, window_size=4)
    restored = window_reverse(windows, window_size=4, height=8, width=12)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)


def test_shifted_window_mask_blocks_cyclic_wraparound_but_allows_local_connections():
    # The three-region labels are the canonical Swin construction.  A pair of
    # tokens gets a zero mask only when it belongs to the same shifted window.
    mask = calculate_mask(8, 8, window_size=4, shift_size=2)
    assert mask.shape == (4, 16, 16)
    assert torch.all(mask.diagonal(dim1=-2, dim2=-1) == 0)
    # Every shifted window contains local pairs that remain connected.  The
    # pair at opposite sides of a cyclic wrap is manufactured by torch.roll
    # and must be blocked by the region labels.
    assert torch.all(mask[:, 0, 0] == 0)
    assert torch.any(mask == -100)
    assert mask[1, 0, 1] == 0
    assert mask[1, 0, 2] == -100
    assert torch.all(mask == mask.transpose(-1, -2))


def test_window_utilities_reject_noncanonical_shapes_and_invalid_shifts():
    with pytest.raises(ValueError, match="BHWC"):
        window_partition(torch.zeros(1, 4, 4), window_size=2)
    with pytest.raises(ValueError, match="divisible"):
        window_partition(torch.zeros(1, 3, 4, 2), window_size=2)
    with pytest.raises(ValueError, match="does not match"):
        window_reverse(torch.zeros(2, 3, 3, 2), window_size=2, height=4, width=4)
    with pytest.raises(ValueError, match="expects windows"):
        window_reverse(torch.zeros(2, 2, 2), window_size=2, height=4, width=4)
    with pytest.raises(ValueError, match="divisible"):
        window_reverse(torch.zeros(4, 2, 2, 2), window_size=2, height=3, width=4)
    with pytest.raises(ValueError, match="Number of windows"):
        window_reverse(torch.zeros(3, 2, 2, 2), window_size=2, height=4, width=4)
    with pytest.raises(ValueError, match="shift_size"):
        calculate_mask(4, 4, window_size=2, shift_size=2)
    with pytest.raises(ValueError, match="shift_size"):
        calculate_mask(4, 4, window_size=2, shift_size=False)
    with pytest.raises(ValueError, match="divisible"):
        calculate_mask(3, 4, window_size=2, shift_size=1)
    assert calculate_mask(4, 4, window_size=2, shift_size=0) is None


def test_attention_has_gradient_on_qkv_and_relative_position_bias():
    attention = WindowAttention(dim=24, window_size=4, num_heads=3)
    x = torch.randn(2, 16, 24, requires_grad=True)
    y = attention(x).square().mean()
    y.backward()
    assert attention.qkv.weight.grad is not None
    assert attention.qkv.weight.grad.abs().sum() > 0
    assert attention.relative_position_bias_table.grad is not None
    assert attention.relative_position_bias_table.grad.abs().sum() > 0


def test_attention_accepts_repeated_shift_masks_and_rejects_incompatible_ones():
    attention = WindowAttention(dim=24, window_size=4, num_heads=3).eval()
    x = torch.randn(8, 16, 24)
    mask = calculate_mask(8, 8, window_size=4, shift_size=2)
    with torch.no_grad():
        output = attention(x, mask)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    with pytest.raises(ValueError, match="Attention mask"):
        attention(x, torch.zeros(4, 15, 15))
    with pytest.raises(ValueError, match="not divisible"):
        attention(x[:3], mask)
    with pytest.raises(ValueError, match="incompatible token shape"):
        attention(torch.randn(1, 15, 24))
    with pytest.raises(ValueError, match="expects"):
        attention(torch.randn(1, 4, 4, 24))


def test_drop_path_changes_training_samples_but_is_identity_during_evaluation():
    drop_path = DropPath(drop_prob=0.5)
    x = torch.ones(32, 2, 2, 2)
    drop_path.train()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(3)
        training_output = drop_path(x)
    assert set(training_output.unique().tolist()) == {0.0, 2.0}
    drop_path.eval()
    assert torch.equal(drop_path(x), x)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: DropPath(-0.1), "drop_prob"),
        (lambda: Mlp(dim=1, mlp_ratio=0.1, drop=0.0), "at least one"),
        (lambda: WindowAttention(dim=10, window_size=2, num_heads=3), "divisible"),
        (
            lambda: ResidualSwinTransformerGroup(
                4,
                0,
                1,
                2,
                mlp_ratio=2.0,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=[],
                use_checkpoint=False,
            ),
            "at least one",
        ),
        (
            lambda: ResidualSwinTransformerGroup(
                4,
                2,
                1,
                2,
                mlp_ratio=2.0,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=[0.0],
                use_checkpoint=False,
            ),
            "one value",
        ),
    ],
)
def test_internal_swin_components_validate_their_contracts(constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor()


def test_transformer_block_rejects_feature_width_mismatch():
    block = SwinTransformerBlock(
        4,
        1,
        2,
        0,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )
    with pytest.raises(ValueError, match="Expected 4"):
        block(torch.zeros(1, 3, 2, 2))


def test_generator_gradients_reach_attention_and_reconstruction_head():
    model = _generator(scale=2)
    x = torch.randn(1, 4, 5, 7)
    model(x).mean().backward()
    named = dict(model.named_parameters())
    qkv = [p for n, p in named.items() if n.endswith("qkv.weight")]
    bias = [
        p
        for n, p in named.items()
        if n.endswith("relative_position_bias_table")
    ]
    head = [p for n, p in named.items() if n.startswith("upsample.0")]
    assert qkv and any(p.grad is not None and p.grad.abs().sum() > 0 for p in qkv)
    assert bias and any(p.grad is not None and p.grad.abs().sum() > 0 for p in bias)
    assert head and any(p.grad is not None and p.grad.abs().sum() > 0 for p in head)


def test_checkpoint_roundtrip_is_deterministic_at_different_input_sizes(tmp_path):
    torch.manual_seed(7)
    model = _generator(scale=3).eval()
    x = torch.randn(1, 4, 5, 6)
    with torch.no_grad():
        expected = model(x)
    path = tmp_path / "swinir.pt"
    torch.save(model.state_dict(), path)
    restored = _generator(scale=3).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        actual = restored(x)
        other = restored(torch.randn(1, 4, 7, 3))
    assert torch.equal(actual, expected)
    assert other.shape == (1, 4, 21, 9)


def test_checkpointing_path_matches_regular_forward_and_backward():
    torch.manual_seed(11)
    regular = _generator(scale=2, use_checkpoint=False)
    checkpointed = _generator(scale=2, use_checkpoint=True)
    checkpointed.load_state_dict(regular.state_dict())
    x1 = torch.randn(1, 4, 5, 6, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    y1 = regular(x1)
    y2 = checkpointed(x2)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
    y1.square().mean().backward()
    y2.square().mean().backward()
    assert torch.isfinite(x2.grad).all()
    for p1, p2 in zip(regular.parameters(), checkpointed.parameters()):
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=2e-5, rtol=2e-4)


def test_checkpointing_path_propagates_gradients_without_input_grad():
    model = _generator(scale=2, use_checkpoint=True).train()
    x = torch.randn(1, 4, 5, 6)
    model(x).square().mean().backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def test_bfloat16_autocast_forward_backward_is_finite():
    model = _generator(scale=2).train()
    x = torch.randn(1, 4, 5, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_generator_rejects_invalid_runtime_inputs_before_attention():
    model = _generator(scale=2).eval()
    with pytest.raises(ValueError, match="BCHW"):
        model(torch.randn(4, 8, 8))
    with pytest.raises(ValueError, match="input bands"):
        model(torch.randn(1, 3, 8, 8))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(1, 4, 8, 8, dtype=torch.int64))


def test_generator_supports_distinct_output_band_count_and_reflection_padding():
    model = _generator(out_channels=2, scale=2).eval()
    x = torch.randn(1, 4, 7, 9)
    padded, original_height, original_width = model._pad_input(x)
    assert padded.shape[-2:] == (8, 12)
    assert (original_height, original_width) == (7, 9)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 2, 14, 18)


def test_generator_keeps_aligned_inputs_unpadded():
    model = _generator(scale=2)
    x = torch.randn(1, 4, 8, 8)
    padded, height, width = model._pad_input(x)
    assert padded is x
    assert (height, width) == (8, 8)


def test_generator_defaults_output_bands_to_input_bands():
    model = SwinIRGenerator(
        in_channels=4,
        embed_dim=12,
        depths=(1,),
        num_heads=(3,),
        window_size=4,
        scale=2,
        drop_path_rate=0.0,
    )
    assert model.out_channels == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 10, "num_heads": (3, 3)},
        {"depths": (2,), "num_heads": (3, 3)},
        {"depths": (2, 2), "num_heads": (3,)},
        {"window_size": 0},
        {"window_size": 1},
        {"scale": 0},
        {"scale": 5},
        {"depths": "2"},
        {"num_heads": "3"},
        {"mlp_ratio": 0},
        {"drop_rate": 1.0},
        {"qkv_bias": 1},
        {"use_checkpoint": 1},
    ],
)
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(
        (ValueError, TypeError),
        match="(divisible|divide|length|window|scale|positive|greater|lie|bool)",
    ):
        _generator(**kwargs)
