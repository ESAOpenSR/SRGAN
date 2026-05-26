# opensr_srgan/tests/test_ema.py
import torch
from torch import nn
import pytest

from opensr_srgan.model.model_blocks.EMA import ExponentialMovingAverage


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(1))

    def forward(self, x):
        return self.lin(x) * self.scale


def test_register_and_state_dict():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    # all named parameters registered
    assert set(ema.shadow_params.keys()) == {"lin.weight", "lin.bias"}
    assert set(ema.shadow_buffers.keys()) == {"scale"}
    state = ema.state_dict()
    assert "decay" in state and "shadow_params" in state


def test_update_moves_toward_model():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.5, use_num_updates=False)
    old_shadow = ema.shadow_params["lin.weight"].clone()
    # modify model weights
    with torch.no_grad():
        model.lin.weight.add_(1.0)
    ema.update(model)
    new_shadow = ema.shadow_params["lin.weight"]
    # shadow should have increased but not fully matched
    diff = (new_shadow - old_shadow).abs().mean()
    assert diff > 0 and diff < 1.0


def test_apply_and_restore_swap():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    # change shadow so we can detect the swap
    with torch.no_grad():
        ema.shadow_params["lin.weight"].add_(10.0)

    original_weight = model.lin.weight.clone()
    ema.apply_to(model)
    assert torch.allclose(model.lin.weight, ema.shadow_params["lin.weight"])
    ema.restore(model)
    assert torch.allclose(model.lin.weight, original_weight)


def test_average_parameters_context_manager():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    with ema.average_parameters(model):
        # inside context, weights are swapped
        assert torch.allclose(model.lin.weight, ema.shadow_params["lin.weight"])
    # restored after exit
    assert not ema.collected_params


def test_to_device_and_load_state_dict():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.to("cpu")  # no-op but covers code path

    state = ema.state_dict()
    ema2 = ExponentialMovingAverage(model, decay=0.1)
    ema2.load_state_dict(state)

    for k in ema.shadow_params:
        assert torch.allclose(ema.shadow_params[k], ema2.shadow_params[k])
    assert ema2.decay == pytest.approx(ema.decay)
    assert ema2.num_updates == ema.num_updates


def test_invalid_decay_raises():
    model = TinyNet()
    with pytest.raises(ValueError):
        ExponentialMovingAverage(model, decay=1.5)


def test_update_with_num_updates_uses_warmup_decay_and_updates_buffer():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9, use_num_updates=True)
    old_shadow = ema.shadow_params["lin.weight"].clone()

    with torch.no_grad():
        model.lin.weight.add_(1.0)
        model.scale.fill_(3.0)

    ema.update(model)

    expected_decay = min(0.9, (1 + 1) / (10 + 1))
    expected = old_shadow.lerp(model.lin.weight.detach(), 1.0 - expected_decay)
    assert ema.num_updates == 1
    assert torch.allclose(ema.shadow_params["lin.weight"], expected)
    assert torch.allclose(ema.shadow_buffers["scale"], torch.tensor([3.0]))


def test_register_and_update_skip_frozen_parameters():
    model = TinyNet()
    model.lin.bias.requires_grad_(False)

    ema = ExponentialMovingAverage(model, decay=0.5, use_num_updates=False)
    assert set(ema.shadow_params) == {"lin.weight"}

    with torch.no_grad():
        model.lin.bias.add_(10.0)
    ema.update(model)

    assert "lin.bias" not in ema.shadow_params


def test_update_registers_new_trainable_parameters_and_buffers():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.5, use_num_updates=False)

    model.extra = nn.Parameter(torch.full((1,), 4.0))
    model.register_buffer("offset", torch.full((1,), 2.0))

    ema.update(model)

    assert torch.allclose(ema.shadow_params["extra"], torch.tensor([4.0]))
    assert torch.allclose(ema.shadow_buffers["offset"], torch.tensor([2.0]))


def test_apply_to_swaps_buffers_and_rejects_reapply_before_restore():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    original_scale = model.scale.clone()

    ema.shadow_buffers["scale"].fill_(5.0)
    ema.apply_to(model)

    assert torch.allclose(model.scale, torch.tensor([5.0]))
    with pytest.raises(RuntimeError, match="already applied"):
        ema.apply_to(model)

    ema.restore(model)
    assert torch.allclose(model.scale, original_scale)
    assert not ema.collected_params
    assert not ema.collected_buffers


def test_restore_without_apply_is_noop():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    original_weight = model.lin.weight.clone()
    original_scale = model.scale.clone()

    ema.restore(model)

    assert torch.allclose(model.lin.weight, original_weight)
    assert torch.allclose(model.scale, original_scale)


def test_average_parameters_restores_after_exception():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.shadow_params["lin.weight"].add_(2.0)
    original_weight = model.lin.weight.clone()

    with pytest.raises(RuntimeError, match="boom"):
        with ema.average_parameters(model):
            assert torch.allclose(model.lin.weight, ema.shadow_params["lin.weight"])
            raise RuntimeError("boom")

    assert torch.allclose(model.lin.weight, original_weight)
    assert not ema.collected_params


def test_load_state_dict_restores_device_and_clears_collected_caches():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    state = ema.state_dict()
    state["decay"] = 0.25
    state["num_updates"] = None
    state["device"] = "cpu"

    ema2 = ExponentialMovingAverage(model, decay=0.1)
    ema2.apply_to(model)
    assert ema2.collected_params

    ema2.load_state_dict(state)

    assert ema2.decay == pytest.approx(0.25)
    assert ema2.num_updates is None
    assert ema2.device == torch.device("cpu")
    assert not ema2.collected_params
    assert not ema2.collected_buffers
    for tensor in ema2.shadow_params.values():
        assert tensor.device.type == "cpu"
