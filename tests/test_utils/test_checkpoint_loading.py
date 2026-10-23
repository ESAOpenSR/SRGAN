from datetime import datetime
import warnings

import pytest
import torch

from opensr_srgan.utils import checkpoint_loading


def test_load_checkpoint_retries_without_weights_only_for_old_torch(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {"loaded": path}

    monkeypatch.setattr(checkpoint_loading.torch, "load", fake_load)

    assert checkpoint_loading.load_checkpoint("model.ckpt", map_location="cpu") == {
        "loaded": "model.ckpt"
    }
    assert calls == [
        {"weights_only": True, "map_location": "cpu"},
        {"map_location": "cpu"},
    ]


def test_load_checkpoint_warns_and_retries_unsafe_fallback(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("weights_only") is True:
            raise RuntimeError("legacy object")
        return {"loaded": path}

    monkeypatch.setattr(checkpoint_loading.torch, "load", fake_load)

    with pytest.warns(RuntimeWarning, match="weights_only=False"):
        result = checkpoint_loading.load_checkpoint("legacy.ckpt")

    assert result == {"loaded": "legacy.ckpt"}
    assert calls == [
        {"weights_only": True, "map_location": None},
        {"weights_only": False, "map_location": None},
    ]


def test_load_checkpoint_loads_tensor_checkpoint_without_warning(tmp_path):
    path = tmp_path / "weights.ckpt"
    weights = torch.arange(4)
    torch.save({"weights": weights}, path)

    with warnings.catch_warnings(record=True) as caught:
        result = checkpoint_loading.load_checkpoint(path, map_location="cpu")

    assert torch.equal(result["weights"], weights)
    assert not caught


def test_load_checkpoint_loads_legacy_objects_with_warning(tmp_path):
    path = tmp_path / "legacy.ckpt"
    created = datetime(2020, 1, 1)
    torch.save({"created": created, "weights": torch.arange(4)}, path)

    with pytest.warns(RuntimeWarning, match="weights_only=False"):
        result = checkpoint_loading.load_checkpoint(path, map_location="cpu")

    assert result["created"] == created
    assert torch.equal(result["weights"], torch.arange(4))
