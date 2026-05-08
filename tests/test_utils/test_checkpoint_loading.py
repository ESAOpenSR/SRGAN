import pytest

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
        if "weights_only" in kwargs:
            raise RuntimeError("legacy object")
        return {"loaded": path}

    monkeypatch.setattr(checkpoint_loading.torch, "load", fake_load)

    with pytest.warns(RuntimeWarning, match="weights_only=False"):
        result = checkpoint_loading.load_checkpoint("legacy.ckpt")

    assert result == {"loaded": "legacy.ckpt"}
    assert calls == [
        {"weights_only": True, "map_location": None},
        {"map_location": None},
    ]
