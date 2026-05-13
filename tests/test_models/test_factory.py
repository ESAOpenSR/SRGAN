from pathlib import Path
import sys
import types

import pytest
import torch

from opensr_srgan import _factory


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.loaded_state = None
        self.ema = types.SimpleNamespace(load_state_dict=lambda state: state)
        self.is_eval = False

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = (state_dict, strict)

    def eval(self):
        self.is_eval = True
        return self


def test_maybe_download_returns_local_path(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    with _factory._maybe_download(checkpoint) as resolved:
        assert resolved == checkpoint
        assert resolved.read_bytes() == b"checkpoint"


def test_maybe_download_downloads_http(monkeypatch, tmp_path):
    target = tmp_path / "downloaded.ckpt"

    def fake_download(url, filename, progress):
        Path(filename).write_bytes(b"downloaded")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)

    url = "https://example.com/weights.ckpt"
    with _factory._maybe_download(url) as resolved:
        assert resolved.exists()
        assert resolved.read_bytes() == b"downloaded"
        assert resolved.suffix == ".ckpt"


def test_load_from_config_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        _factory.load_from_config(Path("nonexistent.yaml"))


def test_load_from_config_loads_state_and_ema(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test")

    checkpoint_path = tmp_path / "state.ckpt"
    state = {"state_dict": {"weight": torch.tensor([1, 2, 3])}, "ema_state": {"w": 1}}
    torch.save(state, checkpoint_path)

    monkeypatch.setattr(_factory, "SRGAN_model", DummyModel)

    model = _factory.load_from_config(config_path, checkpoint_path)

    assert isinstance(model, DummyModel)
    assert model.loaded_state is not None
    assert model.loaded_state[1] is False
    assert model.is_eval is True


def test_load_from_config_without_checkpoint_still_returns_eval_model(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test")

    monkeypatch.setattr(_factory, "SRGAN_model", DummyModel)
    model = _factory.load_from_config(config_path)

    assert isinstance(model, DummyModel)
    assert model.loaded_state is None
    assert model.is_eval is True


def test_maybe_download_raises_for_non_path_non_url():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        with _factory._maybe_download("not-a-real-ckpt-name"):
            pass


def test_load_inference_model_raises_for_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        _factory.load_inference_model("unknown-model")


def test_load_inference_model_downloads_and_calls_load_from_config(
    monkeypatch, tmp_path
):
    calls = []

    def fake_hf_hub_download(repo_id, filename, cache_dir=None):
        calls.append((repo_id, filename, cache_dir))
        return str(tmp_path / filename)

    huggingface_stub = types.ModuleType("huggingface_hub")
    huggingface_stub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_stub)

    sentinel = object()

    def fake_load_from_config(
        config_path, checkpoint_uri, *, map_location=None, mode="train"
    ):
        assert config_path.endswith(".yaml")
        assert checkpoint_uri.endswith(".ckpt")
        assert map_location == "cpu"
        assert mode == "eval"
        return sentinel

    monkeypatch.setattr(_factory, "load_from_config", fake_load_from_config)

    model = _factory.load_inference_model(
        "rgb_nir", cache_dir=tmp_path, map_location="cpu"
    )

    assert model is sentinel
    assert len(calls) == 2
    assert calls[0][1].endswith(".yaml")
    assert calls[1][1].endswith(".ckpt")
