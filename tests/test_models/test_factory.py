from pathlib import Path
import sys
import types
from dataclasses import replace

import pytest
import torch
from torch import nn

from opensr_srgan import _factory
from opensr_srgan import pretrained


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


def test_load_from_config_does_not_restore_checkpoint_cuda_ema_device(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test")
    checkpoint_path = tmp_path / "state.ckpt"
    torch.save(
        {
            "state_dict": {"weight": torch.tensor([1])},
            "ema_state": {"device": "cuda:0"},
        },
        checkpoint_path,
    )
    received = {}

    class ModelWithEma(DummyModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config = types.SimpleNamespace(
                Training=types.SimpleNamespace(EMA=types.SimpleNamespace(device=None))
            )
            self.ema = types.SimpleNamespace(
                load_state_dict=lambda state: received.update(state)
            )

    monkeypatch.setattr(_factory, "SRGAN_model", ModelWithEma)
    _factory.load_from_config(config_path, checkpoint_path)

    assert received["device"] is None


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


def test_sen2_sr_preset_is_strict_and_passes_revision(monkeypatch, tmp_path):
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"].split("/")[-1])

    huggingface_stub = types.ModuleType("huggingface_hub")
    huggingface_stub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_stub)
    monkeypatch.setitem(
        _factory._PRESETS,
        "SEN2-SR",
        replace(_factory._PRESETS["SEN2-SR"], revision="immutable-test-revision"),
    )

    captured = {}

    def fake_load_from_config(config_path, checkpoint_uri, **kwargs):
        captured.update(config_path=config_path, checkpoint_uri=checkpoint_uri, **kwargs)
        return object()

    monkeypatch.setattr(_factory, "load_from_config", fake_load_from_config)
    _factory.load_inference_model("sen2_sr")

    assert [call["filename"] for call in calls] == [
        "sen2-sr-v1/config.yaml",
        "sen2-sr-v1/model_ema.ckpt",
    ]
    assert all(call["revision"] == "immutable-test-revision" for call in calls)
    assert captured["strict"] is True
    assert captured["mode"] == "eval"


class _TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))


class _TinyInferenceModel:
    def __init__(self, config, mode):
        assert mode == "eval"
        self.generator = _TinyGenerator()


def test_export_ema_generator_checkpoint_is_minimal_and_cpu_safe(monkeypatch, tmp_path):
    source = tmp_path / "trusted.ckpt"
    target = tmp_path / "release" / "model_ema.ckpt"
    config = tmp_path / "config.yaml"
    config.write_text("unused: test")
    torch.save(
        {
            "state_dict": {
                "generator.weight": torch.full((2, 3), -1.0),
                "discriminator.weight": torch.ones(1),
            },
            "optimizer_states": [{"should": "not be exported"}],
            "global_step": 41,
            "epoch": 3,
            "ema_state": {
                "device": "cuda:0",
                "decay": 0.99,
                "num_updates": 37,
                "shadow_params": {"weight": torch.full((2, 3), 2.0)},
                "shadow_buffers": {"counter": torch.tensor(9, dtype=torch.long)},
            },
        },
        source,
    )
    monkeypatch.setattr(pretrained, "SRGAN_model", _TinyInferenceModel)

    metadata = pretrained.export_ema_generator_checkpoint(
        source, target, config_path=config
    )
    exported = torch.load(target, map_location="cpu", weights_only=True)

    assert set(exported) == {"state_dict", "metadata"}
    assert set(exported["state_dict"]) == {"generator.weight", "generator.counter"}
    assert torch.equal(
        exported["state_dict"]["generator.weight"], torch.full((2, 3), 2.0)
    )
    assert exported["state_dict"]["generator.weight"].device.type == "cpu"
    assert exported["metadata"] == metadata
    assert metadata["source_global_step"] == 41
    assert metadata["ema_num_updates"] == 37


def test_export_ema_generator_checkpoint_rejects_incomplete_ema(monkeypatch, tmp_path):
    source = tmp_path / "trusted.ckpt"
    config = tmp_path / "config.yaml"
    config.write_text("unused: test")
    torch.save(
        {
            "state_dict": {"generator.weight": torch.zeros(2, 3)},
            "ema_state": {
                "shadow_params": {"weight": torch.zeros(2, 3)},
                "shadow_buffers": {},
            },
        },
        source,
    )
    monkeypatch.setattr(pretrained, "SRGAN_model", _TinyInferenceModel)

    with pytest.raises(ValueError, match="missing: generator.counter"):
        pretrained.export_ema_generator_checkpoint(
            source, tmp_path / "model_ema.ckpt", config_path=config
        )


def test_export_ema_generator_checkpoint_never_replaces_its_source(tmp_path):
    source = tmp_path / "trusted.ckpt"
    source.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="Destination must differ"):
        pretrained.export_ema_generator_checkpoint(source, source)
