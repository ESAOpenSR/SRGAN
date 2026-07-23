import importlib
import os
import runpy
import sys
from types import SimpleNamespace

import pytest


class DummySRGAN:
    def __init__(self):
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return self


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture()
def inference_module():
    return importlib.reload(importlib.import_module("opensr_srgan.inference"))


def test_load_model_defaults_cpu(monkeypatch, inference_module):
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    calls = {}
    dummy_model = DummySRGAN()

    def fake_load_from_config(config_path, checkpoint_uri, **kwargs):
        calls.update(
            config_path=config_path,
            checkpoint_uri=checkpoint_uri,
            **kwargs,
        )
        return dummy_model

    monkeypatch.setattr(inference_module, "load_from_config", fake_load_from_config)

    model, device = inference_module.load_model(config_path="cfg.yaml")

    assert device == "cpu"
    assert model is dummy_model
    assert model.to_device == "cpu"
    assert calls == {
        "config_path": "cfg.yaml",
        "checkpoint_uri": None,
        "map_location": "cpu",
        "mode": "eval",
    }


def test_load_model_with_checkpoint(monkeypatch, inference_module):
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    calls = {}
    dummy_model = DummySRGAN()

    def fake_load_from_config(config_path, checkpoint_uri, **kwargs):
        calls.update(
            config_path=config_path,
            checkpoint_uri=checkpoint_uri,
            **kwargs,
        )
        return dummy_model

    monkeypatch.setattr(inference_module, "load_from_config", fake_load_from_config)

    model, device = inference_module.load_model(
        config_path="cfg.yaml", ckpt_path="weights.ckpt"
    )

    assert device == "cpu"
    assert model is dummy_model
    assert model.to_device == "cpu"
    assert calls == {
        "config_path": "cfg.yaml",
        "checkpoint_uri": "weights.ckpt",
        "map_location": "cpu",
        "mode": "eval",
    }


def test_run_sen2_inference_invokes_pipeline(monkeypatch, inference_module):
    dummy_model = object()

    def fake_load_model(**kwargs):
        return dummy_model, "cpu"

    created_objects = {}

    class DummyProcessor:
        def __init__(self, **kwargs):
            created_objects.update(kwargs)
            self.start_called = False

        def start_super_resolution(self):
            self.start_called = True

    dummy_utils = SimpleNamespace(large_file_processing=DummyProcessor)
    monkeypatch.setitem(sys.modules, "opensr_utils", dummy_utils)
    monkeypatch.setattr(inference_module, "load_model", fake_load_model)

    result = inference_module.run_sen2_inference(
        sen2_path="/tmp/safe",
        config_path="cfg.yaml",
        ckpt_path="weights.ckpt",
        window_size=(64, 64),
        overlap=4,
        eliminate_border_px=1,
        save_preview=True,
        debug=True,
    )

    assert isinstance(result, DummyProcessor)
    assert result.start_called is True
    assert created_objects["root"] == "/tmp/safe"
    assert created_objects["model"] is dummy_model
    assert created_objects["window_size"] == (64, 64)
    assert created_objects["overlap"] == 4
    assert created_objects["eliminate_border_px"] == 1
    assert created_objects["save_preview"] is True
    assert created_objects["debug"] is True
    assert created_objects["gpus"] == []


def test_run_sen2_inference_sets_cuda_devices(monkeypatch, inference_module):
    dummy_model = object()
    monkeypatch.setattr(
        inference_module, "load_model", lambda **_: (dummy_model, "cuda")
    )

    class DummyProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.start_called = False

        def start_super_resolution(self):
            self.start_called = True

    dummy_utils = SimpleNamespace(large_file_processing=DummyProcessor)
    monkeypatch.setitem(sys.modules, "opensr_utils", dummy_utils)

    inference_module.run_sen2_inference(
        sen2_path="/tmp/safe",
        gpus=[1, 2],
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_main_calls_run_sen2_inference(monkeypatch, inference_module):
    recorded = {}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def fake_run_sen2_inference(**kwargs):
        recorded.update(kwargs)
        return "ok"

    monkeypatch.setattr(inference_module, "run_sen2_inference", fake_run_sen2_inference)

    inference_module.main()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert recorded["gpus"] == [0]
    assert recorded["ckpt_path"].endswith("last.ckpt")
    assert recorded["config_path"].endswith("config_20m.yaml")
    assert recorded["sen2_path"].endswith("S2A_MSIL2A_EXAMPLE.SAFE")


def test_inference_module_main_guard(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    from opensr_srgan import _factory

    monkeypatch.setattr(_factory, "load_from_config", lambda *_, **__: DummySRGAN())

    class DummyProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start_super_resolution(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "opensr_utils",
        SimpleNamespace(large_file_processing=DummyProcessor),
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["inference.py"])

    runpy.run_module("opensr_srgan.inference", run_name="__main__")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
