from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import deployment.srgan_hpc.inference as inference_module
from deployment.srgan_hpc.config import (
    InferenceConfig,
    ModelSourceConfig,
    ProductConfig,
)
from deployment.srgan_hpc.inference import (
    _build_runner,
    load_srgan_model,
    run_inference,
)


class FakeModel:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str):
        self.device = device
        return self


def test_load_srgan_model_uses_explicit_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_from_config(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("opensr_srgan.load_from_config", fake_load_from_config)

    model, device = load_srgan_model(
        ProductConfig(
            bands=["B04"],
            resolution=10,
            factor=4,
            model=ModelSourceConfig(
                preset=None,
                config_path="/tmp/config.yaml",
                checkpoint_path="/tmp/model.ckpt",
            ),
        )
    )

    assert device == "cpu"
    assert model.device == "cpu"
    assert captured["args"][:2] == ("/tmp/config.yaml", "/tmp/model.ckpt")
    assert captured["kwargs"]["mode"] == "eval"


def test_load_srgan_model_uses_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_inference_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("opensr_srgan.load_inference_model", fake_load_inference_model)

    model, device = load_srgan_model(
        ProductConfig(
            bands=["B04"],
            resolution=10,
            factor=4,
            model=ModelSourceConfig(preset="RGB-NIR", cache_dir="/tmp/cache"),
        )
    )

    assert device == "cpu"
    assert model.device == "cpu"
    assert captured["args"] == ("RGB-NIR",)
    assert captured["kwargs"]["cache_dir"] == "/tmp/cache"


def test_load_srgan_model_requires_source() -> None:
    with pytest.raises(ValueError, match="preset or config_path"):
        load_srgan_model(
            ProductConfig(
                bands=["B04"],
                resolution=10,
                factor=4,
                model=ModelSourceConfig(preset=None),
            )
        )


def test_build_runner_creates_datamodule(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class BaseLargeFileProcessing:
        def __init__(self, *args, **kwargs) -> None:
            self.input_type = "geotiff"
            self.root = kwargs["root"]
            self.image_meta = {
                "image_windows": ["window"],
                "lr_file_dict": {"B04": "input.tif"},
            }
            self.messages: list[str] = []

        def _log(self, message: str) -> None:
            self.messages.append(message)

    class FakePredictionDataModule:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            self.dataset = [object(), object()]
            self.did_setup = False

        def setup(self) -> None:
            self.did_setup = True

    monkeypatch.setitem(
        sys.modules,
        "opensr_utils.data_utils.datamodule",
        types.SimpleNamespace(PredictionDataModule=FakePredictionDataModule),
    )
    runner_cls = _build_runner(
        types.SimpleNamespace(large_file_processing=BaseLargeFileProcessing),
        InferenceConfig(batch_size=7),
    )

    runner = runner_cls(root="/tmp/input.tif", batch_size=7)
    runner.create_datamodule()

    assert created["batch_size"] == 7
    assert created["num_workers"] == 4
    assert runner.datamodule.did_setup is True
    assert "batch_size=7" in runner.messages[0]


def test_run_inference_compresses_runner_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_tif = tmp_path / "input.tif"
    input_tif.write_bytes(b"input")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    sr_path = output_dir / "raw_sr.tif"
    sr_path.write_bytes(b"raw")
    captured: dict[str, object] = {}

    class FakeLargeFileProcessing:
        final_path = sr_path

        def __init__(self, **kwargs) -> None:
            captured["runner_kwargs"] = kwargs
            self.final_sr_path = self.final_path

        def start_super_resolution(self) -> None:
            captured["started"] = True

    def fake_compress(src_path: Path, dest_path: Path, band_names: list[str]):
        captured["compress"] = (src_path, dest_path, band_names)
        dest_path.write_bytes(src_path.read_bytes())
        return dest_path

    monkeypatch.setitem(
        sys.modules,
        "opensr_utils",
        types.SimpleNamespace(large_file_processing=FakeLargeFileProcessing),
    )
    monkeypatch.setattr(
        inference_module, "load_srgan_model", lambda product: (FakeModel(), "cpu")
    )
    monkeypatch.setattr(inference_module, "compress_geotiff", fake_compress)

    result = run_inference(
        input_tif=input_tif,
        output_dir=output_dir,
        product_name="rgbnir",
        product=ProductConfig(
            bands=["B04", "B03"],
            resolution=10,
            factor=4,
            model=ModelSourceConfig(preset="RGB-NIR"),
        ),
        inference=InferenceConfig(window_size=(64, 96), batch_size=3, overlap=8),
    )

    assert captured["started"] is True
    assert captured["runner_kwargs"]["root"] == str(input_tif)
    assert captured["runner_kwargs"]["window_size"] == (64, 96)
    assert "batch_size" not in captured["runner_kwargs"]
    assert result == output_dir / "rgbnir_sr.tif"
    assert captured["compress"] == (sr_path, result, ["B04", "B03"])
    assert not sr_path.exists()


def test_run_inference_uses_default_sr_path_when_runner_does_not_set_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_tif = tmp_path / "input.tif"
    input_tif.write_bytes(b"input")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    default_sr = output_dir / "sr.tif"
    default_sr.write_bytes(b"raw")

    class FakeLargeFileProcessing:
        def __init__(self, **_kwargs) -> None:
            pass

        def start_super_resolution(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "opensr_utils",
        types.SimpleNamespace(large_file_processing=FakeLargeFileProcessing),
    )
    monkeypatch.setattr(
        inference_module, "load_srgan_model", lambda product: (FakeModel(), "cpu")
    )
    monkeypatch.setattr(
        inference_module,
        "compress_geotiff",
        lambda src_path, dest_path, band_names: dest_path.write_bytes(
            src_path.read_bytes()
        ),
    )

    result = run_inference(
        input_tif=input_tif,
        output_dir=output_dir,
        product_name="swir",
        product=ProductConfig(
            bands=["B11"],
            resolution=20,
            factor=8,
            model=ModelSourceConfig(preset="SWIR"),
        ),
        inference=InferenceConfig(),
    )

    assert result == output_dir / "swir_sr.tif"
    assert not default_sr.exists()


def test_run_inference_rejects_missing_runner_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLargeFileProcessing:
        def __init__(self, **_kwargs) -> None:
            self.final_sr_path = tmp_path / "missing.tif"

        def start_super_resolution(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "opensr_utils",
        types.SimpleNamespace(large_file_processing=FakeLargeFileProcessing),
    )
    monkeypatch.setattr(
        inference_module, "load_srgan_model", lambda product: (FakeModel(), "cpu")
    )

    with pytest.raises(FileNotFoundError, match="Expected SR output"):
        run_inference(
            input_tif=tmp_path / "input.tif",
            output_dir=tmp_path,
            product_name="rgbnir",
            product=ProductConfig(
                bands=["B04"],
                resolution=10,
                factor=4,
                model=ModelSourceConfig(preset="RGB-NIR"),
            ),
            inference=InferenceConfig(),
        )
