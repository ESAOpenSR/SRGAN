import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opensr_srgan.data.sen2naip.sen2naip_dataset import SEN2NAIP


class _FakeSample:
    def read(self, idx):
        return "lr.tif" if idx == 0 else "hr.tif"


class _FakeTacoDataset:
    def __init__(self):
        self.iloc = [_FakeRow({"tortilla:id": f"id-{index}"}) for index in range(4)]

    def __len__(self):
        return 4

    def read(self, _idx):
        return _FakeSample()


class _FakeRow(dict):
    pass


class _FakeRaster:
    def __init__(self, path):
        self.path = path
        self.count = 1
        self.width = 2 if path == "lr.tif" else 8
        self.height = 2 if path == "lr.tif" else 8

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, window=None):
        height, width = self.height, self.width
        if window is not None:
            height, width = int(window.height), int(window.width)
        if self.path == "lr.tif":
            return np.full((1, height, width), 5000.0, dtype=np.float32)
        return np.full((1, height, width), 10000.0, dtype=np.float32)


@pytest.fixture(autouse=True)
def _stub_external_modules(monkeypatch):
    tacoreader_mod = types.ModuleType("tacoreader")
    tacoreader_mod.load = lambda _path: _FakeTacoDataset()
    monkeypatch.setitem(sys.modules, "tacoreader", tacoreader_mod)

    rasterio_mod = types.ModuleType("rasterio")
    rasterio_mod.open = lambda path: _FakeRaster(path)
    monkeypatch.setitem(sys.modules, "rasterio", rasterio_mod)
    windows_mod = types.ModuleType("rasterio.windows")
    windows_mod.Window = lambda col_off, row_off, width, height: types.SimpleNamespace(
        col_off=col_off, row_off=row_off, width=width, height=height
    )
    monkeypatch.setitem(sys.modules, "rasterio.windows", windows_mod)


def test_sen2naip_uses_config_normalization():
    cfg = SimpleNamespace(
        Data=SimpleNamespace(
            normalization="normalise_10k",
            sen2naip_taco_file="/tmp/fake.taco",
            sen2naip_phase="train",
            sen2naip_val_fraction=0.25,
        )
    )
    ds = SEN2NAIP(cfg)

    lr, hr = ds[0]

    assert ds.normalizer.method == "normalise_10k"
    assert isinstance(lr, torch.Tensor)
    assert isinstance(hr, torch.Tensor)
    assert torch.allclose(lr, torch.full_like(lr, 0.5))
    assert torch.allclose(hr, torch.full_like(hr, 1.0))


def test_sen2naip_requires_taco_file():
    cfg = SimpleNamespace(
        Data=SimpleNamespace(
            normalization="identity",
            sen2naip_phase="train",
            sen2naip_val_fraction=0.25,
        )
    )

    with pytest.raises(ValueError, match="sen2naip_taco_file"):
        SEN2NAIP(cfg)


def test_sen2naip_aligned_crop_and_split_are_reproducible():
    cfg = SimpleNamespace(
        Data=SimpleNamespace(
            normalization="normalise_10k",
            sen2naip_taco_file="/tmp/fake.taco",
            sen2naip_val_fraction=0.25,
            sen2naip_crop_size=1,
            sen2naip_split_seed=7,
            sen2naip_num_bands=1,
            sen2naip_augment=False,
        )
    )
    train = SEN2NAIP(cfg, phase="train")
    val = SEN2NAIP(cfg, phase="val")
    assert set(train.indices).isdisjoint(val.indices)
    lr, hr = train[0]
    assert lr.shape == (1, 1, 1)
    assert hr.shape == (1, 4, 4)
    assert torch.allclose(lr, torch.full_like(lr, 0.5))
    assert torch.allclose(hr, torch.full_like(hr, 1.0))


def test_sen2naip_validates_explicit_split_ids_and_coverage(tmp_path):
    import json

    split = {
        "nrecords": 4,
        "indices": {"train": [0, 1], "val": [2], "test": [3]},
        "ids": {"train": ["id-0", "id-1"], "val": ["id-2"], "test": ["id-3"]},
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    cfg = SimpleNamespace(Data=SimpleNamespace(
        normalization="identity", sen2naip_taco_file="fake",
        sen2naip_split_file=str(split_path),
    ))
    # The fixture's fake rows need the same iloc shape as pandas.
    import sys
    sys.modules["tacoreader"].load = lambda _path: _FakeTacoDataset()
    ds = SEN2NAIP(cfg, phase="val")
    assert ds.indices == [2]

    split["ids"]["val"] = ["tampered"]
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="split IDs"):
        SEN2NAIP(cfg, phase="val")

    split["ids"]["val"] = ["id-2"]
    split["indices"]["val"] = [1]
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="disjoint"):
        SEN2NAIP(cfg, phase="val")
