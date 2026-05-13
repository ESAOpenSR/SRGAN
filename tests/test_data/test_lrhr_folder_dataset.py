from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opensr_srgan.data.lrhr_folder.lrhr_folder_dataset import LRHRFolderDataset


def _write_pair(
    root: Path, phase: str, name: str, lr_shape=(8, 8, 3), hr_shape=(16, 16, 3)
):
    (root / phase / "LR").mkdir(parents=True, exist_ok=True)
    (root / phase / "HR").mkdir(parents=True, exist_ok=True)
    np.save(root / phase / "LR" / name, np.zeros(lr_shape, dtype=np.float32))
    np.save(root / phase / "HR" / name, np.ones(hr_shape, dtype=np.float32))


def _make_config(root: Path, normalization="identity"):
    return SimpleNamespace(
        Data=SimpleNamespace(
            dataset_type="LRHRFolderDataset",
            root_dir=str(root),
            normalization=normalization,
        )
    )


def test_lrhr_folder_dataset_reads_phase_pairs(tmp_path):
    _write_pair(tmp_path, "train", "a.npy")
    _write_pair(tmp_path, "train", "b.npy")
    _write_pair(tmp_path, "val", "v.npy")
    _write_pair(tmp_path, "test", "t.npy")

    ds = LRHRFolderDataset(config=_make_config(tmp_path), phase="train")
    assert len(ds) == 2

    lr, hr = ds[0]
    assert isinstance(lr, torch.Tensor)
    assert isinstance(hr, torch.Tensor)
    assert lr.shape == (3, 8, 8)
    assert hr.shape == (3, 16, 16)


def test_lrhr_folder_dataset_invalid_phase_raises(tmp_path):
    with pytest.raises(ValueError):
        LRHRFolderDataset(config=_make_config(tmp_path), phase="dev")


def test_lrhr_folder_dataset_missing_pair_raises(tmp_path):
    (tmp_path / "train" / "LR").mkdir(parents=True, exist_ok=True)
    (tmp_path / "train" / "HR").mkdir(parents=True, exist_ok=True)
    np.save(
        tmp_path / "train" / "LR" / "only_lr.npy", np.zeros((8, 8, 1), dtype=np.float32)
    )

    with pytest.raises(FileNotFoundError):
        LRHRFolderDataset(config=_make_config(tmp_path), phase="train")


def test_lrhr_folder_dataset_applies_normalization(tmp_path):
    (tmp_path / "train" / "LR").mkdir(parents=True, exist_ok=True)
    (tmp_path / "train" / "HR").mkdir(parents=True, exist_ok=True)
    np.save(
        tmp_path / "train" / "LR" / "x.npy",
        np.full((4, 4, 1), 10000.0, dtype=np.float32),
    )
    np.save(
        tmp_path / "train" / "HR" / "x.npy",
        np.full((8, 8, 1), 10000.0, dtype=np.float32),
    )

    ds = LRHRFolderDataset(
        config=_make_config(tmp_path, "normalise_10k"), phase="train"
    )
    lr, hr = ds[0]

    assert torch.allclose(lr, torch.ones_like(lr))
    assert torch.allclose(hr, torch.ones_like(hr))


def test_lrhr_folder_dataset_rejects_wrong_dataset_type(tmp_path):
    _write_pair(tmp_path, "train", "a.npy")
    bad_cfg = SimpleNamespace(
        Data=SimpleNamespace(
            dataset_type="ExampleDataset",
            root_dir=str(tmp_path),
            normalization="identity",
        )
    )

    with pytest.raises(ValueError):
        LRHRFolderDataset(config=bad_cfg, phase="train")
