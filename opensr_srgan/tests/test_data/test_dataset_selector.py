import sys
import types

import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from opensr_srgan.data import dataset_selector


class _StubExampleDataset(Dataset):
    """Simple dataset used to intercept ExampleDataset creation in tests."""

    created_args = []

    def __init__(self, folder, phase):
        # record constructor arguments for assertions
        self.__class__.created_args.append((folder, phase))
        self.phase = phase
        # give the train dataset more samples than val to ensure lengths differ
        length = 6 if phase == "train" else 3
        self._data = torch.arange(length, dtype=torch.float32)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # return deterministic single-value tensors for DataLoader consumption
        return self._data[idx]


class _LightningDataModuleStub:
    """Minimal stand-in for ``pytorch_lightning.LightningDataModule``."""

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def stub_lightning(monkeypatch):
    module = types.ModuleType("pytorch_lightning")
    module.LightningDataModule = _LightningDataModuleStub
    monkeypatch.setitem(sys.modules, "pytorch_lightning", module)
    return module


def _make_config(**data_kwargs):
    """Helper that builds a lightweight config namespace for the selector."""

    data_namespace = types.SimpleNamespace(**data_kwargs)
    generator_namespace = types.SimpleNamespace(scaling_factor=4)
    return types.SimpleNamespace(Data=data_namespace, Generator=generator_namespace)


def test_select_dataset_example_uses_stub(monkeypatch, capsys):
    # Patch the ExampleDataset with our stub so the selector avoids filesystem I/O.
    import opensr_srgan.data.example_data.example_dataset as example_module

    monkeypatch.setattr(example_module, "ExampleDataset", _StubExampleDataset)
    _StubExampleDataset.created_args.clear()

    config = _make_config(
        dataset_type="ExampleDataset",
        train_batch_size=2,
        val_batch_size=3,
        num_workers=0,
    )

    datamodule = dataset_selector.select_dataset(config)

    # Ensure both train and validation datasets were instantiated with the expected path.
    assert _StubExampleDataset.created_args == [
        ("example_dataset/", "train"),
        ("example_dataset/", "val"),
    ]

    # The initialization prints dataset statistics; capture the output to verify it references the selection.
    printed = capsys.readouterr().out
    assert "ExampleDataset" in printed
    assert "training samples" in printed

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 3
    assert train_loader.num_workers == 0
    assert not train_loader.persistent_workers

    first_batch = next(iter(train_loader))
    assert isinstance(first_batch, torch.Tensor)
    assert first_batch.shape[0] == 2


def test_datamodule_from_datasets_uses_defaults():
    train_ds = TensorDataset(torch.arange(10))
    val_ds = TensorDataset(torch.arange(4))

    config = _make_config(
        dataset_type="stub",
        batch_size=5,
        num_workers=1,
        prefetch_factor=3,
    )

    datamodule = dataset_selector.datamodule_from_datasets(config, train_ds, val_ds)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader.batch_size == 5
    assert val_loader.batch_size == 5
    assert train_loader.num_workers == 1
    assert train_loader.prefetch_factor == 3
    assert train_loader.persistent_workers is True
    assert val_loader.prefetch_factor == 3


def test_select_dataset_unknown_raises():
    config = _make_config(dataset_type="does-not-exist")

    with pytest.raises(NotImplementedError) as exc:
        dataset_selector.select_dataset(config)

    assert "does-not-exist" in str(exc.value)
