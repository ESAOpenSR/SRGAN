"""Data utilities exposed for documentation and package imports."""

from .dataset_selector import datamodule_from_datasets, select_dataset

__all__ = [
    "select_dataset",
    "datamodule_from_datasets",
]
