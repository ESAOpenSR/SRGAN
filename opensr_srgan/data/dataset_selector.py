from pathlib import Path


def select_dataset(config):
    """
    Build train/val datasets from `config` and wrap them into a LightningDataModule.

    Expected `config` fields (OmegaConf/dict-like):
    - config.Data.dataset_type : str
        One of {"ExampleDataset", "SEN2NAIP", "LRHRFolderDataset"}.
    - config.Data.root_dir / config.Data.dataset_root : str, optional
        Required for LRHRFolderDataset.
    - config.Data.sen2naip_taco_file : str, optional
        Required for SEN2NAIP unless the dataset class receives another source.

    Returns
    -------
    pl_datamodule : LightningDataModule
        A tiny DataModule that exposes train/val DataLoaders built from the selected datasets.
    """
    dataset_selection = config.Data.dataset_type

    # Legacy keys from older configs intentionally fall through to the error below.

    if dataset_selection == "ExampleDataset":
        from opensr_srgan.data.example_data.example_dataset import ExampleDataset
        print("WARNING -- Using Example Dataset!")
        print("This dataset is exclusively meant for demonstration and debugging, not training or evaluation.")
        print("Please use a proper dataset for any serious work.")
        path = "example_dataset/"
        ds_train = ExampleDataset(folder=path, phase="train")
        ds_val = ExampleDataset(folder=path, phase="val")

    elif str(dataset_selection).lower() == "sen2naip":
        from opensr_srgan.data.sen2naip.sen2naip_dataset import SEN2NAIP

        taco_file = getattr(config.Data, "sen2naip_taco_file", None)
        ds_train = SEN2NAIP(config=config, phase="train", taco_file=taco_file)
        ds_val = SEN2NAIP(config=config, phase="val", taco_file=taco_file)
        
    elif dataset_selection == "LRHRFolderDataset":
        from opensr_srgan.data.lrhr_folder.lrhr_folder_dataset import LRHRFolderDataset

        root_folder = getattr(config.Data, "root_dir", None)
        if root_folder is None:
            root_folder = getattr(config.Data, "dataset_root", None)
        if root_folder is None:
            raise ValueError(
                "LRHRFolderDataset requires Data.root_dir or Data.dataset_root in the config."
            )

        path = Path(root_folder).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"LRHRFolderDataset root path does not exist: '{path}'. "
                "Set Data.root_dir to a valid dataset directory."
            )

        ds_train = LRHRFolderDataset(config=config, root_folder=path, phase="train")
        ds_val = LRHRFolderDataset(config=config, root_folder=path, phase="val")


    else:
        # Centralized error so unsupported keys fail loudly & clearly.
        raise NotImplementedError(
            f"Dataset {dataset_selection} not implemented!"
            f"This can happen when:"
            f" - (a) you misspelled the dataset name in the config"
            f" - (b) the dataset is not implemented in the data folder."
            f" - (c) you are trying to use a custom dataset but forgot to add it in data/dataset_selector.py."
        )

    # Wrap the two datasets into a LightningDataModule with config-driven loader knobs.
    pl_datamodule = datamodule_from_datasets(config, ds_train, ds_val)
    return pl_datamodule


def datamodule_from_datasets(config, ds_train, ds_val):
    """
    Convert a pair of prebuilt PyTorch Datasets into a minimal PyTorch Lightning DataModule.

    Parameters
    ----------
    config : OmegaConf/dict-like
        Expected to contain:
          - Data.train_batch_size : int (fallback: Data.batch_size or 8)
          - Data.val_batch_size   : int (fallback: Data.batch_size or 8)
          - Data.num_workers      : int (default: 4)
          - Data.prefetch_factor  : int (default: 2)
    ds_train : torch.utils.data.Dataset
        Training dataset (already instantiated).
    ds_val : torch.utils.data.Dataset
        Validation dataset (already instantiated).

    Returns
    -------
    LightningDataModule
        Exposes `train_dataloader()` and `val_dataloader()` using the settings above.
    """
    from pytorch_lightning import LightningDataModule
    from torch.utils.data import DataLoader

    class CustomDataModule(LightningDataModule):
        """Tiny DataModule that forwards config-driven loader settings to DataLoader."""

        def __init__(self, ds_train, ds_val, config):
            super().__init__()
            self.ds_train = ds_train
            self.ds_val = ds_val

            # Pull loader settings from config with safe fallbacks.
            self.train_bs = getattr(
                config.Data, "train_batch_size", getattr(config.Data, "batch_size", 8)
            )
            self.val_bs = getattr(
                config.Data, "val_batch_size", getattr(config.Data, "batch_size", 8)
            )
            self.num_workers = getattr(config.Data, "num_workers", 4)
            self.prefetch_factor = getattr(config.Data, "prefetch_factor", 2)

            # print dataset sizes for sanity
            print(
                f"Created Dataset type {config.Data.dataset_type} with {len(self.ds_train)} training samples and {len(self.ds_val)} validation samples.\n"
            )

        def train_dataloader(self):
            """Return the training DataLoader with common performance flags."""
            kwargs = dict(
                batch_size=self.train_bs,
                shuffle=True,  # Shuffle only in training
                num_workers=self.num_workers,
                pin_memory=True,  # Speeds up host→GPU transfer on CUDA
                persistent_workers=self.num_workers
                > 0,  # Keep workers alive between epochs
            )
            # prefetch_factor is only valid when num_workers > 0
            if self.num_workers > 0:
                kwargs["prefetch_factor"] = self.prefetch_factor
            return DataLoader(self.ds_train, **kwargs)

        def val_dataloader(self):
            """Return the validation DataLoader (no shuffle)."""
            kwargs = dict(
                batch_size=self.val_bs,
                shuffle=True,  # shuffle ordering for validation - more diversity in batches
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
            if self.num_workers > 0:
                kwargs["prefetch_factor"] = self.prefetch_factor
            return DataLoader(self.ds_val, **kwargs)

    return CustomDataModule(ds_train, ds_val, config)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config_path = "opensr_srgan/configs/config_xray.yaml"
    config = OmegaConf.load(config_path)
    _ = select_dataset(config)
