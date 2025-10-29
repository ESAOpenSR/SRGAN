def select_dataset(config):
    """
    Build train/val datasets from `config` and wrap them into a LightningDataModule.

    Expected `config` fields (OmegaConf/dict-like):
    - config.Data.dataset_selection : str
        One of {"S2_6b", "S2_4b"} in this file.
    - config.Generator.scaling_factor : int
        Super-resolution scale factor (e.g., 2, 4, 8). Passed to dataset as `sr_factor`.

    Hard-coded choices below (kept as-is, not modified):
    - manifest_json : Path to prebuilt SAFE window manifest.
    - band orders   : Fixed lists for each selection.
    - hr_size       : (512, 512)
    - group_by      : "granule"
    - group_regex   : r".*?/GRANULE/([^/]+)/IMG_DATA/.*"

    Returns
    -------
    pl_datamodule : LightningDataModule
        A tiny DataModule that exposes train/val DataLoaders built from the selected datasets.
    """
    dataset_selection = config.Data.dataset_type

    # Please Note: The "S2_6b","S2_4b","SISR_WW" settings are leftover from previous versions
    # I dont want to delete them in case they are needed again.
    # Only the "ExampleDataset" is actively used in the current version.

    if dataset_selection == "S2_6b":
        # Import here to avoid import costs when other datasets are used elsewhere.
        from .SEN2_SAFE.S2_6b_ds import S2SAFEDataset

        # 6 × 20 m bands (B05, B06, B07, B8A, B11, B12) in a fixed order
        desired_20m_order = [
            "B05_20m",
            "B06_20m",
            "B07_20m",
            "B8A_20m",
            "B11_20m",
            "B12_20m",
        ]

        # NOTE: This manifest path is hard-coded on purpose (per your snippet).
        # Consider moving it into config later if you want to switch datasets easily.
        ds_train = S2SAFEDataset(
            phase="train",
            manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",
            group_by="granule",
            group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
            bands_keep=desired_20m_order,
            band_order=desired_20m_order,
            dtype="float32",
            hr_size=(512, 512),
            sr_factor=config.Generator.scaling_factor,
            antialias=True,
        )
        ds_val = S2SAFEDataset(
            phase="val",
            manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",
            group_by="granule",
            group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
            bands_keep=desired_20m_order,
            band_order=desired_20m_order,
            dtype="float32",
            hr_size=(512, 512),
            sr_factor=config.Generator.scaling_factor,
            antialias=True,
        )

    elif dataset_selection == "S2_4b":
        # If there is a dedicated 4-band dataset file, swap the import accordingly.
        from .SEN2_SAFE.S2_6b_ds import S2SAFEDataset

        # 4 × 10 m bands (R, G, B, NIR) in a fixed order.
        # FYI: The manifest path below still points to the 20 m manifest. If you truly
        # use 10 m inputs here, you may want a 10 m manifest. Keeping as-is intentionally.
        desired_20m_order = ["B05_10m", "B04_10m", "B03_10m", "B02_10m"]

        ds_train = S2SAFEDataset(
            phase="train",
            manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",  # See FYI above.
            group_by="granule",
            group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
            bands_keep=desired_20m_order,
            band_order=desired_20m_order,
            dtype="float32",
            hr_size=(512, 512),
            sr_factor=config.Generator.scaling_factor,
            antialias=True,
        )
        ds_val = S2SAFEDataset(
            phase="val",
            manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",  # See FYI above.
            group_by="granule",
            group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
            bands_keep=desired_20m_order,
            band_order=desired_20m_order,
            dtype="float32",
            hr_size=(512, 512),
            sr_factor=config.Generator.scaling_factor,
            antialias=True,
        )

    elif dataset_selection == "SISR_WW":
        from .SISR_WW.SISR_WW_dataset import SISRWorldWide

        path = "/data3/SEN2NAIP_global"
        ds_train = SISRWorldWide(path=path, split="train")
        ds_val = SISRWorldWide(path=path, split="val")

    elif dataset_selection == "ExampleDataset":
        from opensr_srgan.data.example_data.example_dataset import ExampleDataset

        path = "example_dataset/"
        ds_train = ExampleDataset(folder=path, phase="train")
        ds_val = ExampleDataset(folder=path, phase="val")
        
    elif dataset_selection == "ChestXRay":
        from opensr_srgan.data.chestxrays.dataset import XRayDataset

        ds_train = XRayDataset(phase="train", sr_factor=config.Generator.scaling_factor)
        ds_val = XRayDataset(phase="val", sr_factor=config.Generator.scaling_factor)
    else:
        # Centralized error so unsupported keys fail loudly & clearly.
        raise NotImplementedError(
            f"Dataset {dataset_selection} not implemented!"
            f"Add your dataset in data/dataset_selector.py to train on that."
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
