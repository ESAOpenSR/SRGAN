from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from opensr_srgan.data.utils.normalizer import Normalizer


class SEN2NAIP(torch.utils.data.Dataset):
    """SEN2NAIP cross-sensor dataset loader backed by a Taco manifest."""

    def __init__(self, config: Any, phase=None, taco_file=None):
        if config is None:
            raise ValueError("SEN2NAIP requires a config object.")

        if isinstance(config, (str, Path)):
            from omegaconf import OmegaConf

            config = OmegaConf.load(config)

        data_cfg = getattr(config, "Data", None)
        if data_cfg is None:
            raise ValueError("SEN2NAIP requires config.Data.")

        if taco_file is None:
            taco_file = getattr(data_cfg, "sen2naip_taco_file", None)
        if phase is None:
            phase = getattr(data_cfg, "sen2naip_phase", "train")
        val_fraction = getattr(data_cfg, "sen2naip_val_fraction", 0.1)
        split_seed = int(getattr(data_cfg, "sen2naip_split_seed", 2025))
        self.crop_size = int(getattr(data_cfg, "sen2naip_crop_size", 0) or 0)
        self.expected_bands = getattr(data_cfg, "sen2naip_num_bands", None)
        if self.expected_bands is not None:
            self.expected_bands = int(self.expected_bands)

        if not taco_file:
            raise ValueError("SEN2NAIP requires config.Data.sen2naip_taco_file.")
        if phase not in {"train", "val"}:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of: train, val.")
        self.phase = phase
        self.split_seed = split_seed
        self.augment = bool(getattr(data_cfg, "sen2naip_augment", False))
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(
                f"val_fraction must be in (0, 1), received {val_fraction}."
            )

        try:
            import tacoreader
        except ImportError as exc:
            raise ImportError(
                "SEN2NAIP requires 'tacoreader'. Install it via "
                "'pip install tacoreader==0.6.5'."
            ) from exc

        self.dataset = tacoreader.load(str(Path(taco_file).expanduser()))
        self.normalizer = Normalizer(config)

        total = len(self.dataset)
        if total < 2:
            raise ValueError(
                "SEN2NAIP dataset requires at least 2 samples to create train/val splits."
            )
        split_file = getattr(data_cfg, "sen2naip_split_file", None)
        if split_file:
            with open(Path(split_file).expanduser(), encoding="utf-8") as handle:
                split = json.load(handle)
            if int(split.get("nrecords", -1)) != total:
                raise ValueError("SEN2NAIP split file does not match manifest record count.")
            split_sets = {key: [int(i) for i in value] for key, value in split.get("indices", {}).items()}
            if not {"train", "val", "test"}.issubset(split_sets):
                raise ValueError("SEN2NAIP split file must contain train, val, and test indices.")
            if any(len(values) != len(set(values)) for values in split_sets.values()):
                raise ValueError("SEN2NAIP split indices must not contain duplicates.")
            all_sets = [set(split_sets[key]) for key in ("train", "val", "test")]
            if len(set.union(*all_sets)) != total or sum(map(len, all_sets)) != total:
                raise ValueError("SEN2NAIP split indices must be disjoint and cover every record.")
            if any(i < 0 or i >= total for indices in all_sets for i in indices):
                raise ValueError("SEN2NAIP split index is outside the manifest.")
            expected_ids = split.get("ids", {})
            if not all(key in expected_ids for key in ("train", "val", "test")):
                raise ValueError("SEN2NAIP split file must contain IDs for all splits.")
            if not hasattr(self.dataset, "iloc"):
                raise ValueError("SEN2NAIP loader cannot validate split IDs for this Taco reader.")
            actual_ids = [str(self.dataset.iloc[i]["tortilla:id"]) for i in range(total)]
            if len(set(actual_ids)) != total:
                raise ValueError("SEN2NAIP manifest contains duplicate IDs.")
            for key, indices in split_sets.items():
                if [actual_ids[i] for i in indices] != [str(value) for value in expected_ids[key]]:
                    raise ValueError(f"SEN2NAIP split IDs do not match indices for {key}.")
            self.indices = split_sets[phase]
        else:
            split_idx = int(round(total * (1.0 - val_fraction)))
            split_idx = min(max(split_idx, 1), total - 1)
            # Shuffle record indices before splitting for a reproducible sample mix.
            split_order = np.random.default_rng(split_seed).permutation(total).tolist()
            self.indices = split_order[:split_idx] if phase == "train" else split_order[split_idx:]

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(data).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, idx):
        rio = sys.modules.get("rasterio")
        if rio is None:
            rio = importlib.import_module("rasterio")

        sample = self.dataset.read(self.indices[idx])
        lr_path = sample.read(0)
        hr_path = sample.read(1)

        with rio.open(lr_path) as src, rio.open(hr_path) as dst:
            src_count = getattr(src, "count", None)
            dst_count = getattr(dst, "count", None)
            if self.expected_bands is not None and src_count is not None and src_count != self.expected_bands:
                raise ValueError(
                    f"SEN2NAIP LR sample has {src.count} bands; expected "
                    f"{self.expected_bands}."
                )
            if src_count is not None and dst_count is not None and src_count != dst_count:
                raise ValueError(
                    f"SEN2NAIP LR/HR band mismatch: {src.count} vs {dst.count}."
                )
            dimensions_available = all(
                hasattr(src, attr) and hasattr(dst, attr)
                for attr in ("height", "width")
            )
            if dimensions_available and (src.height * 4 != dst.height or src.width * 4 != dst.width):
                raise ValueError(
                    "SEN2NAIP sample must have a 4x LR-to-HR spatial ratio; "
                    f"got LR {src.width}x{src.height}, HR {dst.width}x{dst.height}."
                )

            if self.crop_size:
                if not dimensions_available:
                    raise ValueError("Raster metadata is required when using sen2naip_crop_size.")
                if self.crop_size <= 0 or self.crop_size > src.height or self.crop_size > src.width:
                    raise ValueError(
                        f"sen2naip_crop_size must be in [1, {min(src.width, src.height)}]."
                    )
                if self.phase == "train":
                    max_y = src.height - self.crop_size
                    max_x = src.width - self.crop_size
                    # The worker-seeded torch RNG gives fresh crops while
                    # remaining reproducible under a seeded DataLoader.
                    y = int(torch.randint(max_y + 1, (1,)).item())
                    x = int(torch.randint(max_x + 1, (1,)).item())
                else:
                    y = (src.height - self.crop_size) // 2
                    x = (src.width - self.crop_size) // 2
                from rasterio.windows import Window

                lr_window = Window(x, y, self.crop_size, self.crop_size)
                hr_window = Window(x * 4, y * 4, self.crop_size * 4, self.crop_size * 4)
                lr_data = src.read(window=lr_window)
                hr_data = dst.read(window=hr_window)
            else:
                lr_data = src.read()
                hr_data = dst.read()

        lr = self._to_tensor(lr_data)
        hr = self._to_tensor(hr_data)

        if self.phase == "train" and self.augment:
            k = int(torch.randint(4, (1,)).item())
            lr = torch.rot90(lr, k, dims=(-2, -1))
            hr = torch.rot90(hr, k, dims=(-2, -1))
            if bool(torch.randint(2, (1,)).item()):
                lr = torch.flip(lr, dims=(-1,))
                hr = torch.flip(hr, dims=(-1,))

        lr = self.normalizer.normalize(lr)
        hr = self.normalizer.normalize(hr)

        return lr, hr


if __name__ == "__main__":
    ds = SEN2NAIP(
        config="opensr_srgan/configs/config_10m.yaml",
        phase="train",
        taco_file="/data1/datasets/SEN2NAIP/sen2naipv2-crosssensor.taco",
    )
