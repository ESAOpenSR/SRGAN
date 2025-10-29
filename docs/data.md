# Data guide

OpenSR GAN Lab welcomes datasets from medical scanners, satellites, drones, microscopes, and consumer cameras. The goal is to
define how low-resolution (LR) inputs and high-resolution (HR) references are paired, normalised, and sampled without editing
code. This guide explains the dataset abstractions, built-in loaders, and ways to extend them.

## Dataset selectors

Every configuration file references a dataset with a simple key:

```yaml
Data:
  dataset: medical_mri
```

This key is resolved by the dataset registry (`opensr_srgan.data.registry`). You can add new selectors via entry points or by
calling `register_dataset` in Python. Built-in options include:

| Key | Use case |
| --- | --- |
| `medical_mri` | Paired MRI volumes with optional slice sampling strategies. |
| `medical_ct` | CT volumes that may require HU windowing before normalisation. |
| `medical_xray` | 2D radiographs stored as PNG, TIFF, or DICOM. |
| `sentinel2` | Multispectral Sentinel-2 SAFE archives (10 m / 20 m). |
| `multispectral_hdf5` | Generic HDF5 container for hyperspectral stacks. |
| `microscopy_zarr` | Large microscopy tiles stored in Zarr arrays. |
| `rgb_folder` | Standard computer-vision datasets organised as HR/LR folder pairs. |
| `video_frames` | Sequential frames for video SR. |
| `custom` | Hook for user-supplied dataset class. |

## Directory conventions

Folder-based datasets assume the following structure unless overridden:

```
root/
├── train/
│   ├── LR/
│   └── HR/
├── val/
│   ├── LR/
│   └── HR/
└── test/
    ├── LR/
    └── HR/
```

If your data lives in another layout, configure `Data.lr_glob`, `Data.hr_glob`, or implement a small adapter that translates
from your format to the `(lr, hr)` pair interface.

## Normalisation statistics

Normalisation is handled by the `Normalisation` block in configs. Statistics can come from:

* **Inline values** – Provide means/stds directly in YAML.
* **External files** – Reference `.yaml`, `.json`, `.npy`, or `.pt` files bundled with your experiment.
* **Dataset scans** – Enable `stats_source: dataset` to compute statistics on the fly (cached for future runs).

For medical imaging, you can specify modality-aware logic (e.g. CT Hounsfield unit clipping) by pointing to a Python function via
`Normalisation.custom_fn`.

## Augmentations

Augmentations live inside the `Data` block. Supported flags include `hflip`, `vflip`, `rotate90`, `elastic`, `random_crop`, and
`noise`. For volumetric data, rotations can be restricted to the axial plane while preserving anatomy.

Advanced users can define a `Data.augmentation_pipeline` that points to a custom Albumentations or MONAI transform sequence.

## Curriculum & sampling

OpenSR GAN Lab supports weighted sampling and curricula. Example:

```yaml
Data:
  curriculum:
    - until_step: 100_000
      crop_size: [96, 96]
    - until_step: 300_000
      crop_size: [128, 128]
```

You can also specify `class_weights`, `patient_weights`, or `tile_weights` for domains where some regions are rarer or more
important.

## Large-scene tiling

For huge rasters or volumes, enable streaming datasets that only load required tiles:

```yaml
Data:
  dataset: microscopy_zarr
  streaming: true
  tile_size: [256, 256]
  stride: [128, 128]
```

Streaming mode works with both training and inference pipelines, preventing memory blow-ups.

## Custom dataset integration

1. **Implement a dataset class** that returns `(lr, hr, metadata)` and registers augmentations inside `__getitem__`.
2. **Register it** using `from opensr_srgan.data.registry import register_dataset`.
3. **Reference the key** in your YAML file.

Example:

```python
from opensr_srgan.data.registry import register_dataset

@register_dataset("histopathology_wsis")
class HistopathologyTiles(Dataset):
    def __init__(self, cfg, split):
        ...
```

## Metadata and evaluation

Datasets can attach metadata (e.g. patient ID, acquisition date, sensor name). The Lightning module logs these attributes and can
use them for stratified validation or domain-specific evaluation metrics.

## Built-in statistics helpers

The repository ships scripts to compute per-channel statistics:

```bash
python -m opensr_srgan.tools.compute_stats --config configs/medical_mri.yaml
```

This command scans the dataset, computes means/stds/percentiles, and writes them to the location specified by
`Normalisation.stats_file`.

---

By describing your dataset in configuration files, you gain repeatable, shareable pipelines across medical, industrial, and
aerial imaging projects.
