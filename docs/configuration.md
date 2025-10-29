# Configuration guide

Everything in OpenSR GAN Lab is controlled by YAML files. Configuration files define the dataset, normalisation pipeline,
architectures, loss weights, schedulers, logging, and runtime options. This page summarises each section so you can design
experiments for any image modality.

## File layout

Configuration files live in `opensr_srgan/configs/`. Copy an existing template and edit it, or create a new file and run
`python -m opensr_srgan.train --config path/to/your.yaml`.

```
Config
├── Project metadata
├── Data
├── Normalisation
├── Model
│   ├── Generator
│   └── Discriminator
├── Training
│   ├── Losses
│   ├── Optimisers & schedulers
│   ├── Stability controls
│   └── Logging
└── Callbacks / export
```

## Project metadata

```yaml
Project:
  name: mri_x4
  seed: 42
  output_dir: runs/mri_x4
  task: super_resolution
```

Use this block to set seeds, output paths, and naming. `task` influences default callbacks (e.g. image logging).

## Data

```yaml
Data:
  dataset: medical_mri
  root: /data/mri
  batch_size: 8
  num_workers: 8
  crop_size: [128, 128]
  scale: 4
  dimensions: 2d  # set to 3d for volumetric pipelines
  augmentations:
    hflip: true
    vflip: true
    rotate90: true
```

Key fields:

* `dataset` – Selector that maps to a dataset class. Built-ins include `medical_mri`, `medical_ct`, `sentinel2`, `rgb_folder`,
  `multispectral_hdf5`, `microscopy_zarr`, and `custom`.
* `dimensions` – `2d` or `3d`. Controls augmentation helpers and whether 3D convolutions are enabled.
* `scale` – Upscaling factor between LR and HR inputs.
* `crop_size` – Spatial size of random crops (list for height/width or depth/height/width).
* `sampler` – Optional weighted sampling or curriculum (e.g. focus on challenging cases first).

## Normalisation

```yaml
Normalisation:
  stats_source: file  # file | inline | dataset
  stats_file: configs/stats/mri_stats.yaml
  per_channel: true
  lr:
    type: zscore
    clip: [-5, 5]
  hr:
    type: zscore
    clip: [-5, 5]
  histogram_match:
    enabled: true
    reference: configs/stats/mri_reference.npy
```

Set how LR and HR samples are scaled and matched. Options include `zscore`, `minmax`, `percentile`, `identity`, and
`custom` (point to a Python function). Histogram matching can be toggled independently for LR/HR.

## Model

```yaml
Model:
  Generator:
    model_type: rrdb
    in_channels: 1
    out_channels: 1
    n_blocks: 23
    n_channels: 64
    growth_channels: 32
    scale: 4
  Discriminator:
    model_type: esrgan
    in_channels: 1
    n_blocks: 7
    base_channels: 64
    feature_matching: true
```

Change `model_type` to switch architectures. Additional parameters depend on the chosen type (e.g. `window_size` for SwinIR
variants, `kernel_size` for LKA). Setting `in_channels` greater than 3 is fully supported.

### Custom modules

Add your own modules by registering them through entry points (`pyproject.toml`) or Python hooks:

```python
from opensr_srgan.model.registry import register_generator

@register_generator("my_custom_generator")
def build_my_generator(cfg):
    ...
```

Then reference `model_type: my_custom_generator` in the YAML.

## Training & losses

```yaml
Training:
  precision: 16
  max_steps: 500_000
  accumulate_grad_batches: 2
  ema:
    enabled: true
    decay: 0.999
  Losses:
    l1_weight: 1.0
    perceptual_weight: 0.2
    perceptual_metric: lpips
    perceptual_channels: [0]  # choose specific bands
    sam_weight: 0.05
    adv_loss_beta: 1e-3
    adv_warmup:
      steps: 20_000
      mode: cosine
  Optimizers:
    generator:
      name: adam
      lr: 2e-4
      betas: [0.9, 0.99]
    discriminator:
      name: adam
      lr: 1e-4
      betas: [0.9, 0.99]
  Schedulers:
    generator:
      warmup_steps: 5_000
      warmup_type: linear
      plateau_patience: 10_000
      factor: 0.5
    discriminator:
      warmup_steps: 0
      plateau_patience: 15_000
      factor: 0.5
  Stability:
    pretrain_g_only: true
    g_pretrain_steps: 100_000
    d_update_interval: 1
    gradient_clip_val: 1.0
```

Highlights:

* `precision` can be `16`, `bf16`, or `32` depending on hardware.
* `ema` toggles exponential moving averages for the generator.
* `Losses` accepts any combination; omit weights to disable terms.
* `perceptual_channels` restricts which channels feed into perceptual losses (useful for hyperspectral or medical scans).
* `adv_warmup` ramps in adversarial pressure gradually.
* `Optimizers` and `Schedulers` are defined per network. You can reference cosine annealing, OneCycle, or custom schedulers.
* `Stability` houses gradient clipping, update cadence, and generator-only pretraining.

## Logging & callbacks

```yaml
Logging:
  logger: wandb  # wandb | tensorboard | csv
  project: opensr-gans
  entity: my-team
  log_images_every_n_steps: 2_000
  save_normalised_images: false
Callbacks:
  checkpoint:
    monitor: val/perceptual
    mode: min
    save_top_k: 3
  tiling_preview:
    enabled: true
    tiles: [[0, 0], [256, 256]]
  export:
    on_save:
      - onnx
      - torchscript
```

Pick a logger, control how often image panels are captured, and configure checkpoint/export behaviour. Additional callbacks
include early stopping, LR logging, and evaluation hooks that run custom metrics.

## Inference-specific options

Inference scripts accept the same config file. Add a block to control tiling and patch overlap:

```yaml
Inference:
  tile_size: [256, 256]
  overlap: [32, 32]
  batch_size: 4
  save_visualisations: true
  export_format: tif
```

These settings ensure large rasters or volumes can be processed piecewise while preserving seams.

---

Armed with these configuration knobs, you can adapt OpenSR GAN Lab to any dataset, architecture, or training philosophy.
