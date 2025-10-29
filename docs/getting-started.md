# Getting started

Whether you are prototyping on a laptop or orchestrating large multi-GPU jobs, this guide walks you through installing OpenSR GAN
Lab and running your first experiment.

## Installation

### 1. Choose your environment

OpenSR GAN Lab targets Python 3.10–3.12 and PyTorch Lightning 1.9–2.x. Decide if you want a lightweight inference environment or
full training stack.

### 2. Install from PyPI (inference & quick experiments)

```bash
python -m pip install opensr-srgan
```

This installs the package with minimal dependencies. Use it to load checkpoints, run inference scripts, or fine-tune existing
models.

### 3. Install from source (training & development)

```bash
git clone https://github.com/simon-donike/SISR-RS-SRGAN.git
cd SISR-RS-SRGAN
python -m pip install -r requirements.txt
pre-commit install
```

Optional extras:

* `pip install monai nibabel` for medical-imaging data loaders.
* `pip install rasterio satpy` for geospatial rasters.
* `pip install zarr dask` for large microscopy datasets.

### 4. Verify the installation

```bash
python -c "import opensr_srgan; print(opensr_srgan.__version__)"
```

## First configuration

Configuration lives in `opensr_srgan/configs/`. Start by copying a template:

```bash
cp opensr_srgan/configs/examples/mri_x4.yaml my_experiment.yaml
```

Open `my_experiment.yaml` and update:

* `Data.root` to point to your dataset directory.
* `Data.dataset` if you use a different selector.
* `Normalisation.stats_file` to reference your statistics.
* `Model.Generator.in_channels`/`out_channels` to match your modality.

## Running training

```bash
python -m opensr_srgan.train --config my_experiment.yaml
```

Useful flags:

* `--resume` – Continue training from the last checkpoint.
* `--devices` – Select GPUs or set to `cpu` for quick dry runs.
* `--strategy` – Choose Lightning strategies (e.g. `ddp`, `fsdp`).
* `--precision` – Override precision (`16`, `bf16`, `32`).

Training outputs logs, checkpoints, and validation panels into `Project.output_dir`.

## Monitoring progress

By default, the trainer logs to Weights & Biases when credentials are available. Otherwise, it falls back to TensorBoard or CSV
depending on configuration. Expect:

* Scalar plots for each loss component.
* Validation image grids for LR/HR/SR comparisons.
* Histograms of pixel distributions and discriminator logits.

## Running inference

Use the same config file to run inference once you have a checkpoint:

```bash
python -m opensr_srgan.inference \
  --config my_experiment.yaml \
  --checkpoint runs/mri_x4/checkpoints/ema.ckpt \
  --input data/lr_samples \
  --output outputs/sr_results
```

Enable tiled inference for large scenes by defining `Inference.tile_size` and `Inference.overlap` in the config.

## Troubleshooting

* **CUDA mismatch** – Install the PyTorch wheel that matches your driver (see [pytorch.org](https://pytorch.org/get-started/)).
* **Missing dependencies** – Some modality-specific loaders (DICOM, SAFE, Zarr) require optional packages; install the
  recommended extras listed above.
* **Convergence issues** – Refer to the [training guideline](training-guideline.md) for tips on warm-ups, ramps, and loss tuning.
* **Normalisation drift** – Double-check statistics files and ensure LR/HR branches share compatible scaling.

## Next steps

* Explore the [Configuration](configuration.md) reference to fine-tune settings.
* Read the [Training](training.md) chapter for optimisation strategies.
* Learn about [Inference](inference.md) to deploy your models.
* Share your configs and findings with the community via issues or pull requests.
