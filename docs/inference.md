# Inference & deployment

Once your generator is trained, OpenSR GAN Lab offers several pathways to apply it—from research notebooks to clinical pipelines
and large-scale tiling services. This guide covers the built-in inference CLI, programmatic usage, and export options.

## Command-line inference

```bash
python -m opensr_srgan.inference \
  --config configs/mri_x4.yaml \
  --checkpoint runs/mri_x4/checkpoints/ema.ckpt \
  --input /data/mri/lr \
  --output outputs/mri_sr
```

Key flags:

* `--devices` – Force CPU or select specific GPUs.
* `--batch-size` – Override the batch size for inference.
* `--save-metadata` – Persist dataset metadata alongside outputs.
* `--tiling` – Enable patch-based inference even if the config does not define it.

The CLI loads the config, applies normalisation, performs tiling if required, and writes denormalised outputs to the destination
folder.

## Tiled inference for large scenes

Many domains involve huge rasters, volumes, or gigapixel slides. Configure tiling in your YAML or via CLI flags:

```yaml
Inference:
  tile_size: [256, 256]
  overlap: [32, 32]
  blend: hann
  pad_mode: reflect
```

* `tile_size` – Spatial dimensions of the inference window (depth × height × width for 3D).
* `overlap` – Overlapping pixels to smooth seams.
* `blend` – Optional windowing function (`hann`, `linear`, `none`).
* `pad_mode` – How to pad border tiles (`reflect`, `replicate`, `zero`).

Tiling works for both 2D and 3D data and streams patches to keep memory usage predictable.

## Programmatic usage

```python
from opensr_srgan.model.module import OpenSRLightningModule
from opensr_srgan.config import load_config

cfg = load_config("configs/mri_x4.yaml")
module = OpenSRLightningModule.from_config(cfg, checkpoint="runs/mri_x4/checkpoints/ema.ckpt")
module.eval()

sr = module.predict_batch(lr_tensor)  # Accepts NCHW/NCDHW
```

Use `module.normalizer` to apply the same scaling logic to new inputs, and `module.denormalizer` to bring outputs back to the
original range.

## Batch deployment

For production workloads:

* **Lightning inference loops** – Wrap the module in a Lightning `Trainer` with `trainer.predict` for distributed or batched
  inference.
* **ONNX/TorchScript exports** – Add `Callbacks.export` entries in your config to create export artefacts whenever a checkpoint is
  saved.
* **Hugging Face Hub** – Use `opensr_srgan.tools.push_to_hub` to share weights publicly or within your organisation.

## Quality assurance

* **EMA vs. raw weights** – EMA checkpoints often produce cleaner results. Switch between them when evaluating.
* **Domain shifts** – When applying a model to new scanners/sensors, re-compute normalisation stats and consider fine-tuning with
  a few samples.
* **Perceptual validation** – Use the validation scripts to compute LPIPS, SSIM, or domain-specific metrics on held-out data.

## Example: microscopy deployment

1. Train a 3D model with `dimensions: 3d` and tiling enabled.
2. Export to TorchScript for integration into a microscopy workstation.
3. Deploy the script in a watcher service that picks up new LR volumes, runs tiled inference, and saves outputs alongside metadata
   for downstream analysis.

## Example: clinical review

1. Train an MRI super-res model with perceptual losses restricted to certain contrasts.
2. Use the inference CLI with `--save-metadata` to keep patient IDs and acquisition parameters.
3. Load results into a PACS viewer or Jupyter notebook for radiologist assessment.

---

With these tools, you can transition from research experiments to dependable SR services spanning healthcare, geospatial,
microscopy, and consumer applications.
