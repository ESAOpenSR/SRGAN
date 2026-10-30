# Inference

This walkthrough covers the fastest path from zero to an end-to-end super-resolution run on a full Sentinel-2 tile. It uses the published `opensr-srgan` package to grab the pretrained RGB-NIR preset and hands the model to `opensr-utils` for windowed tiling, stitching, and export.

## 1. Install the runtime dependencies

```bash
pip install opensr-srgan
pip install opensr-utils  # optional, for full-tile inference
```

* `opensr-srgan` exposes helpers that reconstruct Lightning checkpoints from YAML configs or download ready-to-run presets.
* `huggingface-hub` is installed with `opensr-srgan` and is used by `load_inference_model` when fetching preset weights from the Hub.
* Install `opensr-utils` separately when you want the large-raster tiling/mosaicking pipeline for Sentinel-2 SAFE folders, GeoTIFFs, or other large rasters.

## 2.1 Instantiate a preset

```python
from opensr_srgan import load_inference_model

model = load_inference_model("RGB-NIR", map_location="cuda")
```

`load_inference_model` retrieves the configuration and checkpoint that correspond to the selected preset (here the four-band RGB-NIR model), restores the Lightning module, and switches it to evaluation mode so that it is ready for inference. If you run on CPU, change `map_location="cuda"` to `map_location="cpu"`.

## 2.2 Instantiate your own model

```python
from opensr_srgan import load_from_config
model = load_from_config(config_path="YOUR_CONFIG_PATH",
                         checkpoint_uri="YOUR_CKPT_PATH")
```

Using the path to your trained model as well as the config file that was used to train your model, you can load your model for inference.


## 3.1 Run SR on tensors

After the model has been created, use `predict_step` for tensor inference. This applies the normalization procedure configured for the model and returns a CPU tensor. It only processes raw tensors, so patching and stitching must be handled separately for large rasters.

```python
sr = model.predict_step(lr)
```

## Prediction matching

`predict_step` uses histogram matching by default, including for older configs
without a `Postprocessing` section. Set the method explicitly in YAML:

```yaml
Postprocessing:
  method: back_projection  # or histogram
  back_projection:
    blur_sigma: 0.0
    max_iterations: 100
    tolerance: 1.0e-6
    relaxation: 1.0
```

Hydra also supports `postprocessing=histogram` and
`postprocessing=back_projection`. Unknown methods raise an error.
The matching method is independent of the generator, so these settings apply
to CNN and SwinIR models alike.

| Method | What it matches | Value range used |
| --- | --- | --- |
| `histogram` | Per-band intensity distributions of SR and the resized LR reference. | Normalized generator output, before denormalization. |
| `back_projection` | LR measurements produced by the configured blur/downsample operator. | Denormalized SR and the original supplied LR values. |

To change the method for an already loaded model:

```python
# model was loaded from a config and checkpoint and is in eval mode.
from omegaconf import OmegaConf

model.config.Postprocessing = OmegaConf.create({
    "method": "back_projection",
    "back_projection": {
        "blur_sigma": 0.0,
        "max_iterations": 100,
        "tolerance": 1e-6,
        "relaxation": 1.0,
    },
})
sr = model.predict_step(lr)

# Restore the historical method without changing model weights.
model.config.Postprocessing.method = "histogram"
sr_histogram = model.predict_step(lr)
```

For SwinIR, load your own matching SwinIR config and trained checkpoint with
`load_from_config(..., mode="eval")`. `load_inference_model` accepts the
published `RGB-NIR` and `SWIR` presets; it does not select SwinIR by name.
See [SwinIR reconstruction training](training.md#train-a-swinir-reconstruction-model).

Histogram matching keeps the historical sequence: normalize, generate, match,
then denormalize. Back-projection corrects the denormalized prediction against
the original LR measurements. With zero blur it matches block means while
preserving within-block differences; optional scalar/per-band Gaussian sigma
is measured in HR pixels and uses periodic boundaries. Scale is inferred.

Back-projection runs without gradients, promotes half precision to float32,
and does not clip after correction. Tolerance is LR RMSE in supplied units.
Zero LR pixels are valid measurements. The standalone
`opensr_srgan.utils.back_projection.back_projection` function also accepts
boolean LR `valid_mask`, HR `output_mask`, and `return_info=True` diagnostics.
The Gaussian operator is an optional periodic blur model, not a calibrated
Sentinel-2 sensor response. Its implementation retains the upstream OpenSR attribution.

The standalone helper also works on existing SR tensors without a model:

```python
from opensr_srgan.utils.back_projection import BackProjectionConfig, back_projection

# raw_sr and lr are floating-point CHW or BCHW tensors, in the same units,
# on the same device; raw_sr has four times the spatial resolution.
matched, info = back_projection(
    raw_sr, lr,
    BackProjectionConfig(scale=4, blur_sigma=0.0),
    return_info=True,
)
# info["converged"] and info["iterations"] report progress per image.
```

Use the [configuration reference](configuration.md#post-processing-postprocessing)
for all settings and the [utility API](api-reference/utils.md#back-projection)
for masks, dtype handling, and diagnostics.

`model.postprocess(sr, lr)` applies the configured method to tensors in a shared
value range without normalization. `forward`, training losses, validation metrics,
and discriminator evaluation use raw generator predictions. Validation panels
show raw and matched SR using a common display range, including alongside
optional EMA/live comparisons.

## 3.2 Super-resolve a full tile with OpenSR-Utils

You can build on top of the `opensr-utils` package to handle stitching, georeferencing, and patching automatically. This currently targets RGB-NIR Sentinel-2 inference.

```python
import opensr_utils

sen2_path = "opensr_srgan/data/S2A_MSIL2A_20230901T104031_N0509_R137_T31TFJ_20230901T130204.SAFE"
sr_runner = opensr_utils.large_file_processing(
    root=sen2_path,
    model=model,
    window_size=(128, 128),
    factor=4,
    overlap=12,
    eliminate_border_px=2,
    device="cuda",
    gpus=[0],
    save_preview=True,
    debug=False,
)
sr_runner.start_super_resolution()
```

`large_file_processing` orchestrates the windowed inference workflow: it slides a `(128 × 128)` LR window over the scene, feeds each crop through the SRGAN, blends overlapping predictions (12 px overlap with 2 px border trimming), and optionally stores both previews and georeferenced outputs. The helper understands either directory-style SAFE products or single GeoTIFFs, and it accepts GPU IDs for accelerated execution. Adjust the paths, window size, and overlap to match your dataset or hardware constraints.
