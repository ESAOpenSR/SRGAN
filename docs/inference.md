# Inference

This walkthrough covers the fastest path from zero to an end-to-end super-resolution run on a full Sentinel-2 tile. It uses the published `opensr-srgan` package to grab the pretrained RGB-NIR preset and hands the model to `opensr-utils` for windowed tiling, stitching, and export.

## 1. Install the runtime dependencies

```bash
pip install opensr-srgan
```

* `opensr-srgan` exposes helpers that reconstruct Lightning checkpoints from YAML configs or download ready-to-run presets.
* The optional `huggingface` extra adds `huggingface-hub`, which `load_inference_model` uses internally when fetching preset weights from the Hub.
* `opensr-utils` provides the tiling/mosaicking pipeline that can super-resolve whole Sentinel-2 SAFE folders, GeoTIFFs, or other large rasters. 

## 2.1 Instantiate a Preset 

```python
from opensr_srgan import load_inference_model

model = load_inference_model("RGB-NIR", map_location="cuda")
```

`load_inference_model` retrieves the configuration and checkpoint that correspond to the selected preset (here the four-band RGB-NIR model), restores the Lightning module, and switches it to evaluation mode so that it is ready for inference.【F:opensr_srgan/_factory.py†L107-L150】 If you run on CPU, change `map_location="cuda"` to `map_location="cpu"`.

## 2.2 Instanciate your own Model
```python
from opensr_srgan import load_from_config
model = load_from_config(config_path="YOUR_CONFIG_PATH",
                         checkpoint_uri="YOUR_CKPT_PATH")
```
Using the path to your trained model as well as the config file that was used to train your model, you can load your model for inference.


## 3.1 Run SR on your Images
After the model has been created, please use the pytorch-lightning native `predict_step` function. Sticking to the pytorch-lightning workflow includes the selected normalization procedures that the model was trained on, and also seamlessly enables multi-GPU processing and other tweaks. Be aware that this only SRs raw tensors, so the patching and stitching needs to be done manually.
```python
sr = model.predict_step(lr)
```
## 3.2 Super-resolve a full tile with OpenSR-Utils
You can build on top of out utility package in order to do all of the stitching, georeferencing and patching automatically (Note: this currently only works for RGB-NIR Sentinel-2 images)
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
