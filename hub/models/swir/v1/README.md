---
license: mit
library_name: opensr-srgan
pipeline_tag: image-to-image
tags:
- super-resolution
- remote-sensing
- legacy
---

# SWIR v1 (legacy preset)

`SWIR` is a maintained legacy 6-band, 8× OpenSR SRGAN inference preset. Its
published architecture uses 32 RCAB blocks with 96 feature channels. It expects
20 m Sentinel-2 bands in `B05/B06/B07/B8A/B11/B12` order and produces an 8×
grid (2.5 m from 20 m input). Inputs use the historical `normalise_10k`
convention: raw reflectance DN divided by 10,000.

```bash
python -m pip install opensr-srgan==1.1.0
```

```python
from opensr_srgan import load_inference_model

model = load_inference_model("SWIR", map_location="cpu")
sr_dn = model.predict_step(raw_dn_tensor)  # BCHW raw DN; returns CPU tensors
```

`predict_step` detects `normalise_10k` inputs and returns the corresponding
range on CPU. The output scale comes from the packaged configuration (8×), so
callers should derive output shape from the returned tensor rather than
hard-coding scene dimensions. The historical configuration does not specify a
postprocessing block, so its default is histogram matching.

The available artifact is `weights/inference.ckpt` with `config.yaml`. It
preserves the original legacy Lightning wrapper: a 304-tensor generator state
dictionary plus two optimizer payloads, three schedulers, epoch 155, and step
80,184. It is not a complete resumable training model. Training provenance and
validation metrics were not retained with this artifact. Do not assume the Sen2
SR dataset or provenance applies here.

See [the shared tensor and GeoTIFF examples](../../../examples/README.md).
This card and its released artifact are MIT licensed; see [LICENSE](LICENSE).
