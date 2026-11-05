---
license: mit
library_name: opensr-srgan
pipeline_tag: image-to-image
tags:
- super-resolution
- remote-sensing
- legacy
---

# RGB-NIR v1 (legacy preset)

`RGB-NIR` is a maintained legacy 4-band, 4× OpenSR SRGAN inference preset.
It expects Sentinel-2 bands in `B04/B03/B02/B08` (red, green, blue, NIR)
order. The published architecture is SRResNet with RRDB blocks: 24 blocks,
64 feature channels, growth 32, and residual scale 0.2. Inputs use the
historical `normalise_10k` convention: raw reflectance DN divided by 10,000.

```bash
python -m pip install 'opensr-srgan @ git+https://github.com/ESAOpenSR/SRGAN.git@dev'
```

```python
from opensr_srgan import load_inference_model

model = load_inference_model("RGB-NIR", map_location="cpu")
sr_dn = model.predict_step(raw_dn_tensor)  # BCHW raw DN; returns CPU tensors
```

`predict_step` detects `normalise_10k` inputs and returns the corresponding
range on CPU. The output scale is defined by the packaged configuration (4×),
so callers should derive output shape from the returned tensor rather than
hard-coding scene dimensions. The historical configuration does not specify a
postprocessing block, so its default is histogram matching.

The available artifact is `weights/inference.ckpt` with `config.yaml`. It is a
legacy raw generator state dictionary (733 tensors, approximately 47 MiB), not
a full resumable training checkpoint. Training provenance and validation
metrics were not retained with this artifact. Its input-band order is defined
by the historical runtime integration, rather than recorded training
provenance. Do not assume the Sen2 SR dataset or provenance applies here.

See [the shared tensor and GeoTIFF examples](../../../examples/README.md).
This card and its released artifact are MIT licensed; see [LICENSE](LICENSE).
