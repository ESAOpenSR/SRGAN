---
license: mit
library_name: opensr-srgan
pipeline_tag: image-to-image
tags:
- super-resolution
- remote-sensing
- sentinel-2
- esrgan
---

# OpenSR SRGAN pretrained models

This repository packages the supported Sen2 SR release and two maintained
legacy presets for the OpenSR SRGAN `dev` package.

| Preset | Model card | Input / scale | Checkpoints |
| --- | --- | --- | --- |
| `SEN2-SR` | [Sen2 SR v1](models/sen2-sr/v1/README.md) | B04/B03/B02/B08, 4× | [inference](models/sen2-sr/v1/weights/inference.ckpt), [training](models/sen2-sr/v1/weights/training.ckpt) |
| `RGB-NIR` | [RGB-NIR v1](models/rgb-nir/v1/README.md) | 4 bands, 4× | [inference](models/rgb-nir/v1/weights/inference.ckpt) |
| `SWIR` | [SWIR v1](models/swir/v1/README.md) | 6 bands, 8× | [inference](models/swir/v1/weights/inference.ckpt) |

Install the package from the supported branch and use a preset:

```bash
python -m pip install 'opensr-srgan @ git+https://github.com/ESAOpenSR/SRGAN.git@dev'
```

```python
from opensr_srgan import load_inference_model

model = load_inference_model("SEN2-SR", map_location="cpu")
sr_dn = model.predict_step(raw_dn_tensor)  # BCHW raw DN; returns CPU tensors
```

See [examples](examples/README.md) for tensor, GeoTIFF, and resume-training
entry points. Sample artifacts are in [data/examples](data/examples/README.md).

## Repository layout

```text
models/
  sen2-sr/v1/
    config.yaml
    weights/{inference.ckpt,training.ckpt}
    training/{config.yaml,splits/crosssensor.json}
    training/provenance/{environment.txt,executed_training.yaml,lineage.json,source_at_stop.tar.gz}
    metrics/validation.json
    verification/{inference.json,resume_original.json,resume_promotion.json}
  rgb-nir/v1/{config.yaml,weights/inference.ckpt}
  swir/v1/{config.yaml,weights/inference.ckpt}
data/examples/{rgb-nir_batch.pt,dataset.zip}
examples/{infer_tensor.py,infer_geotiff.py,resume_training.py,README.md}
```

Each model directory includes a `manifest.json` and `SHA256SUMS` for release
integrity. The former root-level artifact names remain compatibility aliases
for existing downloads; new integrations and documentation use the canonical
paths above.

Model weights, model cards, and authored examples are released under the
[MIT License](LICENSE). See [NOTICE.md](NOTICE.md) for the scope of that
license and the treatment of archived source and sample data.
