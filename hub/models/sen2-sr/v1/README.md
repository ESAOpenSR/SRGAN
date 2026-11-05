---
license: mit
library_name: opensr-srgan
pipeline_tag: image-to-image
datasets:
- tacofoundation/SEN2NAIPv2
tags:
- super-resolution
- sentinel-2
- remote-sensing
- esrgan
---

# Sen2 SR v1: Sentinel-2 RGB/NIR at 4× resolution

`SEN2-SR` is the supported OpenSR SRGAN pretrained model. It promotes the
preserved final checkpoint of run `real48h200k20261103`, intentionally stopped
on November 4, 2026 after plateau review: **154,292 real-data generator
updates** (**308,584 Lightning optimizer steps**) following the selected +200k
synthetic-stage model and real-data pilot.

This is the latest preserved final state, not a top-k metric winner. Strict
multiband fidelity guards were not all satisfied. It is texture-oriented:
generated 2.5 m detail is not an observed measurement.

## Install and infer

```bash
python -m pip install opensr-srgan==1.1.0
```

```python
import torch
from opensr_srgan import load_inference_model

model = load_inference_model("SEN2-SR", map_location="cpu")
# Raw reflectance DN, BCHW, in B04/B03/B02/B08 (red, green, blue, NIR) order.
raw_dn = torch.rand(1, 4, 64, 64) * 10_000
sr_dn = model.predict_step(raw_dn)  # CPU, BCHW, 4× spatial scale
```

The direct `model(...)` interface expects normalized reflectance (`DN / 10000`)
and returns the raw EMA generator output. `predict_step(...)` accepts raw DN
(the recommended interface), restores the input value range, and returns CPU
tensors. It applies packaged measurement-consistency back-projection: 4× block
averaging, blur sigma 0, at most 100 iterations, tolerance `1e-6`. The EMA
generator is already exported in `weights/inference.ckpt`; no runtime EMA swap
or VGG download is needed. Move the model to CUDA before calling it to use a
GPU.

## Release contents

| Path | Purpose |
| --- | --- |
| `weights/inference.ckpt` | Inference-only EMA generator state and provenance |
| `config.yaml` | Matching inference configuration with runtime EMA disabled |
| `weights/training.ckpt` | Original full Lightning checkpoint: live G/D, optimizer moments, schedulers, EMA, callbacks, and loops |
| `training/config.yaml` | Portable full-state continuation configuration |
| `training/splits/crosssensor.json` | Exact spatial split with record IDs |
| `training/provenance/` | Executed configuration, environment, lineage, and preserved source snapshot |
| `metrics/validation.json` | Latest logged real-validation results; not a new test benchmark |
| `verification/` | Promotion checks against the original implementation |
| `manifest.json`, `SHA256SUMS` | Artifact metadata and checksums |

## Resume training

```bash
hf download simon-donike/SR-GAN --include 'models/sen2-sr/v1/*' --local-dir ./sen2-model
hf download simon-donike/SR-GAN examples/resume_training.py --local-dir ./sen2-model
python -m pip install 'opensr-srgan[sen2naip]==1.1.0'
python ./sen2-model/examples/resume_training.py \
  --dataset /path/to/crosssensor.taco --output-dir ./new-run --max-steps 328584
```

The continuation configuration has an absolute target of **328,584 optimizer
steps**, permitting 10,000 further G/D batches from the released checkpoint.
Choose a different target explicitly for a different budget. It clears the old
deadline and W&B identity. The checkpoint restores the actual G/D rates
`1e-6` / `1e-7`; nominal launch rates `5e-6` / `5e-7` in the configuration do
not override restored optimizer state. Dataloader cursor and random augmentation
state were not saved, so resuming mid-epoch cannot replay identical batches.
The example prepares a fresh continuation without starting training. After
inspecting it, run `python -m opensr_srgan.train --config ./new-run/config.yaml`.
Use `--run` only when preparing and starting a fresh output directory together.

## Final-stage recipe

Four-band ESRGAN with 23 RRDBs, 64 feature channels, growth 32, residual scale
0.2, and ICNR 4× upsampling; 3-layer PatchGAN. EMA decay is 0.99. Real
crosssensor training uses batch 12 on one GPU, bf16, aligned random
64×64/256×256 crops, rotations, and horizontal flips. Validation uses fixed
center crops every 250 batches.

Losses: L1 0.01, SAM 0.005, VGG 1.0 (ImageNet normalized, preactivation
conv5_4), local Gram 0.46851534843444825 (grid 4), relativistic-average BCE
beta 0.00125, normalized patch-mean lazy R1 gamma
1.1449030989620114e-6 every 16 batches, and TV 0. Three of four input bands
are shuffled for VGG/local Gram. Validation fixes RGB and separately reports
NIR. `training/config.yaml` contains the full loss, scheduling, and selection
settings.

The spatial split is **6,401 train / 800 validation / 799 reserved test**
pairs. The test set was not used for promotion or checkpoint selection.

## Recorded validation

These are the latest logged full real-validation pass (800 center crops) with
EMA under original bf16 training evaluation. They describe the raw generator
before back-projection; they are not an independent fp32 test-set comparison.

| Metric | Value |
| --- | ---: |
| L1 reflectance | 0.01339648 |
| RGB / NIR SSIM | 0.88137484 / 0.75003165 |
| RGB / NIR PSNR | 36.17022 / 30.34680 |
| SAM | 0.04880648 |
| RGB / NIR VGG distance | 0.99319071 / 1.32213700 |
| Normalized multiband selection | 0.67555183 |

## Scope and license

The model was refined on US Sentinel-2/NAIP crosssensor pairs. Performance
outside those scenes, seasons, processing conventions, and four bands is not
established. Preserve the B04/B03/B02/B08 order and `/10000` scaling. The
4× grid corresponds to 2.5 m for 10 m input; perceptual sharpness does not
establish true ground detail.

This card and the released model artifacts are MIT licensed; see
[LICENSE](LICENSE) and the repository [NOTICE](../../../NOTICE.md). The archived
source snapshot retains its included original license and third-party notices.
