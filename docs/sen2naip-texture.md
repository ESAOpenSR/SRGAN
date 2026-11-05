# SEN2-SR v1: supported Sentinel-2 recipe

SEN2-SR v1 is the supported four-band ESRGAN checkpoint for Sentinel-2 to
2.5 m super-resolution. It is a preserved continuation of the successful
real-pair run, not a from-scratch retraining recipe. The source was the selected
200k-generator-update synthetic checkpoint, followed by real crosssensor
refinement. The preserved final state stopped at 154,292 generator updates
(308,584 Lightning optimizer steps) after the run plateaued.

The full state is preserved locally at
`logs/sen2naip_texture/real-from200k-48h-20261103/texture/final.ckpt`.
`last.ckpt` is byte-identical. Its SHA-256 is
`3848755551ec5aec00ac820840e1dcdf59e7d847d0a7dec09dd3d2d89b956b6f`.
The matching resolved configuration is
`texture/config.yaml`; the release continuation template is
`opensr_srgan/configs/config_sen2naip_texture.yaml`.

## Public inference preset

The Hugging Face repository `simon-donike/SR-GAN` publishes the `SEN2-SR`
preset as `models/sen2-sr/v1/config.yaml` and
`models/sen2-sr/v1/weights/inference.ckpt`. The latter contains the EMA
generator weights exported from the verified full checkpoint; the inference
config therefore sets `Training.EMA.enabled: false`. The public config keeps
four-band `normalise_10k` input handling and explicit back projection.

The preset pins Hugging Face commit
`b31d2728952cfaa022bd1df92d82a54c23cf4f8f` so config and weights resolve together.

The full resumable checkpoint is
`models/sen2-sr/v1/weights/training.ckpt`. Download the checkpoints, portable
training config, and split with:

```bash
hf download simon-donike/SR-GAN --include 'models/sen2-sr/v1/*' --local-dir ./sen2-model
```

The downloaded training inputs are
`models/sen2-sr/v1/training/config.yaml` and
`models/sen2-sr/v1/training/splits/crosssensor.json`. Keep the split beside the
training config when resuming or reproducing the run.

Install this version from `dev` with
`pip install 'opensr-srgan @ git+https://github.com/ESAOpenSR/SRGAN.git@dev'`.
The public preset downloads the matching config and EMA weights:

```python
from opensr_srgan import load_inference_model

model = load_inference_model("SEN2-SR", map_location="cpu")
prediction = model.predict_step(raw_dn_tensor)
```

For files downloaded manually:

```python
from opensr_srgan import load_from_config

model = load_from_config(
    "sen2-model/models/sen2-sr/v1/config.yaml",
    "sen2-model/models/sen2-sr/v1/weights/inference.ckpt",
    mode="eval",
    map_location="cpu",
)
prediction = model.predict_step(raw_dn_tensor)  # CPU tensor in raw DN units
```

`model(raw_normalized_tensor)` or `model.generator(...)` is the raw forward
path and returns reflectance on the normalized scale, without clipping. For user
inputs in Sentinel-2 DN units, `predict_step` performs normalization, runs the
EMA-exported generator, denormalizes to DN units, and applies back projection to
match the LR measurement. It returns a CPU tensor. Large scenes still require
windowing and stitching through the tiled inference tooling.

## Exact continuation recipe

Use the flat config or the Hydra experiment. Replace all `???` paths with the
local TACO manifest, packaged split manifest, and verified full checkpoint.

```bash
python -m opensr_srgan.train \
  --config sen2-model/models/sen2-sr/v1/training/config.yaml

python -m opensr_srgan.train_hydra experiment=sen2naip_texture \
  Data.sen2naip_taco_file=/data/sen2naipv2-crosssensor.taco \
  Data.sen2naip_split_file=sen2-model/models/sen2-sr/v1/training/splits/crosssensor.json \
  Model.continue_training=sen2-model/models/sen2-sr/v1/weights/training.ckpt
```

`Training.max_steps: 328584` is an absolute optimizer-step target: from the
preserved 308,584-step state it permits 10,000 additional generator updates.
The configuration's nominal optimizer rates are 5e-6 for G and 5e-7 for D,
but a full-state resume restores the scheduler and optimizer state. The saved
checkpoint was actually at 1e-6 (G) and 1e-7 (D) when stopped; those restored
values take precedence. Keep `Model.load_checkpoint: false` for a full resume.

The recipe uses four-band red/green/blue/NIR data, aligned 64×64 to 256×256
crops, spatial split seed 2025, EMA decay 0.99, local-Gram texture weight
0.4685153484, VGG perceptual weight 1.0, L1 0.01, SAM 0.005, GAN 0.00125,
normalized patch R1, global relativistic averages, and ImageNet VGG input
normalization with preactivation features. Validation fixes perceptual RGB to
bands `[0, 1, 2]` while training uses shuffled triplets. The split contains
6,401 training, 800 validation, and 799 reserved test pairs. The split manifest
must remain unchanged for comparable evaluation. The packaged manifest preserves
the run's validation ordering, with its 16 displayed scenes first; spatial
membership is unchanged from the original seed-2025 split. Its SHA-256 is
`218d5d73aec72bbb17a3853606759439f70cb28a1cafe298ee505234b2b21c5d`.

The public model uses the preserved final EMA state from the stopped run;
it is not claimed to be the lowest historical validation metric. The strict
multiband fidelity guards were not all met, so this is not a top-k winner. Full
state checkpoints include live G/D tensors, EMA, Adam, scheduler, loop, and
callback state. The inference-only artifact intentionally retains only the EMA
generator needed for deployment.

CSV logging is the default and W&B is disabled in the release config. Set a new
W&B run identity explicitly when reproducing the continuation; do not reuse the
historical run ID or deadline settings.
