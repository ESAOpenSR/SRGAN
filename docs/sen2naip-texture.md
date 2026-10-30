# SEN2NAIP local-Gram fine-tuning

This recipe promotes the final ESRGAN refinement used by
`real-rgb-localgram-100k-to150k-20261030` on `exp` (`fd11c4b`). It continues a
compatible **full training checkpoint**. It does not reproduce the earlier
reconstruction, texture pilots, or dataset comparisons, and it does not start
from random weights. SwinIR is a separate generator and cannot load this recipe's
ESRGAN checkpoint.

## Launch through the shared trainer

Install the optional dataset reader with `pip install -e '.[sen2naip]'` (or
`pip install 'opensr-srgan[sen2naip]'` for an installed release containing this
recipe). The reader is pinned to `tacoreader==0.6.5`; version 2 uses a different API.

Locate the packaged spatial split from any working directory:

```bash
python -c 'from importlib.resources import files; print(files("opensr_srgan").joinpath("configs/splits/sen2naip_crosssensor.json"))'
```

Supply that path, the real crosssensor TACO manifest, and a full checkpoint:

```bash
python -m opensr_srgan.train_hydra experiment=sen2naip_texture \
  Data.sen2naip_taco_file=/data/sen2naipv2-crosssensor.taco \
  Data.sen2naip_split_file=/path/to/sen2naip_crosssensor.json \
  Model.continue_training=/checkpoints/full-state-100000G.ckpt
```

`srgan-train` accepts the same overrides. Inspect the composition without training
with `--cfg job`. Alternatively, copy the packaged
`configs/config_sen2naip_texture.yaml`, replace its three `???` paths, and run
`python -m opensr_srgan.train --config /path/to/recipe.yaml`.
Both interfaces call the same trainer. Hydra uses the existing timestamped run
folder without changing the working directory; the flat YAML uses the normal
trainer output-directory rules. Use a fresh output folder for each continuation.

`Model.continue_training` restores live G/D weights, EMA, optimizers, schedulers,
callback state, and loop counters. Leave `Model.load_checkpoint: false`:
weights-only initialization would reset the successful training history. The
source checkpoint remains unchanged. There is no automatic checkpoint search or
long-running job launch when installing or inspecting the preset.

## Settings and budget

| Component | Promoted settings |
| --- | --- |
| Generator | Four-band ESRGAN, 23 RRDBs, 64 channels, growth 32, residual scale 0.2, 4× ICNR upsampling |
| Discriminator | Three-layer PatchGAN, relativistic-average BCE |
| Loss | L1 0.01; SAM 0.005; RGB VGG 1.0; local Gram 1.874061393737793; GAN 0.005; TV 0 |
| VGG | ImageNet input normalization; preactivation conv5_4 perceptual features; conv2_2/conv3_4 local Gram features; grid 4 |
| Optimizers | G/D 1e-5 / 1e-6; Adam betas 0.9 / 0.99; gradient clip 1.0 |
| EMA | Decay 0.99, with restored update history |
| Data | Four bands, division by 10000, aligned 64×64 / 256×256 crops, batch 6 per GPU |
| Runtime | Two GPUs, bf16 mixed precision, validation every 250 training batches |

`Training.max_steps: 300000` is an **absolute** Lightning optimizer-step target,
not 300,000 additional steps. GAN batches update D and G once each. Continuing
from 100,000 G updates (200,000 optimizer steps) therefore allows 50,000 more G
updates, ending at 150,000 total. Milestones preserve full state every 5,000 G
updates between 105,000 and 150,000. Raise the absolute target explicitly when
continuing a checkpoint that has already reached it.

D/G learning-rate coupling remains 0.1 through warmup, G plateau changes, and
resume. The independent D plateau scheduler is disabled. G monitors raw
validation perceptual loss with an absolute threshold of 0.001, after the
adversarial ramp. Existing configs keep their historical VGG input behavior
unless `Training.Losses.vgg_input_normalization: true` is set; this recipe enables
it explicitly, together with `vgg_before_activation: true`.

Selection history is retained when resuming into a fresh output directory.
Historical best-checkpoint paths remain read-only references in their original
directory.

Top-k selection monitors `perceptual + 0.01 * L1`. Checkpoints become eligible
after the adversarial ramp and only when globally averaged validation L1 is at
most 0.05. Latest/final full-state saves run independently of top-k improvement;
`last.ckpt` follows validation and both `last.ckpt` and `final.ckpt` are saved at
normal training end. Full-state restoration retains scheduler, warmup, optimizer,
and EMA history; configuration LR values do not reset restored optimizers.

## Data and provenance

Use the real Sentinel-2/NAIP crosssensor variant, with bands ordered red, green,
blue, NIR. The packaged split assigns centroid grid cells of 0.25 degrees using
seed 2025: **6,401 train, 800 validation, 799 reserved test** pairs. Its exact
record indices and IDs are validated against the manifest. Training uses fresh
paired crops, rotations, and flips; validation uses fixed center crops. The test
indices are excluded from both training and validation. No synthetic variants or
experiment orchestration scripts are needed.

The recipe's RNG seed is 20261028. Its configured fallback split seed does not
replace the mandatory spatial manifest. Retain that manifest for comparable
validation; changing it defines a different experiment.

- Source config SHA-256: `37786d8a5438030d6617c2b4c788cf4996eace793a60545dea6a756b7a4a3225`.
- Packaged split SHA-256: `d2261258df5af4cca513f7f0f5172de479bd7d7e2a1c2b60ad41b34584b369d6`.
- Preserved 100k-G source checkpoint SHA-256: `6ccd0a35cfbf6461b614e61603dfc03eee71dad50a33f329df8b70e7f86888c9`.

The promoted config changes machine paths, output location, logging identity, and
explicit compatibility settings. It removes the disabled experimental convergence
block. Loss weights, model, optimizer, scheduler, validation, and milestone
settings otherwise match the saved run. The calibrated Gram weight belongs to
this RGB feature setup and is not a universal default for other data or backbones.

## Logging and prediction matching

CSV is the default. Local validation RGB/NIR panels retain EMA/live comparisons
and add matched SR, with a common display range. To enable W&B, set
`Logging.wandb.enabled=true` and your entity/project. No experiment run ID is
reused; optional `name`, `group`, `id`, and `resume` settings remain available for
an intentional continuation of your own logging run.

This recipe selects `postprocessing=back_projection`. Override with
`postprocessing=histogram` if desired. Matching affects prediction and comparison
figures; losses, validation metrics, and model selection use raw output. Older
recipes default to histogram matching. See [prediction matching](inference.md#prediction-matching)
for the value-range and correction settings.

Inspect scene structure and RGB/NIR texture alongside validation metrics. The
recipe records promising experimental settings, not a guarantee of better
performance; test data must remain outside checkpoint selection.
