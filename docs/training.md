# Training workflow

`opensr_srgan/train.py` is the canonical entry point for ESA OpenSR experiments. It ties together configuration loading, model instantiation, dataset selection, logging, and callbacks. This page explains how the script is organised and how to customise the training loop.

!!! tip "Start with a published recipe"
    [Sen2 SR v1](sen2-sr.md) demonstrates ESRGAN/PatchGAN training with spectral,
    perceptual, local-Gram texture, and relativistic adversarial losses. Its
    [training YAML and full checkpoint](sen2-sr.md#checkpoints-and-release-artifacts)
    include the exact spatial split and a [full-state continuation example](sen2-sr.md#continue-training).

!!! note "PyTorch Lightning 2+ only"
    The training stack uses a single manual-optimisation path. `SRGAN_model.setup_lightning()` enforces Lightning >= 2.0 and binds `training_step_PL2()` where GAN training runs with `automatic_optimization = False`. `opensr_srgan.utils.build_trainer_kwargs.build_lightning_kwargs()` forwards resume checkpoints through `Trainer.fit(..., ckpt_path=...)`. See [Trainer Details](trainer-details.md) for a step-by-step breakdown of warm-up checks, adversarial updates, and EMA lifecycle.

This section is a more technical overview; [Training Guideline](training-guideline.md) provides a broader overview of how to monitor the training process.

## Data module construction
In order to train, you need a dataset. `Data.dataset_type` decides which dataset to use and wraps them in a `LightningDataModule`. Should you implement your own, you will need to add it to the dataset_selector.py file with the settings of your choice (see [Data](data.md)). Optionally, the selector instantiates `ExampleDataset` by default—perfect for smoke tests after downloading the sample data, a dataset of 200 RGB-NIR image pairs. The module inherits batch sizes, worker counts, and prefetching parameters from the configuration and prints a summary including dataset size.



## Command-line and Python interfaces

You can launch training from the CLI or by importing the helper inside Python.

```bash
python -m opensr_srgan.train --config path/to/config.yaml
```

```python
from opensr_srgan import train

train("path/to/config.yaml")
```

For experiment management, the Hydra entry point composes grouped configs from `opensr_srgan/configs/hydra` and forwards the resolved config to the same `train()` function:

```bash
python -m opensr_srgan.train_hydra experiment=example
python -m opensr_srgan.train_hydra experiment=10m Training.max_epochs=5 Logging.wandb.enabled=false
srgan-train experiment=20m Training.gpus=[0,1]
```

Use `experiment=...` when you want a complete preset (`example`, `10m`, or `20m`). Use direct overrides such as `Training.max_epochs=5` or `Generator.n_blocks=8` for quick one-off changes. See [Hydra Experiments](hydra.md) for the config layout, override patterns, and output-directory behavior.

Both legacy entry points accept the same configuration file. The legacy CLI exposes a single optional argument:

* `--config / -c`: Path to a YAML file describing the experiment. Defaults to `opensr_srgan/configs/config_10m.yaml`. For a local smoke test with the bundled dataset, pass `opensr_srgan/configs/config_training_example.yaml`.

GPU assignment is handled directly in the configuration. Set `Training.gpus` to a list of device indices (for example `[0, 1, 2, 3]`) to enable multi-GPU training; a single value such as `[0]` keeps the run on one card. When more than one GPU is listed the trainer activates PyTorch Lightning DDP using `ddp_find_unused_parameters_true` by default, which fits the manual GAN update pattern.

## Train a SwinIR model

Set `Generator.model_type: swinir` or select `model=10m_swinir`, then combine it
with compatible data and training settings. SwinIR checkpoints must match this
implementation; switching generators does not convert weights. Prediction
matching remains independent of the reconstruction objective. See
[Architecture](architecture.md#swinir-generator) for the generator settings.

## Initialisation steps - Overview
The code performs the following, regardless of whether the script is launched from the CLI or via import.
1. **Import dependencies.** Torch, PyTorch Lightning, OmegaConf, and logging backends are loaded up-front.
2. **Parse arguments.** `argparse` reads the configuration path and ensures the file exists.
3. **Load configuration.** `OmegaConf.load()` parses the YAML file into an object used throughout the run.
4. **Construct the model.**
   * If `Model.load_checkpoint` is set, the script calls `model.load_weights_from_checkpoint()` to import learned weights only while
     respecting the new configuration values. If `Model.continue_training` is passed with a path to a Lightning checkpoint, optimizer/scheduler state, epoch counters, global step, and EMA state are restored by Lightning before continuing the run.
   * `Model.load_checkpoint` and `Model.continue_training` are mutually exclusive. Use only one, depending on whether you want weight initialization or full training-state resume.
    * Otherwise, it initialises a fresh `SRGAN_model`, which immediately builds the generator/discriminator and prints a
      parameter summary.
5. **Launch Training.** The training is launched with the model, weights and settings passed in the config.


## Logging setup

* **Weights & Biases.** `WandbLogger` records scalar metrics, adversarial diagnostics, and validation image panels.
* **CSV logs.** When `Logging.wandb.enabled: false`, `CSVLogger` writes lightweight local logs under `logs/` or under `Logging.output_dir` for Hydra runs.

To disable W&B logging, set `Logging.wandb.enabled: false` in YAML or override it from Hydra with `Logging.wandb.enabled=false`.

## Metrics

The Lightning module logs scalar metrics through the active Lightning logger, either W&B or CSV. Generator-only pretraining, adversarial training, and the EMA helper each contribute their own indicators, so the
logs make it clear which subsystem is active at any given step.

| Metric | Description |
| --- | --- |
| `training/pretrain_phase` | Flag indicating whether generator-only warm-up is active. |
| `discriminator/adversarial_loss` | Discriminator loss on real and generated batches. |
| `discriminator/D(y)_prob` | Mean discriminator confidence for HR inputs. |
| `discriminator/D(G(x))_prob` | Mean discriminator confidence for generated inputs. |
| `train_metrics/l1` / `val_metrics/l1` | Mean absolute error between SR and HR tensors. |
| `train_metrics/sam` / `val_metrics/sam` | Spectral angle mapper averaged over pixels. |
| `train_metrics/perceptual` / `val_metrics/perceptual` | Perceptual distance from VGG or LPIPS features. |
| `train_metrics/tv` / `val_metrics/tv` | Total-variation regularization term. |
| `train_metrics/psnr` / `val_metrics/psnr` | Peak signal-to-noise ratio. |
| `train_metrics/ssim` / `val_metrics/ssim` | Structural similarity index. |
| `generator/content_loss` | Configured weighted reconstruction objective. |
| `generator/total_loss` | Content and adversarial terms used for the generator update. |
| `validation/DISC_adversarial_loss` | Discriminator loss on validation batches. |
| `training/adv_loss_weight` | Current adversarial weight applied to the generator loss. |
| `lr_discriminator` / `lr_generator` | Optimizer learning rates. |
| `EMA/enabled`, `EMA/decay`, `EMA/update_after_step` | EMA configuration values. |
| `EMA/use_num_updates`, `EMA/is_active`, `EMA/steps_until_activation` | EMA update state. |
| `EMA/last_decay`, `EMA/num_updates` | EMA update diagnostics. |

## Callbacks

The following callbacks are registered with the Lightning trainer:

| Callback | Purpose |
| --- | --- |
| `ModelCheckpoint` | Saves the top checkpoints according to the selected monitor and keeps the latest checkpoint. |
| `EarlyStopping` | Defaults to `Schedulers.metric_g`, 250 validation checks, and finite-check enabled; monitor, mode, and patience are configurable. |

Checkpoint directories default to `logs/<project>/<timestamp>`. Hydra runs set `Logging.output_dir`, so checkpoints and the resolved `config.yaml` land in Hydra's run directory.

## Trainer configuration

The script builds a `Trainer` with the following notable arguments:

* `accelerator='gpu'` with `devices=config.Training.gpus` for CUDA/GPU runs, or `accelerator='cpu'` with `devices=1` for CPU runs. When more than one GPU is requested, the script selects `ddp_find_unused_parameters_true` by default because manual GAN optimisation can leave one branch unused on a given step.
* `limit_val_batches=config.Training.limit_val_batches` controls validation work. Optional `precision`, `max_steps`, training-batch limits, and sanity-validation limits are forwarded when configured.
* `logger=[wandb_logger]` to register the active W&B or CSV logger.
* `callbacks` contains checkpointing and finite-check early stopping. Learning rates are logged by the Lightning module at the end of each training batch.

Finally, `trainer.fit(model, datamodule=pl_datamodule)` launches the optimisation loop and `wandb.finish()` ensures clean shutdown
of the W&B session.

For an opt-in check of distributed training, see the [two-GPU DDP smoke test](tests.md#optional-two-gpu-ddp-smoke-test).

## Generator EMA lifecycle

If `Training.EMA.enabled` is `True`, the Lightning module keeps a shadow copy of the generator weights using the decay set in
`Training.EMA.decay`. The EMA state:

* updates immediately after each generator optimiser step once `Training.EMA.update_after_step` has been reached,
* lives on the device requested via `Training.EMA.device` (falling back to the generator's device), and
* automatically swaps in for evaluation, testing, and inference before being restored for continued training.

Checkpoints store both the live and EMA weights, so resuming training preserves the smoothed model.

## Practical tips

* **Gradient stability.** Tune `Training.pretrain_g_only`, `g_pretrain_steps`, and `adv_loss_ramp_steps` when experimenting with
  new generator architectures. Longer warm-ups often help deeper networks converge.
* **Learning-rate warmup.** `Schedulers.g_warmup_steps` and `Schedulers.g_warmup_type` apply a step-wise warmup (cosine or linear)
  to the generator LR before handing control back to the plateau scheduler. Start with 1–5k steps to avoid shocking freshly
  initialised weights.
* **Checkpoint hygiene.** Periodically prune the timestamped checkpoint directories to reclaim disk space, especially after
  exploratory runs.
* **Validation images.** Reduce `Logging.num_val_images` if logging slows down training, or set it to zero to disable qualitative
  logging entirely.
* **Experiment tracking.** Use descriptive W&B run names by exporting `WANDB_NAME="S2_8x_rrdb"` before launching the script.
* **EMA tuning.** Adjust `Training.EMA.decay` between 0.995 and 0.9999 depending on how aggressively you want to smooth the
  generator. Lower values react faster but may track noise; higher values provide the cleanest validation swaps.

With these components understood, you can safely modify the trainer arguments, replace callbacks, or integrate advanced logging
without losing the benefits of the existing automation.
