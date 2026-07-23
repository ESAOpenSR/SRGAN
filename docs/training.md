# Training workflow

`opensr_srgan/train.py` is the canonical entry point for ESA OpenSR experiments. It ties together configuration loading, model instantiation, dataset selection, logging, and callbacks. This page explains how the script is organised and how to customise the training loop.

!!! note "PyTorch Lightning 2+ only"
    The training stack uses a single manual-optimisation path. `SRGAN_model.setup_lightning()` enforces Lightning >= 2.0 and binds `training_step_PL2()` where GAN training runs with `automatic_optimization = False`. `opensr_srgan.utils.build_trainer_kwargs.build_lightning_kwargs()` forwards resume checkpoints through `Trainer.fit(..., ckpt_path=...)`. See [Trainer Details](trainer-details.md) for a step-by-step breakdown of warm-up checks, adversarial updates, and EMA lifecycle.

This section is a more technical overview; [Training Guideline](training-guideline.md) provides a broader overview of how to monitor the training process.

## Data module construction
In order to train, you need a dataset. `Data.dataset_type` decides which dataset to use and wraps them in a `LightningDataModule`. Should you implement your own, you will need to add it to the dataset_selector.py file with the settings of your choice (see [Data](data.md)). Optionally, the selector instantiates `ExampleDataset` by default—perfect for smoke tests after downloading the sample data, a dataset of 200 RGB-NIR image pairs. The module inherits batch sizes, worker counts, and prefetching parameters from the configuration and prints a summary including dataset size.



## Command-line and Python interfaces

You can launch training from the CLI or by importing the helper inside Python.

```bash
python opensr_srgan.train --config path/to/config.yaml
```

```python
from opensr_srgan import train

train("path/to/config.yaml")
```

Both entry points accept the same configuration file. The CLI exposes a single optional argument:

* `--config / -c`: Path to a YAML file describing the experiment. Defaults to `opensr_srgan/configs/config_20m.yaml`.

GPU assignment is handled directly in the configuration. Set `Training.gpus` to a list of device indices (for example `[0, 1, 2, 3]`) to enable multi-GPU training; a single value such as `[0]` keeps the run on one card. When more than one device is listed the trainer automatically activates PyTorch Lightning's Distributed Data Parallel (DDP) backend for significantly faster epochs.

## Initialisation steps - Overview
The code performs the following, regardless of whether the script is launched from the CLI or via import.
1. **Import dependencies.** Torch, PyTorch Lightning, OmegaConf, and logging backends are loaded up-front.
2. **Parse arguments.** `argparse` reads the configuration path and ensures the file exists.
3. **Load configuration.** `OmegaConf.load()` parses the YAML file into an object used throughout the run.
4. **Construct the model.**
   * If `Model.load_checkpoint` is set, the script calls `model.load_weights_from_checkpoint()` to import learned weights only while
     respecting the new configuration values. If `Model.continue_training` is passed with a path to a pretrained checkpoint, all scheduler states, epochs and step numbers, EMA weights, etc are loaded in order to seamlessly continue training from a previous run.
   * `Model.load_checkpoint` and `Model.continue_training` are mutually exclusive. Use only one, depending on whether you want weight initialization or full training-state resume.
    * Otherwise, it initialises a fresh `SRGAN_model`, which immediately builds the generator/discriminator and prints a
      parameter summary.
5. **Launch Training.** The training is launched with the model, weights and settings passed in the config.


## Logging setup

* **Weights & Biases.** `WandbLogger` records scalar metrics, adversarial diagnostics, and validation image panels.
* **TensorBoard.** `TensorBoardLogger` writes the same scalar metrics locally under `logs/<project>/<timestamp>`.
* **Manual SummaryWriter.** A temporary TensorBoard writer (`logs/tmp`) remains available for quick custom logging if needed.

To disable W&B logging, either remove the logger from the list or unset your API key before launching the script.

## Metrics

The Lightning module pushes the same scalar streams to both TensorBoard and W&B so you can monitor convergence from either
interface. Generator-only pretraining, adversarial training, and the EMA helper each contribute their own indicators, so the
dashboard quickly reveals which subsystem is active at any given step.

| Metric | Description | Expected behaviour |
| --- | --- | --- |
| `training/pretrain_phase` | Flag indicating whether the generator-only warm-up is running. | Stays at `1` until `g_pretrain_steps` elapses, then remains `0`. |
| `discriminator/adversarial_loss` | Binary cross-entropy loss of the discriminator on real vs. fake batches. | Drops below ~0.7 as the discriminator learns; continues trending down when D keeps up. |
| `discriminator/D(y)_prob` | Mean discriminator confidence that HR inputs are real. | Rises toward 0.8–1.0 during stable training. |
| `discriminator/D(G(x))_prob` | Mean discriminator confidence that SR predictions are real. | Starts low (~0.0–0.2) and climbs toward 0.5 as the generator improves. |
| `train_metrics/l1` | Mean absolute error between SR and HR tensors. In generator-only pretraining this is the hardwired optimization target. | Decreases toward 0 as reconstructions sharpen. |
| `train_metrics/sam` | Spectral angle mapper (radians) averaged over pixels. | Falls toward 0; values <0.1 indicate strong spectral fidelity. |
| `train_metrics/perceptual` | Perceptual distance (VGG or LPIPS) on selected RGB bands. | Decreases as textures align; exact range depends on the chosen metric. |
| `train_metrics/tv` | Total variation penalty capturing SR smoothness. | Remains small; near-zero means little high-frequency noise. |
| `train_metrics/psnr` | Peak signal-to-noise ratio (dB) on normalised tensors. | Climbs above 20 dB early; mature models reach 25–35 dB depending on data. |
| `train_metrics/ssim` | Structural Similarity Index (0–1). | Increases toward 1.0; >0.8 is typical for converged runs. |
| `generator/content_loss` | Configured weighted content objective used in both generator-only pretraining and adversarial training. | Should steadily decline and remain stable when adversarial training starts. |
| `generator/total_loss` | Sum of content and adversarial terms used to update the generator. | Tracks `generator/content_loss` early, then stabilises once adversarial weight ramps in. |
| `val_metrics/l1` | Validation MAE. | Should roughly match `train_metrics/l1`; lower is better. |
| `val_metrics/sam` | Validation SAM. | Mirrors the training trend; values <0.1 rad indicate good spectra. |
| `val_metrics/perceptual` | Validation perceptual distance. | Declines as validation textures improve. |
| `val_metrics/tv` | Validation total variation. | Stays low; spikes may signal noisy SR outputs. |
| `val_metrics/psnr` | Validation PSNR. | Rises with image quality; plateaus signal convergence. |
| `val_metrics/ssim` | Validation SSIM. | Increases toward 1.0; >0.85 suggests good structural reconstructions. |
| `validation/DISC_adversarial_loss` | Discriminator loss evaluated on validation batches. | Tracks the training discriminator loss; large swings may hint at instability. |
| `training/adv_loss_weight` | Instantaneous adversarial weight applied to the generator loss. | Sits at 0 during pretrain and ramps to `Training.Losses.adv_loss_beta`. |
| `lr_discriminator` | Learning rate used for the discriminator optimiser. | Starts at `Optimizers.optim_d_lr` and changes only when schedulers trigger. |
| `lr_generator` | Learning rate used for the generator optimiser. | Starts at `Optimizers.optim_g_lr` and follows warm-up/plateau scheduling. |
| `EMA/enabled` | Indicates whether the exponential moving average helper is active. | Constant `1` when EMA is configured, otherwise `0`. |
| `EMA/decay` | EMA decay coefficient applied to generator weights. | Fixed to the configured decay (e.g. 0.995–0.9999). |
| `EMA/update_after_step` | Step index after which EMA updates start. | Constant equal to `Training.EMA.update_after_step`. |
| `EMA/use_num_updates` | Flag showing whether the EMA tracks the number of applied updates. | `1` when `use_num_updates=True`, else `0`. |
| `EMA/is_active` | Per-step indicator that the EMA performed an update. | `0` until the warm-up expires, then `1` on steps where EMA applies. |
| `EMA/steps_until_activation` | Countdown of steps remaining before EMA activation. | Decrements to 0 and stays there once active. |
| `EMA/last_decay` | Effective decay used on the latest EMA update. | Matches the configured decay whenever the EMA updates. |
| `EMA/num_updates` | Total count of EMA updates applied so far. | Monotonically increases after activation when `use_num_updates=True`. |

## Callbacks

The following callbacks are registered with the Lightning trainer:

| Callback | Purpose |
| --- | --- |
| `ModelCheckpoint` | Saves the top two checkpoints according to `Schedulers.metric` and always keeps the last epoch. |
| `LearningRateMonitor` | Logs learning rates for both optimisers every epoch. |
| `EarlyStopping` | Monitors the same metric as the schedulers with a patience of 250 epochs and finite-check enabled. |

Checkpoint directories are nested under the TensorBoard log folder using the W&B project name and a timestamp, making it easy to
correlate files across tooling.

## Trainer configuration

The script builds a `Trainer` with the following notable arguments:

* `accelerator='cuda'` with `devices=config.Training.gpus`. When more than one device index is provided the script selects the
  `ddp` strategy automatically, so scaling across multiple GPUs is as simple as enumerating them in the config.
* `check_val_every_n_epoch=1` to evaluate after every epoch.
* `limit_val_batches=250` as a safeguard against excessive validation time on large datasets.
* `logger=[wandb_logger]` to register external logging backends (add `tb_logger` if you prefer TensorBoard-driven monitoring).
* `callbacks=[checkpoint_callback, early_stop_callback, lr_monitor]` to activate the components described above.

Finally, `trainer.fit(model, datamodule=pl_datamodule)` launches the optimisation loop and `wandb.finish()` ensures clean shutdown
of the W&B session.

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
