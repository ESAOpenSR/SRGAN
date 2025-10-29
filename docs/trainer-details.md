# Trainer details

This page dives into the PyTorch Lightning trainer configuration used in OpenSR GAN Lab, explaining how key hooks are implemented
and how to customise them for specialised workloads.

## Automatic vs. manual optimisation

The Lightning module detects the installed Lightning version and toggles between automatic and manual optimisation modes:

* **Lightning ≥ 2.0** – Uses automatic optimisation; optimisers are returned from `configure_optimizers` and Lightning handles
  stepping.
* **Lightning 1.x** – Switches to manual optimisation to retain fine-grained control over generator/discriminator updates.

You can force a mode via the config (`Training.manual_optimization: true/false`) if desired.

## Training step flow

1. **Fetch batch** containing LR/HR tensors and metadata.
2. **Normalise** using the configured normaliser (per-branch logic allowed).
3. **Generator forward** to produce SR output.
4. **Compute reconstruction losses** (L1, SSIM, etc.).
5. **If adversarial active:**
   * Update discriminator according to cadence (`d_update_interval`).
   * Compute adversarial and feature-matching losses.
   * Apply EMA update after generator step if enabled.
6. **Log** scalar metrics, images, and optional histograms.

## Callback suite

### Checkpointing

`ModelCheckpoint` stores top-*k* checkpoints and monitors user-defined metrics. EMA checkpoints are saved automatically alongside
the raw generator weights.

### Learning-rate monitoring

`LearningRateMonitor` records LR schedules for both optimisers. Enable it via `Callbacks.lr_monitor: true`.

### Gradient monitoring

`GradientNormLogger` (custom callback) logs gradient norms per network. Configure thresholds to trigger warnings in long-running
jobs.

### EMA swapper

`EMACallback` swaps EMA weights in before validation/prediction and restores raw weights afterwards. This ensures validation uses
the smoothed generator.

### Extra validation hooks

Add custom validation logic (e.g. running domain-specific metrics) by registering callbacks under `Callbacks.extra_validation`.
Each callback receives the Lightning module and current outputs.

## Mixed precision considerations

Lightning handles autocast and gradient scaling when `Training.precision` is 16 or bf16. If you introduce custom CUDA ops, ensure
they support the selected precision or guard them with `torch.cuda.amp.autocast(enabled=...)`.

## Distributed strategies

* **DDP** – Default for multi-GPU runs. Ensure `find_unused_parameters` is set appropriately for custom modules.
* **FSDP** – Supported for very large models; configure auto-wrapping policies via `Training.fsdp` block.
* **DeepSpeed** – Available for enthusiasts wanting ZeRO optimisations. Requires additional configuration entries for stage,
  offload, and micro-batching.

## Logging integrations

* **Weights & Biases** – Configured via `Logging.logger: wandb`. The trainer logs metrics, media, and configuration snapshots.
* **TensorBoard** – Set `Logging.logger: tensorboard` to write local event files.
* **CSV** – Minimal logging for constrained environments.

## Early stopping & alerts

Add `Callbacks.early_stopping` with `monitor`, `mode`, and `patience` fields. Combine with gradient monitors or custom alert
hooks to notify operators when training stalls.

## Custom trainer arguments

Provide additional `Trainer` kwargs through the config:

```yaml
Trainer:
  enable_progress_bar: true
  log_every_n_steps: 50
  gradient_clip_algorithm: norm
  detect_anomaly: false
```

Any argument supported by Lightning's `Trainer` can be set this way, enabling precise tuning for HPC clusters, hospital servers,
or research laptops.

---

Understanding these trainer details helps you reason about behaviour during long training runs and customise the pipeline for new
domains without editing the core codebase.
