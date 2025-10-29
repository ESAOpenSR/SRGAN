# Training reference

This chapter explains how the training loop in OpenSR GAN Lab is structured and which switches you can flip to adapt it to your
hardware and modality.

## Workflow overview

1. **Config parsing** – YAML files are loaded and validated.
2. **Module creation** – Generator, discriminator, losses, and normalisation pipelines are instantiated.
3. **Trainer setup** – PyTorch Lightning `Trainer` is configured with devices, precision, callbacks, and logging.
4. **Training loop** – Generator-only pretraining (optional), followed by adversarial updates with warm-ups and schedules.
5. **Checkpointing & logging** – Images, metrics, and model weights are saved throughout the run.

## Command-line arguments

`python -m opensr_srgan.train` accepts the following flags:

| Flag | Description |
| --- | --- |
| `--config PATH` | Path to YAML configuration file. |
| `--resume` | Resume from the latest checkpoint in `Project.output_dir`. |
| `--checkpoint PATH` | Start training from a specific checkpoint. |
| `--devices N` | Number of GPUs or `cpu`. |
| `--accelerator` | Lightning accelerator (`gpu`, `cpu`, `mps`). |
| `--strategy` | Parallelisation strategy (`ddp`, `fsdp`, `deepspeed`, etc.). |
| `--precision` | Override numerical precision. |
| `--max-steps` | Cap the number of optimisation steps regardless of config. |
| `--limit-train-batches` | Fraction of batches to use (handy for debugging). |

## Optimisers & schedulers

* **Default optimisers** – Adam for both networks with config-defined learning rates/betas.
* **Warm-ups** – Configure linear or cosine warm-up per optimiser (`Schedulers.*.warmup_steps`).
* **Plateau schedulers** – Reduce LR on validation plateau with patience and factor controls.
* **Cosine annealing / OneCycle** – Enable via `Schedulers.generator.name` or `Schedulers.discriminator.name`.

## Mixed precision & accumulation

* Set `Training.precision` to `16` or `bf16` for mixed precision. Lightning handles loss-scaling.
* Use `Training.accumulate_grad_batches` for gradient accumulation when GPU memory is tight.

## Gradient clipping & penalties

* `Training.Stability.gradient_clip_val` – Clip global norm of gradients.
* `Training.Stability.d_reg_every` – Enable discriminator regularisation (e.g. R1 penalty) every *n* steps.

## Logging

The module logs via the configured logger:

* **Scalars** – Individual loss terms, learning rates, gradient norms.
* **Images** – LR/HR/SR triplets, optionally denormalised.
* **Histograms** – Output distributions and discriminator logits.
* **System info** – GPU utilisation, memory footprint.

Adjust logging cadence through `Logging.log_every_n_steps` and `Logging.log_images_every_n_steps`.

## Checkpoints

* **Top-K checkpoints** – Controlled by `Callbacks.checkpoint` (metric, mode, count).
* **EMA checkpoints** – Saved alongside raw weights when EMA is enabled.
* **Periodic snapshots** – Add `Callbacks.periodic` with `every_n_steps` for archival.

## Validation & testing

* Validation runs automatically according to `Trainer.check_val_every_n_epoch`.
* You can trigger additional evaluation loops by enabling `Callbacks.extra_validation` with custom hooks.
* For final testing, run `python -m opensr_srgan.validate --config ... --checkpoint ...` to compute metrics on the test split.

## Multi-node training

* Launch with `torchrun` or a Lightning SLURM script.
* Configure `strategy: ddp` (or `fsdp` for large models) and set `num_nodes`/`devices` accordingly.
* Ensure shared storage for checkpoints and dataset access.

## Debugging tips

* Use `--fast-dev-run` to sanity-check configs without committing to long runs.
* Set `limit_train_batches`/`limit_val_batches` to small fractions for quick iterations.
* Enable `Logging.log_gradients` and `Logging.log_parameters` to diagnose instability.
* Monitor GPU memory using Lightning's profiler (`Trainer.enable_progress_bar=False` for cleaner logs during debugging).

---

The training stack is designed to be transparent: every stabilisation trick is optional and controlled through configuration so
you can tailor runs to healthcare, geospatial, microscopy, or consumer imaging workloads.
