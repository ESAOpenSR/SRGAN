# Architecture

This document outlines how ESA OpenSR organises its super-resolution GAN, the major components that make up the model, and how each piece interacts during training and inference.

## SRGAN Lightning module

`opensr_srgan/model/SRGAN.py` defines `SRGAN_model`, a `pytorch_lightning.LightningModule` that encapsulates the full adversarial workflow. The module is initialised from a YAML configuration file and provides the following responsibilities:

* **Configuration ingestion.** Uses OmegaConf to load hyperparameters, dataset choices, and logging options. Convenience helpers
  such as `_pretrain_check()` and `_compute_adv_loss_weight()` translate config values into runtime behaviour.
* **Model factory.** `get_models()` builds the generator and discriminator at runtime via the generator factory using
  `Generator.model_type`/`block_type` and `Discriminator.model_type`. Unsupported combinations fail fast with clear error
  messages.
* **Loss construction.** `GeneratorContentLoss` (from `opensr_srgan.model.loss`) provides L1, spectral angle mapper (SAM), perceptual, and
  total-variation terms. Adversarial supervision uses `torch.nn.BCEWithLogitsLoss` with optional label smoothing.
* **Optimiser scheduling.** `configure_optimizers()` returns paired Adam optimisers (generator + discriminator) with
  `ReduceLROnPlateau` schedulers that monitor a configurable validation metric.
* **Training orchestration.** `training_step()` alternates discriminator (`optimizer_idx == 0`) and generator (`optimizer_idx ==
  1`) updates. During the warm-up period configured by `Training.pretrain_g_only`, discriminator weights are frozen via
  `on_train_batch_start()` and a dedicated `pretraining_training_step()` computes purely content-driven updates.
* **Validation and logging.** `validation_step()` computes the same content metrics, logs discriminator diagnostics, and pushes
  qualitative image panels to Weights & Biases according to `Logging.num_val_images`.
* **Inference pipeline.** `predict_step()` automatically normalises Sentinel-2 style 0–10000 inputs, runs the generator,
  histogram matches the result to the low-resolution source, and denormalises if necessary.

### Key helper methods

| Method | Purpose |
| --- | --- |
| `_pretrain_check()` | Determines whether the generator-only warm-up is active. |
| `_compute_adv_loss_weight()` | Produces the ramped adversarial weight using `linear` or `cosine` schedules. |
| `_log_generator_content_loss()` and `_log_adv_loss_weight()` | Centralise logging so metrics remain consistent across phases. |
| `on_fit_start()` | Prints informative status messages when training begins. |

## Generator options

The generator zoo lives under `opensr_srgan/model/generators/` and can be selected via `Generator.model_type` in the configuration.

* **`SRResNet` (`srresnet.py`).** Classic residual blocks with pixel shuffle upsampling. Ideal for baseline experiments or when a
  lightweight architecture is required.
* **Flexible residual families (`flexible_generator.py`).** Parameterised factory that instantiates residual, RCAB, RRDB, or
  large-kernel attention blocks while reusing the same interface. Channel counts, block depth, kernel sizes, and scaling factor
  are all read from the YAML file.
* **Stochastic GAN generator (`cgan_generator.py`).** Extends the flexible generator with conditioning inputs and latent noise,
  enabling experiments where auxiliary metadata influences the super-resolution output.
* **Advanced variants (`SRGAN_advanced.py`).** Provides additional block implementations and compatibility aliases exposed in
  `__init__.py` for backwards compatibility.

Common traits across generators include configurable input channel counts (`Model.in_bands`), support for upscaling factors from 2× to 8×, and residual scaling to stabilise deeper networks.

## Discriminator options

`opensr_srgan/model/discriminators/` exposes two complementary discriminators:

* **Standard SRGAN discriminator (`srgan_discriminator.py`).** Deep convolutional stack tailored for multispectral imagery. The
  number of convolutional blocks is configurable through `Discriminator.n_blocks`.
* **PatchGAN discriminator (`patchgan.py`).** Operates on local patches, which can improve high-frequency fidelity when training
  with large images. The depth is controlled by `n_blocks` and defaults to three layers.

Both discriminators use LeakyReLU activations and strided convolutions to progressively downsample the input until a real/fake logit map is produced.

## Loss suite and metrics

`opensr_srgan/model/loss` contains the perceptual and pixel-based criteria applied to the generator outputs. The primary entry point is `GeneratorContentLoss`, which supports:

* **L1 reconstruction** over all spectral bands.
* **Spectral Angle Mapper (SAM)** to preserve spectral signatures.
* **Perceptual similarity** via VGG or LPIPS feature spaces, depending on `Training.Losses.perceptual_metric`.
* **Total variation regularisation** for smoothing when `tv_weight` is non-zero.

The same module exposes `return_metrics()` so validation can log PSNR/SSIM-style diagnostics without recomputing forward passes.

## Data flow and normalisation

The Lightning module expects batches of `(lr_imgs, hr_imgs)` tensors supplied by the `LightningDataModule` returned from
`opensr_srgan/data/dataset_selector.py`. `predict_step()` and the validation hooks rely on two utilities from `opensr_srgan.utils.spectral_helpers`:

* `normalise_10k`: Converts Sentinel-2 style reflectance values between `[0, 10000]` and `[0, 1]`.
* `histogram`: Matches the SR histogram to the LR reference to minimise domain gaps during inference.

These helpers allow the generator to operate in a normalised space while still reporting outputs in physical units when needed.

## Putting it together

1. `opensr_srgan/train.py` loads the YAML configuration and instantiates `SRGAN_model`.
2. The model initialises the selected generator/discriminator, prepares losses, and prints a summary via
   `opensr_srgan.utils.model_descriptions.print_model_summary`.
3. During each training batch, the discriminator receives real HR crops and fake SR predictions, while the generator combines
   content loss and a ramped adversarial term.
4. Validation reuses the same modules to compute quantitative metrics and log qualitative examples.
5. When exported, `predict_step()` can be called directly or wrapped in a Lightning `Trainer.predict()` loop for large-scale
   inference.

This modular design keeps the research workflow flexible: swap components with configuration changes, extend the factories with new architectures, or plug in custom losses without touching the training loop itself.
