# Architecture

This document outlines how ESA OpenSR organises its super-resolution GAN, the major components that make up the model, and how each piece interacts during training and inference.

## Background

OpenSR-SRGAN follows the single-image super-resolution (SISR) formulation in which the generator learns a mapping from a low-resolution observation \(x\) to a plausible high-resolution reconstruction \(x'\). The generator head widens the receptive field, a configurable trunk of \(N\) residual-style blocks extracts features, and an upsampling tail increases spatial resolution. The residual fusion keeps skip connections active so the network focuses on high-frequency corrections rather than relearning the full signal:

$$
x' = \mathrm{Upsample}\!\left( \mathrm{Conv}_{\text{tail}}\!\left(\mathrm{Body}(x_{\text{head}}) + x_{\text{head}}\right)\! \right).
$$

Because every generator variant (residual, RCAB, RRDB, large-kernel attention, ESRGAN, SwinIR, or stochastic conditional) shares this template, you can swap block implementations without altering the training pipeline or configuration schema.


## SRGAN Lightning module

`opensr_srgan/model/SRGAN.py` defines `SRGAN_model`, a `pytorch_lightning.LightningModule` that encapsulates the full adversarial workflow. The module is initialised from a YAML configuration file and provides the following responsibilities:

* **Configuration ingestion.** Uses OmegaConf to load hyperparameters, dataset choices, and logging options. Convenience helpers
  such as `_pretrain_check()` and `_compute_adv_loss_weight()` translate config values into runtime behaviour.
* **Model factory.** `get_models()` builds the generator and discriminator at runtime via the generator factory using
  `Generator.model_type`/`block_type` and `Discriminator.model_type`. Unsupported combinations fail fast with clear error
  messages.
* **Loss construction.** `GeneratorContentLoss` (from `opensr_srgan.model.loss`) provides L1, spectral angle mapper (SAM), perceptual, and
  total-variation terms, plus optional local-Gram texture statistics. Adversarial supervision uses BCE-with-logits or Wasserstein objectives, with optional label smoothing, relativistic-average BCE, and R1 penalty depending on the config.
* **Optimiser scheduling.** `configure_optimizers()` returns paired Adam optimisers (generator + discriminator) with
  `ReduceLROnPlateau` schedulers that monitor a configurable validation metric.
* **Training orchestration.** `setup_lightning()` binds `training_step_PL2()` and enables manual optimisation
  (`automatic_optimization = False`). Each step performs explicit discriminator and generator optimiser updates; during the
  warm-up period configured by `Training.pretrain_g_only`, the generator runs content-driven updates while discriminator metrics
  are logged without stepping discriminator weights.
* **Validation and logging.** `validation_step()` computes the same content metrics, logs discriminator diagnostics, and pushes
  qualitative image panels to Weights & Biases according to `Logging.num_val_images`.
* **Inference pipeline.** `predict_step()` automatically normalises Sentinel-2 style 0–10000 inputs, runs the generator,
  applies the configured histogram matching or back-projection, and returns the supplied input units.
  Histogram matching precedes denormalisation; back-projection follows it.

### Key helper methods

| Method | Purpose |
| --- | --- |
| `_pretrain_check()` | Determines whether the generator-only warm-up is active. |
| `_compute_adv_loss_weight()` | Produces the ramped adversarial weight using `linear` or `cosine` schedules. |
| `_log_generator_content_loss()` and `_log_adv_loss_weight()` | Centralise logging so metrics remain consistent across phases. |
| `postprocess(sr, lr)` | Applies the selected histogram matching or back-projection to tensors in a shared value range. |
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
* **ESRGAN generator (`esrgan.py`).** Implements the RRDBNet trunk introduced with ESRGAN, exposing `n_blocks`, `growth_channels`,
  and `res_scale` so you can dial in deeper receptive fields and sharper textures. The shared training loop supports optional relativistic-average BCE and a two-step training phase (content-oriented pretraining of the generator followed by adversarial training with the discriminator).
* **SwinIR generator (`swinir.py`).** Uses residual groups of regular and shifted-window self-attention, followed by a pixel-shuffle reconstruction head. See the settings and example below.
* **Advanced variants (`SRGAN_advanced.py`).** Provides additional block implementations and compatibility aliases exposed in
  `__init__.py` for backwards compatibility.

All generators accept `Model.in_bands` and `Generator.scaling_factor`. Supported
scales and trunk settings depend on the selected backbone; CNN residual scaling
settings do not apply to SwinIR.

### SwinIR generator

SwinIR combines local self-attention with shifts between attention windows so
features can interact across window boundaries. This implementation takes
multispectral tensors, pads them to complete windows, and crops the output to
exactly the requested scaled size. It supports 2×, 3×, and 4× upscaling, including
small or non-square inputs. It applies no normalization or clipping internally;
the shared data and inference pipeline handles value ranges.

Select `model=10m_swinir` in Hydra, or replace the `Generator` section of an
existing flat YAML with:

```yaml
Generator:
  model_type: swinir
  scaling_factor: 4
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  window_size: 8
  mlp_ratio: 2.0
  qkv_bias: true
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
  use_checkpoint: false
```

Input and default output bands come from `Model.in_bands`; the shipped model
preset sets both to four. `depths` and `num_heads` must have equal nonzero lengths,
and `embed_dim` must divide evenly by each head count. Set `use_checkpoint: true`
to trade additional backward computation for lower activation memory use.
CNN-only fields such as `n_channels`, `n_blocks`, and `growth_channels` are ignored
with a factory notice.

The [SwinIR training example](training.md#train-a-swinir-reconstruction-model)
uses `experiment=swinir_sen2naip` for reconstruction pretraining from scratch.
ESRGAN/RRDB checkpoints and official upstream SwinIR checkpoints are not compatible
with this implementation. The successful [local-Gram recipe](sen2naip-texture.md)
continues an ESRGAN model and remains a separate experiment.
See the [configuration reference](configuration.md#swinir-settings) for all
SwinIR controls and the [model API](api-reference/model.md#swinir-transformer)
for its constructor.

## Discriminator options

`opensr_srgan/model/discriminators/` exposes three complementary discriminators:

* **Standard SRGAN discriminator (`srgan_discriminator.py`).** Deep convolutional stack tailored for multispectral imagery. The
  number of convolutional blocks is configurable through `Discriminator.n_blocks`.
* **PatchGAN discriminator (`patchgan.py`).** Operates on local patches, which can improve high-frequency fidelity when training
  with large images. The depth is controlled by `n_blocks` and defaults to three layers.
* **ESRGAN discriminator (`esrgan.py`).** Deep VGG-style stack with configurable `base_channels` and `linear_size`; pairs well
  with RRDB generators when perceptual sharpness is the priority.

Both discriminators use LeakyReLU activations and strided convolutions to progressively downsample the input until a real/fake logit map is produced.

## Loss suite and metrics

`opensr_srgan/model/loss` contains the perceptual and pixel-based criteria applied to the generator outputs. The primary entry point is `GeneratorContentLoss`, which supports:

* **L1 reconstruction** over all spectral bands.
* **Spectral Angle Mapper (SAM)** to preserve spectral signatures.
* **Perceptual similarity** via VGG or LPIPS feature spaces, depending on `Training.Losses.perceptual_metric`.
* **Local-Gram texture loss** on preactivation VGG conv2_2/conv3_4 features when `texture_weight` is non-zero.
* **Total variation regularisation** for smoothing when `tv_weight` is non-zero.

The same module exposes `return_metrics()` so validation can log PSNR/SSIM-style diagnostics without recomputing forward passes.

## Data flow and normalisation

The Lightning module expects batches of `(lr_imgs, hr_imgs)` tensors supplied by the `LightningDataModule` returned from
`opensr_srgan/data/dataset_selector.py`. `predict_step()` uses the configured `opensr_srgan.data.utils.Normalizer` plus the matching method selected by `Postprocessing.method`:

* `Normalizer`: Applies the configured `Data.normalization` strategy, such as `normalise_10k`, `zero_one`, or a custom callable.
* `histogram` (default): Matches the SR histogram to the LR reference before denormalisation.
* `back_projection`: Corrects denormalised SR so its downsampled measurements reproduce the LR input.

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
