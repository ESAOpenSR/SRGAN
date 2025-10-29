# Architecture overview

OpenSR GAN Lab is structured around a set of modular building blocks that can be remixed for any imaging domain. The project is
implemented in PyTorch Lightning; every component is configurable through YAML and can be swapped without touching training
loops.

## Lightning module

`opensr_srgan.model.module.OpenSRLightningModule` orchestrates the training, validation, and inference lifecycle:

* Instantiates generators, discriminators, and loss functions via registry-driven factories.
* Normalises inputs and denormalises outputs according to the dataset configuration.
* Computes content, perceptual, and adversarial losses while managing warm-up schedules and ramps.
* Logs metrics, images, and histograms to the configured logger (Weights & Biases or TensorBoard).
* Exposes `predict_step` and tiling helpers for inference on large rasters, volumes, or long videos.

### Dual-optimiser setup

The module supports both Lightning automatic-optimisation (for Lightning 2.x) and manual optimisation (for Lightning 1.x) to
maintain backwards compatibility. Each step updates the generator and discriminator separately while respecting cadence controls
(e.g. update the discriminator every *n* steps) and gradient clipping thresholds specified in the config.

## Generators

Generators live under `opensr_srgan.model.generator`. The registry includes:

* **SRResNet / residual blocks** – Strong baselines for RGB and grayscale imagery.
* **RCAB / channel attention** – Residual channel attention blocks for hyperspectral or medical volumes where channel
  interactions matter.
* **RRDB (ESRGAN)** – Residual-in-residual blocks with dense connections and adjustable growth channels.
* **Large-kernel attention (LKA)** – Convolution + attention hybrids for detail-rich microscopy or satellite data.
* **Stochastic generators** – Latent-conditioned branches for perceptual diversity and hallucinated detail.

Custom generators can be registered via `opensr_srgan.model.registry.register_generator`. As long as they expose the same
signature, they can be configured through YAML like any built-in model.

## Discriminators

Discriminators live in `opensr_srgan.model.discriminator` and share the same registry pattern:

* **Standard SRGAN discriminator** – Convolutional classifier operating on whole images.
* **PatchGAN variants** – Local adversaries ideal for texture-heavy microscopy or photographic enhancement.
* **ESRGAN discriminator** – Deeper architecture with spectral-normalised layers and feature matching heads.
* **3D-ready options** – 3D convolutions for volumetric data (enable by setting `Data.dimensions: 3d`).

Cadence and learning-rate scheduling can be tuned per discriminator via configuration keys.

## Loss suite

`opensr_srgan.model.losses` provides a palette of loss functions that can be blended together:

* **Pixel/structural:** L1, L2, Huber, SSIM, total variation.
* **Spectral/geometric:** Spectral angle mapper, histogram matching penalties, gradient-domain losses.
* **Perceptual:** VGG19, LPIPS, and custom feature extractors with channel-selection masks so you can target only the relevant
  bands.
* **Adversarial:** BCE-with-logits, relativistic GAN, and feature-matching losses for discriminator stabilisation.

Loss weights, warm-up phases, and perceptual channel masks are all defined in configuration files.

## Normalisation & statistics

The `opensr_srgan.data.normalizers` package converts raw sensor/scanner values into network-friendly ranges. It supports
per-channel z-score, percentile, min-max, histogram matching, and custom loaders. Normalisers can operate on 2D or 3D data and
have separate behaviour for low-resolution (LR) and high-resolution (HR) branches.

## Data pipeline

Datasets are defined in `opensr_srgan.data`. The key abstractions are:

* **Dataset selectors** – YAML-driven wrappers that map modality keywords (e.g. `medical_mri`, `sentinel2`, `rgb_folder`) to
  dataset classes.
* **Paired datasets** – Return aligned LR/HR pairs with optional augmentation pipelines.
* **Tiling datasets** – Stream patches from large rasters or volumes without loading entire scenes into memory.

You can create custom dataset classes and register them via entry points or Python hooks referenced in configuration files.

## Utilities

* **EMA manager (`opensr_srgan.utils.ema`)** – Maintains exponential moving averages of generator weights.
* **Scheduler helpers (`opensr_srgan.utils.schedulers`)** – Implement warm-ups, cosine ramps, and plateau detectors for both
  optimisers.
* **Logging utilities (`opensr_srgan.utils.loggers`)** – Standardise image grids, scalar tracking, and histogram logging across
  loggers.
* **Inference tiler (`opensr_srgan.inference.tiler`)** – Splits huge images or volumes into overlapping patches, stitches
  predictions, and restores value ranges.

Understanding how these pieces interact will help you design new models, integrate domain-specific metrics, or extend the toolkit
for novel modalities.
