<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

# OpenSR GAN Lab

| **Runtime** | **Docs / License** | **Tests** |
|:-----------:|:------------------:|:---------:|
| ![PythonVersion](https://img.shields.io/badge/Python-v3.10%20%E2%80%93%20v3.12-blue.svg)<br>![PLVersion](https://img.shields.io/badge/PyTorch%20Lightning-v1.x%20%E2%80%93%20v2.x-blue.svg) | [![Docs](https://img.shields.io/badge/docs-mkdocs%20material-brightgreen)](https://srgan.opensr.eu)<br>![License: Apache](https://img.shields.io/badge/license-Apache%20License%202.0-blue) | [![CI](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml)<br>[![codecov](https://codecov.io/gh/simon-donike/SISR-RS-SRGAN/graph/badge.svg?token=PWZND7MHRR)](https://codecov.io/gh/simon-donike/SISR-RS-SRGAN) |

![Super-resolved example collage](assets/6band_banner.png)

OpenSR GAN Lab is a **general-purpose super-resolution research environment**. It began in Earth observation and has grown into a
flexible playground for any image modality: MRI volumes, CT slices, microscopy stacks, cultural heritage scans, security
footage, UAV imagery, and consumer photographs. If your data can be described as channels, OpenSR GAN Lab lets you normalise it,
train it, and deploy it with adversarial super-resolution.

## What makes it different?

* **Domain-agnostic normalisation.** Configure per-channel statistics, histogram targets, and clipping to match the quirks of
  scanners, satellites, or cameras.
* **Composable architectures.** Swap generator or discriminator families, stack attention modules, or plug in your own PyTorch
  modules without rewriting the training harness.
* **Perceptual losses that understand your bands.** Assign which channels feed into VGG, LPIPS, or custom feature extractors so
  hyperspectral, medical, or grayscale data still benefit from perceptual guidance.
* **Robust training loop.** Warm-up phases, adversarial ramps, mixed precision, gradient accumulation, EMA, and Lightning 1.x/2.x
  compatibility are baked in.
* **Experiment telemetry.** Automated logging, validation panels, and checkpoint summaries keep researchers and stakeholders in
  sync.

> These docs capture the rebuilt, modality-agnostic version of the project. Expect references to medical, remote-sensing, and
> standard computer-vision workflows side by side.

## Choose your adventure

| Goal | Where to start |
| --- | --- |
| Install the toolkit | [Getting started](getting-started.md) |
| Understand the module layout | [Architecture](architecture.md) |
| Configure experiments | [Configuration](configuration.md) |
| Prepare datasets | [Data](data.md) |
| Launch training | [Training](training.md) + [Training guideline](training-guideline.md) |
| Run inference or deploy | [Inference](inference.md) |
| Inspect Lightning internals | [Trainer details](trainer-details.md) |
| Explore benchmarks | [Results](results.md) |

## Typical workflow

1. **Duplicate a template config.** Everything starts with YAML: define modality, dataset loader, normalisation, and architecture.
2. **Point to your data.** Use the built-in dataset selectors or register your own loader for DICOM, NIfTI, Zarr, or tiling
   pipelines.
3. **Train with confidence.** Kick off `python -m opensr_srgan.train --config <config.yaml>` and monitor metrics in Weights &
   Biases or TensorBoard.
4. **Validate and export.** Swap in EMA weights, export to ONNX/TorchScript, or tile huge scenes without running out of memory.

## Ecosystem

OpenSR GAN Lab sits within the [OpenSR](https://www.opensr.eu) initiative. It complements diffusion, transformer, and classical
SR baselines while sharing data utilities and evaluation harnesses. Use it solo or alongside the rest of the OpenSR stack.
