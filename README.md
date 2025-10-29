<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

| **PyPI** | **Versions** | **Docs / License** | **Tests** |
|:---------:|:-------------:|:------------------:|:----------:|
| [![PyPI](https://img.shields.io/pypi/v/opensr-srgan)](https://pypi.org/project/opensr-srgan/) | ![PythonVersion](https://img.shields.io/badge/Python-v3.10%20v3.12-blue.svg)<br>![PLVersion](https://img.shields.io/badge/PytorchLightning-v1.9%20v2.0-blue.svg) | [![Docs](https://img.shields.io/badge/docs-mkdocs%20material-brightgreen)](https://srgan.opensr.eu)<br>![License: Apache](https://img.shields.io/badge/license-Apache%20License%202.0-blue) | [![CI](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml)<br>[![codecov](https://codecov.io/gh/simon-donike/SISR-RS-SRGAN/graph/badge.svg?token=PWZND7MHRR)](https://codecov.io/gh/simon-donike/SISR-RS-SRGAN) |

![banner](docs/assets/6band_banner.png)

# 🌐 OpenSR GAN Lab

**OpenSR GAN Lab** turns this repository into a domain-agnostic playground for **adversarial super-resolution**. It scales from
clinical greyscale imagery to satellite stacks, microscopy volumes, and everyday RGB photos. Every component—normalisation,
architecture, loss suite, augmentation, logging—can be swapped from YAML so you can grow from quick experiments to production
pipelines without leaving the toolkit.

The project started with multispectral remote-sensing research and has since been rebuilt to serve **any modality** with
structured channels: medical imaging (MRI, CT, X-ray), aerial or orbital sensors (RGB, RGB-NIR, SWIR, SAR), industrial
inspection, cultural-heritage scans, or plain computer-vision photos. If it is an image (or a stack of bands), OpenSR GAN Lab
can model it.

---

## 📖 Documentation

Dive into the full documentation at **[srgan.opensr.eu](https://srgan.opensr.eu/)** for installation recipes, dataset templates,
training guides, and architecture deep dives tailored to multi-domain usage.

---

## 🚀 Why practitioners pick OpenSR GAN Lab

* **Any channel count, any distribution.** Describe input statistics, histogram targets, and clipping behaviour in a few YAML
  lines and the normaliser adapts—no code edits required.
* **Generator & discriminator zoo.** Mix-and-match SRResNet, ESRGAN-style RRDB, residual channel attention, stochastic
  generators, PatchGAN variants, large-kernel attention, and custom backbones you register yourself.
* **Perceptual intelligence for arbitrary bands.** LPIPS, VGG, and custom feature extractors can be scoped to the channels you
  care about so hyperspectral, multispectral, and medical volumes gain perceptual guidance without reshaping tensors.
* **Battle-tested training loop.** Warm-up phases, adversarial ramps, EMA checkpoints, mixed precision, and multi-GPU support
  are first-class citizens with reproducible defaults.
* **Observability by default.** Automatic logging to Weights & Biases or TensorBoard, validation image grids, histograms, and
  metric summaries keep stakeholders aligned regardless of domain expertise.

---

## 🧠 Capabilities at a glance

| Area | What you control |
| --- | --- |
| **Data ingestion** | Folder trees, HDF5, Zarr, Sentinel SAFE, medical DICOM/NIfTI via pluggable dataset adapters. Configure crop
sizes, sampling balance, augmentations, and multi-resolution pairings straight from YAML. |
| **Normalisation** | Z-score, min-max, percentile, histogram matching, per-channel clipping, and custom statistics loaders for
medical scanners or remote sensors. |
| **Architectures** | Prebuilt SRResNet, ESRGAN, RCAB, LKA, SwinIR-inspired blocks, plus user-registered modules with configurable
depth/width/attention. |
| **Losses** | Weighted mixes of L1/L2, spectral angle, SSIM, total variation, perceptual, relativistic adversarial, and
feature-matching objectives. |
| **Schedulers** | Generator/discriminator LR schedules (cosine, plateau, warm-up), adversarial weight ramps, gradient clipping,
and discriminator cadence controls. |
| **Deployment** | Lightning checkpoints, Hugging Face Hub integration, ONNX export helpers, patch-based tiling for arbitrarily
large rasters or volumes. |

---

## 🧪 Ready for every modality

* **Medical imaging:** Handle grayscale or multi-contrast MRI, CT, or PET by defining per-channel statistics and anatomical crop
  policies. Integrate PACS exports via simple dataset adapters.
* **Remote sensing:** Continue supporting RGB, RGB-NIR, SWIR, radar, or custom multispectral stacks with spectral-angle losses
  and histogram preservation.
* **Microscopy & materials:** Combine volumetric slices with anisotropic scaling, use tiled inference helpers, and plug in
  structure-aware perceptual metrics.
* **Consumer photography:** Fine-tune ESRGAN-style presets for compressed JPEG restoration or mobile camera enhancement.

---

## 🛠️ Configuration-first workflow

All controls live under `opensr_srgan/configs/`. Example highlights:

* **Generator** – `in_channels`, `n_channels`, `n_blocks`, `scale`, `block_type ∈ {srresnet, res, rcab, rrdb, lka, stochastic}`
  plus ESRGAN extras (`growth_channels`, `res_scale`, `out_channels`).
* **Discriminator** – `model_type ∈ {standard, esrgan, patchgan}`, `n_blocks`, `base_channels`, feature matching toggles.
* **Losses** – `l1_weight`, `mse_weight`, `sam_weight`, `ssim_weight`, `perceptual_weight`, `perceptual_metric`, `tv_weight`,
  `adv_loss_beta`, channel selection for perceptual/feature extractors.
* **Schedulers** – Warm-ups via `Schedulers.g_warmup_steps`, cosine or linear ramps, plateau-based reductions, EMA toggles.
* **Data** – Channel order, tiling, augmentations, dataset modules, cross-validation splits, and domain-specific normalisation
  statistics.

The YAML templates are heavily commented and designed to be copied, renamed, and tweaked for new projects.

---

## 🎚️ Stability features built in

* **Generator pretraining** keeps the discriminator idle for `g_pretrain_steps` so structural losses lock in early detail.
* **Adversarial weight ramping** (linear or cosine) stabilises training by gradually introducing GAN pressure until it reaches
  `adv_loss_beta`.
* **Learning-rate warm-up + restarts** smooth optimisation shocks and play nicely with mixed precision and large batch sizes.
* **EMA checkpoints** provide polished validation results and deployment-ready weights at any time.

---

## 📦 Installation

Follow the [installation guide](https://srgan.opensr.eu/getting-started/#installation) for detailed steps. In short:

```bash
# Inference or lightweight experiments
python -m pip install opensr-srgan

# Full training environment
python -m pip install -r requirements.txt
pre-commit install
```

---

## ⚡ Quickstart

1. **Select a template** from `opensr_srgan/configs/` that matches your modality (e.g., `medical_mri.yaml`, `multispectral.yaml`,
   `rgb_finetune.yaml`).
2. **Point to your dataset** by editing the `Data` section (path, format, statistics). Custom dataset adapters can be registered
   via entry points or a few lines of Python.
3. **Launch training** with `python -m opensr_srgan.train --config <config.yaml>`. Monitor progress in Weights & Biases or
   TensorBoard automatically.
4. **Run inference** using `python -m opensr_srgan.inference --config <config.yaml> --checkpoint <ckpt.ckpt>` or import the
   Lightning module in your own pipeline.

---

## 📂 Repository structure

```
SISR-RS-SRGAN/
├── opensr_srgan/         # Library + training code
├── docs/                 # MkDocs documentation sources
├── paper/                # Publication, figures, and supporting material
├── pyproject.toml        # Packaging metadata
└── requirements.txt      # Development dependencies
```

---

## 🤝 Contributing & community

Contributions are welcome! Share modality-specific configs, new generator blocks, domain adaptors, or tutorials. Review the
[contribution guidelines](CONTRIBUTING.md), open discussions, or file issues—especially if you integrate OpenSR GAN Lab into new
industries. We are building a cross-domain super-resolution hub together.

---

## 🌌 Related OpenSR projects

* **OpenSR-Model** – Latent diffusion super-resolution experiments.
* **OpenSR-Utils** – Dataset preparation, tiling, and evaluation infrastructure.
* **OpenSR-Test** – Benchmark harness for SR metrics across modalities.
* **SEN2NEON** – High-resolution reference dataset for multispectral comparison.

---

## ✍️ Citation

If you use this work, please cite:

```bibtex
coming soon...
```

---

## 👩‍🚀 Authors & AI assistance

Created by **Simon Donike** (IPL–UV) as part of the ESA Φ-lab / OpenSR initiative and now evolved into a general-purpose GAN lab.
Sections of the documentation, dataset integrations, and normalisation utilities were co-designed with support from AI tooling
(CodeX). All code and docs are reviewed to ensure accuracy across domains.
