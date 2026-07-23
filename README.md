<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

| **PyPI** | **Versions** | **Docs & License** | **Tests** | **Reference** |
|:---------:|:-------------:|:------------------:|:----------:|:--------------:|
| [![PyPI](https://img.shields.io/pypi/v/opensr-srgan)](https://pypi.org/project/opensr-srgan/) | ![PythonVersion](https://img.shields.io/badge/Python-v3.12--v3.14-blue.svg)<br>![PLVersion](https://img.shields.io/badge/PyTorchLightning-v2%2B-blue.svg) | [![Docs](https://img.shields.io/badge/docs-mkdocs%20material-brightgreen)](https://srgan.opensr.eu)<br>![License: Apache](https://img.shields.io/badge/license-Apache%20License%202.0-blue) | [![CI](https://github.com/ESAOpenSR/SRGAN/actions/workflows/ci.yml/badge.svg)](https://github.com/ESAOpenSR/SRGAN/actions/workflows/ci.yml)<br>[![codecov](https://codecov.io/github/ESAOpenSR/SRGAN/graph/badge.svg?token=LQ69MIMLVE)](https://codecov.io/github/ESAOpenSR/SRGAN) |  [![arXiv](https://img.shields.io/badge/arXiv-2511.10461-b31b1b.svg)](https://arxiv.org/abs/2511.10461)  <br> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088994.svg)](https://doi.org/10.5281/zenodo.20088994)




![banner](docs/assets/6band_banner.png)

# 🌍 Single Image Super-Resolution Remote Sensing 'SRGAN'

**Remote-Sensing-SRGAN** is a research-grade GAN framework for super-resolution of Sentinel-2 and other remote-sensing imagery. It is specifically **not meant for SOTA performance, but quick implementation and experimentation**. It supports arbitrary band counts, configurable generator/discriminator designs, scalable depth/width, and a modular loss system designed for stable GAN training on EO data. Check out how inference and training works right now in this interactive example! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ESAOpenSR/SRGAN/blob/main/notebooks/SRGAN_Showcase.ipynb)


---

## 📖 Documentation

Full docs live at **[srgan.opensr.eu](https://srgan.opensr.eu/)**. They cover usage, configuration, training recipes, and deployment tips in depth.

## 🧠 Highlights

* **Flexible models:** swap between SRResNet, RCAB, RRDB, LKA, ESRGAN, and stochastic generators with YAML-only changes.
* **Remote-sensing aware losses:** combine spectral, perceptual, and adversarial objectives with tunable weights.
* **Stable training loop:** generator pretraining, adversarial ramp-ups, EMA, and multi-GPU Lightning support out of the box.
* **PyPI distribution:** `pip install opensr-srgan` for ready-to-use presets or custom configs.
* **Experiment tracking:** Log metrics and validation imagery to Weights & Biases, or use local CSV logs for lightweight runs.

---

## 🏗️ Configuration Examples

All key knobs are exposed via YAML in the `opensr_srgan/configs` folder. You can keep using the legacy single-file workflow, or use the Hydra entry point for composable experiment presets and command-line overrides:

```bash
python -m opensr_srgan.train --config opensr_srgan/configs/config_training_example.yaml
python -m opensr_srgan.train_hydra experiment=10m Training.max_epochs=5 Logging.wandb.enabled=false
srgan-train experiment=20m Training.gpus=[0,1]
```

* **Model**: `Model.in_bands`, `Generator.n_channels`, `Generator.n_blocks`, `Generator.scaling_factor`, ESRGAN knobs (`growth_channels`, `res_scale`, `out_channels`, `use_icnr`), and `Generator.block_type ∈ {standard, res, rcab, rrdb, lka}`
* **Losses**: `l1_weight`, `sam_weight`, `perceptual_weight`, `tv_weight`, `adv_loss_beta`
* **Training**: `pretrain_g_only`, `g_pretrain_steps` (`-1` keeps generator-only pretraining active indefinitely), `adv_loss_ramp_steps`, `label_smoothing`, generator LR warmup (`Schedulers.g_warmup_steps`, `Schedulers.g_warmup_type`), EMA, and manual gradient clipping
* **Adversarial mode**: `Training.Losses.adv_loss_type` (`bce`/`wasserstein`) and optional `Training.Losses.relativistic_average_d` for BCE-based relativistic-average GAN updates
* **Data**: `Data.dataset_type`, dataloader sizes/workers, dataset paths/manifests, and normalization policy

---

## 🎚️ Training Stabilization Strategies

* **G‑only pretraining:** Train with content/perceptual losses while the adversarial term is held at zero during the first `g_pretrain_steps`.
* **Adversarial ramp‑up:** Increase the BCE adversarial weight **linearly** or smoothly (**cosine**) over `adv_loss_ramp_steps` until it reaches `adv_loss_beta`.
* **Generator LR warmup:** Ramp the generator optimiser with a **cosine** or **linear** schedule for the first 1–5k steps via `Schedulers.g_warmup_steps`/`g_warmup_type` before switching to plateau-based reductions.
* **EMA smoothing:** Enable `Training.EMA.enabled` to keep a shadow copy of the generator. Decay values in the 0.995–0.9999 range balance responsiveness with stability and are swapped in automatically for validation/inference.
* **Spectral normalization:** Optional for the SRGAN discriminator via `Discriminator.use_spectral_norm` to better control its Lipschitz constant and stabilize adversarial updates. [Miyato et al., 2018](https://arxiv.org/abs/1802.05957)
* **Wasserstein critic + R1 penalty:** Switch `Training.Losses.adv_loss_type: wasserstein` to enable a critic objective and pair it with the configurable `Training.Losses.r1_gamma` gradient penalty on real images for smoother discriminator updates. [Arjovsky et al., 2017](https://arxiv.org/abs/1701.07875); [Mescheder et al., 2018](https://arxiv.org/abs/1801.04406)
* **Relativistic average GAN (BCE):** Set `Training.Losses.relativistic_average_d: true` to train D/G on relative real-vs-fake logits instead of absolute logits. This is supported in the Lightning 2+ manual-optimization training path.
The schedule and ramp make training **easier, safer, and more reproducible**.

---

## ⚙️ Config‑driven components

| Component | Options | Config keys |
|-----------|---------|-------------|
| **Generators** | `SRResNet`, `res`, `rcab`, `rrdb`, `lka`, `esrgan`, `stochastic_gan` | `Generator.model_type`, depth via `Generator.n_blocks`, width via `Generator.n_channels`, kernels/scale plus ESRGAN-specific `growth_channels`, `res_scale`, `out_channels`. |
| **Discriminators** | `standard` SRGAN, `patchgan`, `esrgan` | `Discriminator.model_type`, granularity with `Discriminator.n_blocks`, spectral norm toggle via `Discriminator.use_spectral_norm`, ESRGAN-specific `base_channels`, `linear_size`. |
| **Content losses** | L1, Spectral Angle Mapper, VGG19/LPIPS perceptual metrics, Total Variation | Weighted by `Training.Losses.*` (e.g. `l1_weight`, `sam_weight`, `perceptual_weight`, `perceptual_metric`, `tv_weight`). |
| **Adversarial loss** | BCE‑with‑logits or Wasserstein critic | Controlled by `Training.Losses.adv_loss_type`, warmup via `Training.pretrain_g_only`, ramped by `adv_loss_ramp_steps`, capped at `adv_loss_beta`, optional label smoothing. For BCE, enable `Training.Losses.relativistic_average_d` for RaGAN-style relative logits. |

The YAML keeps the SRGAN flexible: swap architectures or rebalance perceptual vs. spectral fidelity without touching the code.


## 🧰 Installation

Follow the [installation instructions](https://srgan.opensr.eu/getting-started/) for package, source, and dependency setup options.

---

## 🚀 Quickstart

* To test the package immediately, launch the Google Colab right now and follow along the introduction! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ESAOpenSR/SRGAN/blob/main/notebooks/SRGAN_Showcase.ipynb)
* **Datasets:** Grab the bundled example dataset or learn how to register your own sources in the [data guide](https://srgan.opensr.eu/data/).
* **Training:** Launch a smoke test with `python -m opensr_srgan.train --config opensr_srgan/configs/config_training_example.yaml`, or use Hydra with `python -m opensr_srgan.train_hydra experiment=example`. Use the `10m` and `20m` presets once the matching datasets are configured.
* **Inference:** Ready-made presets and large-scene pipelines are described in the [inference section](https://srgan.opensr.eu/inference/).

---

## 🏗️ Configuration & Stabilization

All tunable knobs—architectures, loss weights, schedulers, and EMA—are exposed via YAML files under `opensr_srgan/configs`. Strategy tips for warm-ups, adversarial ramps, and EMA usage are summarised in the [training concepts chapter](https://srgan.opensr.eu/training-guideline/).


## 📂 Repository Structure

```
SRGAN/
├── opensr_srgan/         # Library + training code
├── docs/                 # MkDocs documentation sources
├── paper/                # Publication, figures, and supporting material
├── pyproject.toml        # Packaging metadata
└── requirements.txt      # Development dependencies
```

## Contribution and Issues
If you wish to contribute (such as new models, data or functionalities), please review the contribution guidelines and open a PR here on Github. If you're having problems or need support, please open an Issue here on Github.


---

## 📚 Related Projects

* **OpenSR-Model** – Latent Diffusion SR (LDSR-S2)
* **OpenSR-Utils** – Large-scale inference & data plumbing
* **OpenSR-Test** – Benchmarks & metrics
* **SEN2NEON** – Multispectral HR reference dataset

---

## ✍️ Citation

If you use this work, please cite:

```bibtex  
@misc{donike2025opensrsrganflexiblesuperresolutionframework,
      title={OpenSR-SRGAN: A Flexible Super-Resolution Framework for Multispectral Earth Observation Data}, 
      author={Simon Donike and Cesar Aybar and Julio Contreras and Luis Gómez-Chova},
      year={2025},
      eprint={2511.10461},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.10461}, 
}
```
---

## 🧑‍🚀 Authors, Acknowledgements, and Notes on AI Usage

Developed within the **ESA Φ-lab / OpenSR** initiative. Simon Donike is the main contributor and maintainer of the repository. Cesar Aybar and Julio Contreras contributed the datasets as well as implementation, documentation and publishing support. Prof. Luis Gómez-Chova contributed the remote sensing-specific perspective and signal processing advice.  
> The development history of this code began in 2020 with the implementation of an SR-GAN for a MSc thesis project. Since then, over several iterations, the codebase has been expanded and many training tweaks implemented, based on the experiences made training SR-GANs for the OpenSR project. The fundamental training outline, training tweaks, normalizations, and inference procedures are built upon that experience.  
The added complexity that came with (a) the implementation of many different models and blocks, (b) more data sources, (c) according normalizations, and (d) testing and documentation structures, was handled to varying degrees with the help of *Codex*. Specifically, the docs, the automated testing workflows, and the normalizer class are in (large) parts AI generated. This code and its functionalities have been verified and tested to the best of my ability.

---
