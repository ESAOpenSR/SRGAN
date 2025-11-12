---
title: "OpenSR-SRGAN: A Flexible Super-Resolution Framework for Multispectral Earth Observation Data"
tags:
  - super-resolution
  - remote sensing
  - GAN
  - ESRGAN
  - Sentinel-2
  - multispectral
  - PyTorch
  - OpenSR
  - technical report
authors:
  - name: Simon Donike
    orcid: 0000-0002-4440-3835
    corresponding: true
    email: simon.donike@uv.es
    affiliation: 1
  - name: Luis Gómez-Chova
    orcid: 0000-0003-3924-1269
    affiliation: 1
affiliations:
  - index: 1
    name: Image and Signal Processing Group, University of Valencia, Spain
date:  15 November 2025
bibliography: paper.bib
version:
  report: v0.1.0
  software: v0.1.0
doi: 10.5281/zenodo.17590993
number-sections: true
---

# Summary

We present OpenSR-SRGAN, an open and modular framework for single-image super-resolution in Earth Observation. The software provides a unified implementation of SRGAN-style models that is easy to configure, extend, and apply to multispectral satellite data such as Sentinel-2. Instead of requiring users to modify model code, OpenSR-SRGAN exposes generators, discriminators, loss functions, and training schedules through concise configuration files, making it straightforward to switch between architectures, scale factors, and band setups. The framework is designed as a practical tool and benchmark implementation rather than a state-of-the-art model. It ships with ready-to-use configurations for common remote-sensing scenarios, sensible default settings for adversarial training, and built-in hooks for logging, validation, and large-scene inference.
By turning GAN-based super-resolution into a configuration-driven workflow, OpenSR-SRGAN lowers the entry barrier for researchers and practitioners who wish to experiment with SRGANs, compare models in a reproducible way, and deploy super-resolution pipelines across diverse Earth-observation datasets.


# Introduction

Optical satellite imagery plays a key role in monitoring the Earth's surface for applications such as agriculture [@agriculture], land cover mapping [@mapping], ecosystem assessment [@ecosysetm], and disaster management [@disaster]. The European Space Agency’s Sentinel-2 mission provides freely available multispectral imagery at 10 m spatial resolution with a revisit time of five days, enabling dense temporal monitoring at global scale. In contrast, very-high-resolution sensors, such as Pleiades or SPOT, offer much richer spatial detail but limited temporal coverage and high acquisition costs.  
Consequently, a trade-off exists between spatial and temporal resolution in Earth-Observation (EO) imagery.

Single-image super-resolution (SISR) aims to enhance the spatial detail of low-resolution (LR) observations by learning a mapping to a plausible high-resolution (HR) counterpart.  
In remote sensing, SR can bridge the gap between freely available medium-resolution imagery and costly commercial data, potentially improving downstream tasks such as land-cover classification, object detection, and change detection.  
The advent of deep convolutional networks led to major breakthroughs in both reconstruction fidelity and perceptual realism [@dong2015imagesuperresolutionusingdeep; @kim2016deeplyrecursiveconvolutionalnetworkimage].

Generative Adversarial Networks (GANs) [@goodfellow2014generativeadversarialnetworks] introduced an adversarial learning framework in which a generator and a discriminator are trained in competition, enabling the synthesis of realistic, high-frequency image details.  
Since their introduction, GANs have been rapidly adopted in the remote-sensing community for tasks such as cloud removal, image translation, domain adaptation, and data synthesis [@11159252; @su2024intriguingpropertycounterfactualexplanation].  
These applications demonstrated the potential of adversarial training to generate spatially coherent and perceptually plausible remote-sensing imagery.

Building on these successes, the computer-vision community introduced the Super-Resolution GAN (SRGAN) [@ledig2017photo], which combined perceptual and adversarial losses to reconstruct photo-realistic high-resolution images from their low-resolution counterparts.  
The approach inspired a wave of research applying SRGAN-like architectures to remote-sensing super-resolution [@rs15205062; @9787539; @10375518; @satlassuperres], where the ability to recover fine spatial detail from coarse observations can significantly enhance analysis of land cover, infrastructure, and environmental change.

Recent advances in diffusion and transformer-based architectures have shifted the state of the art in image super-resolution toward generative models with stronger probabilistic and contextual reasoning [@s1; @s2; @s3].  
Nevertheless, GAN-based approaches continue to be actively explored [@g1] and remain a practical choice for operational production settings [@allen].

# Problem Statement

Despite their success in computer vision, Generative Adversarial Networks (GANs) remain notoriously difficult to train [@p1; @p2; @p3]. The simultaneous optimization of generator and discriminator networks often leads to unstable dynamics, mode collapse, and high sensitivity to hyperparameters.  

In remote-sensing applications, these issues are amplified by domain-specific challenges such as multispectral or hyperspectral inputs, high dynamic range reflectance values, varying sensor characteristics, and limited availability of perfectly aligned high-resolution ground-truth data.  

Moreover, researchers in remote sensing rarely work with fixed RGB imagery. They frequently need to adapt existing GAN architectures to support arbitrary numbers of spectral bands, retrain models for different satellite sensors (e.g., Sentinel-2, SPOT, Pleiades, PlanetScope), or implement benchmarks for newly collected datasets.  
These modifications usually require non-trivial changes to the model architecture, preprocessing pipeline, and loss configuration, making reproducibility and experimentation cumbersome.  

Implementing the full set of heuristics that make GAN training stable, such as generator pretraining, adversarial loss ramping, label smoothing, learning-rate warmup, and exponential moving-average (EMA) tracking, adds another layer of complexity. Consequently, reproducing and extending GAN-based SR models in the Earth-Observation (EO) domain is often time-consuming, fragile, and inconsistent across studies.



# Contribution Summary

OpenSR-SRGAN was developed to address these challenges by providing a unified, modular, and extensible framework for training and evaluating GAN-based super-resolution models in remote sensing.  
The software integrates multiple state-of-the-art SR architectures, loss functions, and training strategies within a configuration-driven design that allows users to flexibly adapt experiments without modifying the source code.

The main contributions of this work include:

- **Modular GAN framework:** Supports interchangeable generator and discriminator architectures with customizable depth, width, and scale factors.  
- **Configuration-first workflow:** Enables fully reproducible training and evaluation through concise YAML files, independent of code changes.  
- **Training stabilization techniques:** Includes generator pretraining, adversarial ramp-up, learning-rate warmup, label smoothing, and EMA smoothing.  
- **Multispectral compatibility:** Provides native support for arbitrary band configurations from different satellite sensors.  
- **OpenSR ecosystem integration:** Connects seamlessly to the SEN2NAIP dataset, leverages the unified evaluation framework `opensr-test` [@osrtest], and supports scalable inference via `opensr-utils` [@osrutils].

Together, these features make OpenSR-SRGAN a reliable boilerplate for researchers and practitioners to train, benchmark, and deploy GAN-based SR models across diverse Earth-observation datasets and sensor types.

# Software Overview and Framework Design

OpenSR-SRGAN follows a modular and configuration-driven architecture. All model definitions, loss compositions, and training schedules are controlled through a single YAML configuration file, ensuring that experiments remain reproducible and easily adaptable to new sensors, band configurations, or datasets. The framework is implemented in PyTorch and PyTorch Lightning, providing seamless GPU acceleration and built-in experiment logging.  

The system consists of four main components:  

- A flexible **generator–discriminator architecture** supporting multiple SR backbones.  
- A configurable **multi-loss system** combining pixel, perceptual, spectral, and adversarial objectives.  
- A robust **training pipeline** with pretraining, warmup, ramp-up, and EMA stabilization mechanisms.  
- Integration with the **OpenSR ecosystem** for dataset access, evaluation (`opensr-test`), and large-scale inference (`opensr-utils`).  

## Generator Architectures

The generator network can be configured with different backbone types, each providing a unique trade-off between complexity, receptive field, and textural detail (see Table~\ref{tab:arch} in~\ref{app:components}).  

The `Generator` class provides a unified implementation of SR backbones that share a common convolutional structure while differing in their internal residual block design.
The module is initialized with a `model_type` flag selecting one of `res`, `rcab`, `rrdb`, `lka`, `esrgan`, `cgan`, each drawn from a shared registry of block factories or dedicated ESRGAN implementation.
Given an input tensor $x$, the model applies a wide receptive-field head convolution, followed by $N$ residual blocks of the selected type, a tail convolution for residual fusion, and an upsampling module that increases spatial resolution by a factor of 2, 4, or 8.  
The network ends with a linear output head producing the super-resolved image:
$$
x' = \mathrm{Upsample}\!\left( \mathrm{Conv}_{\text{tail}}\!\left(\mathrm{Body}(x_{\text{head}}) + x_{\text{head}}\right)\! \right).
$$

This modular structure allows researchers to experiment with different block designs, standard residual, channel attention (RCAB), dense residual (RRDB), large-kernel attention (LKA), or noise-augmented variants, without altering the training pipeline or configuration schema. All models share identical input-output interfaces and residual scaling for stability, ensuring drop-in interchangeability across experiments.

### Conditional Generator with Noise Injection

The `cgan` variant extends the standard generator by conditioning on both the low-resolution image and a latent noise vector $z$.  
It replaces standard residual blocks with *NoiseResBlocks*, which introduce controlled stochasticity. Each `NoiseResBlock` uses a small Multi-Layer Perceptron (MLP) to project the noise code $z\!\in\!\mathbb{R}^{d}$ into per-channel scale and bias parameters $(\gamma, \beta)$ that modulate intermediate activations before the nonlinearity:

$$
x_{\text{mod}} = (1 + \gamma)\odot \mathrm{Conv}_1(x) + \beta
$$

followed by a second convolution and residual fusion:

$$
x' = x + \mathrm{Conv}(\mathrm{PReLU}(x_{\text{mod}})) \cdot s
$$

where $s$ is a residual scaling factor. This mechanism maintains spatial coherence while enabling the generation of multiple plausible high-frequency realizations for the same LR input. During training, a random noise vector is sampled per image; during inference, users may supply explicit latent codes or fix random seeds for deterministic behavior. The module also exposes `sample_noise(batch_size)` and a `return_noise` flag for reproducibility and logging.  


## Discriminator Architectures

The discriminator can be selected to prioritize either global consistency or fine local realism. The different architectures and their purposes are outlined in Table~\ref{tab:disc} in~\ref{app:components}. Three discriminator variants are implemented to complement the different generator types: a global `Discriminator`, a local `PatchGANDiscriminator`, and the deeper `ESRGANDiscriminator`. All are built from shared convolutional blocks with LeakyReLU activations and instance normalization.

The standard discriminator follows the original SRGAN [@ledig2017photo] design and evaluates the realism of the entire super-resolved image and the actual HR image. It stacks a sequence of strided convolutional layers with progressively increasing feature channels, an adaptive average pooling layer to a fixed spatial size, and two fully connected layers producing a scalar real/fake score. This 'global' discriminator promotes coherent large-scale structure and overall photorealism.

The `PatchGANDiscriminator` instead outputs a grid of patch-level predictions, classifying each overlapping region as real or fake. Built upon the CycleGAN/pix2pix [@cyclegan; @px2px] reference implementation, it uses a configurable number of convolutional layers and normalization schemes (batch, instance, or none). The resulting patch map acts as a spatial realism prior, emphasizing texture fidelity and fine detail.

Finally, the `ESRGANDiscriminator` mirrors the deeper VGG-style stack from ESRGAN. Its `base_channels` and fully connected `linear_size` can be tuned to match the generator capacity, offering an aggressive adversarial signal when paired with RRDB-based generators.

Together, these architectures allow users to select the appropriate adversarial granularity: global consistency through SRGAN-style discrimination, local realism through PatchGAN, or sharper perceptual contrast via ESRGAN.

# Training Features

Training stability is improved through several built-in mechanisms that address common issues of adversarial optimization (summarized in Table~\ref{tab:train}, ~\ref{app:components}). These are configured in the `Training` section of the YAML `config` file.

## General Training Optimizations
Several additional methods contribute to stable adversarial optimization. Label smoothing replaces hard discriminator targets (1 for real, 0 for fake) with softened values such as 0.9 and 0.1, preventing overconfidence and promoting smoother gradients. A short generator warmup phase allows $G$ to learn basic low-frequency structure before adversarial feedback is introduced, often combined with a linear or cosine learning-rate ramp to avoid abrupt updates. The discriminator holdback delays $D$ updates for the first few epochs so that $G$ can stabilise; when enabled, $D$ also follows a short warmup schedule to balance learning rates. Finally, both optimisers employ adaptive scheduling via `ReduceLROnPlateau`, lowering the learning rate when progress stagnates. These implementations mitigate divergence and improve convergence stability in adversarial training. All of these techniques can be configured from the `config` file as the unified entry-point.



## Exponential Moving Average (EMA) Stabilisation

The EMA mechanism [@ema] is an optional stabilisation technique applied to the generator weights to produce smoother outputs and more reliable validation metrics, commonly used throughout model training pipelines in general and generative image applications in particular. Instead of evaluating the generator using its raw, rapidly fluctuating parameters, an auxiliary set of duplicate weights $\theta_{\text{EMA}}$ is maintained as a smoothed version of the online weights $\theta$. After each training step, the model parameters are updated as an exponential moving average:
$$
\theta_{\text{EMA}}^{(t)} = \beta \, \theta_{\text{EMA}}^{(t-1)} + (1 - \beta)\, \theta^{(t)}
$$
where $\beta \in [0,1)$ is the decay factor controlling how much past states influence the smoothed estimate.  
A higher $\beta$ (e.g., 0.999) gives longer memory and stronger smoothing, while a lower value responds more quickly to new updates.  

During validation and inference, the EMA parameters replace the instantaneous generator weights, yielding more temporally consistent reconstructions and reducing the variance of perceptual and adversarial loss curves.  
The inference process thus evaluates:
\begin{equation}
\hat{y}_{\text{SR}} = G(x; \theta_{\text{EMA}})
\end{equation}
where $\hat{y}_{\text{SR}}$ denotes the final super-resolved output produced by the EMA-stabilised generator. Empirically, applying EMA has been shown to stabilise adversarial training by mitigating oscillations between the generator and discriminator and by improving the perceptual smoothness and reproducibility of the resulting super-resolved images [@ema2].

## Loss Functions

Each loss term (see Table~\ref{tab:loss} in~\ref{app:components}) can be weighted independently, allowing users to balance spectral accuracy and perceptual realism. Typical configurations combine L1, Perceptual, and Adversarial losses, optionally augmented by SAM and TV for multispectral consistency and smoothness. The overall objective is a weighted sum of these terms defined in the `Training.Losses ` section of the configuration. A detailed description of the internal training and validation metrics logged alongside these losses is given in ~\ref{tab:metrics}.


# Limitations
Super-resolution techniques, including those implemented in OpenSR-SRGAN, can enhance apparent spatial detail but can never substitute for true high-resolution observations acquired by native sensors.  
While OpenSR-SRGAN provides a stable and extensible foundation for GAN-based super-resolution in remote sensing, several limitations remain. First, the framework focuses on the engineering and reproducibility aspects of model development rather than achieving state-of-the-art quantitative performance. It is therefore intended as a research and benchmarking blueprint, not as an optimized production model. Second, although the modular configuration system greatly simplifies experimentation, users are still responsible for ensuring proper data preprocessing, radiometric normalization, and accurate LR–HR alignment, factors that strongly influence training stability and reconstruction quality. Third, adversarial optimization in multispectral domains remains sensitive to dataset size and diversity; small or unbalanced datasets may still yield mode collapse or spectral inconsistencies despite the provided stabilization mechanisms. Finally, the current release does not include native uncertainty estimation or automatic hyperparameter tuning; these remain open areas for future extension.

# Licensing and Availability
`OpenSR-SRGAN` is licensed under the Apache-2.0 license, with all source code stored at [ESAOpenSR/OpenSR-SRGAN](https://github.com/ESAOpenSR/SRGAN) Github repository. In the spirit of open science and collaboration, we encourage feature requests and updates, bug fixes and reports, as well as general questions and concerns via direct interaction with the repository. A reproducible notebook is permanently hosted on [Google Colab](https://colab.research.google.com/drive/16W0FWr6py1J8P4po7JbNDMaepHUM97yL?usp=sharing).

# Acknowledgement
This work has been supported by the European Space Agency (ESA) $\Phi$-Lab, within the framework of the ['Explainable AI: Application to Trustworthy Super-Resolution (OpenSR)'](https://eo4society.esa.int/projects/opensr/) Project.

# Appendix
## Appendix A – Architecture and Training Components

**Table A1. Implemented generator types and their characteristics.** {#tbl:arch}

| **Generator Type** | **Description** |
|:-------------------|:----------------|
| `res`  [@ledig2017photo] | SRResNet generator using residual blocks without batch normalization. Stable and effective for content pretraining. |
| `rcab` [@rcab] | Residual Channel Attention Blocks. Adds channel-wise reweighting to enhance textures and small structures. |
| `rrdb` [@rrdb] | Residual-in-Residual Dense Blocks (RRDB) as in ESRGAN. Deep structure with dense connections, improving detail sharpness. |
| `lka`  [@lka] | Large-Kernel Attention blocks. Capture wide spatial context, beneficial for structured RS patterns (e.g., fields, roads). |
| `esrgan` [@rrdb] | Full ESRGAN generator with configurable RRDB count, growth channels, and residual scaling. |
| `cgan` [@cgan]| Stochastic Conditional Generator with `NoiseResBlock`. |


**Table A2. Implemented discriminator types and their purposes.** {#tbl:disc}

| **Discriminator Type** | **Description** |
|:-----------------------|:----------------|
| `standard` [@ledig2017photo] | A global SR-GAN-style CNN discriminator that judges the overall realism of the full image. Promotes coherent global structure. |
| `patchgan` [@patchgan] | A PatchGAN discriminator that outputs patch-level predictions. Focuses on local realism and texture detail. Patch size is implicitly controlled by network depth (`n_blocks`). |
| `esrgan` [@rrdb] | ESRGAN discriminator with configurable base channels and linear head size to complement RRDB generators. |


**Table A3. Implemented training features for stable adversarial optimization.** {#tbl:train}

| **Feature** | **Description** |
|:-------------|:----------------|
| `pretrain_g_only` | Trains only the generator (content losses) for a specified number of steps (`g_pretrain_steps`) before enabling the adversarial loss. |
| `adv_loss_ramp_steps` | Gradually increases the weight of the adversarial loss from 0 to the maximum value (`adv_loss_beta`), improving stability. |
| `label_smoothing` | Applies soft labels (e.g., 0.9 for real) to stabilize the discriminator and reduce overconfidence. |
| `g_warmup_steps`, `g_warmup_type` | Warmup schedule for the generator’s learning rate, linear or cosine, ensuring smooth optimizer convergence. |
| `EMA.enabled` | Enables Exponential Moving Average tracking of generator weights for smoother validation and inference outputs. |
| TTUR LRs (`optim_g_lr`, `optim_d_lr`) | Two-time-scale update rule with discriminator LR defaulting to a slower schedule than the generator to maintain balance. |
| Adam betas/epsilon (`betas`, `eps`) | GAN-friendly defaults (0.0, 0.99) and $10^{-7}$ avoid stale momentum and numerical noise during adversarial updates. |
| Weight-decay exclusions | Normalization and bias parameters are automatically removed from decay groups so regularization targets convolutional kernels only. |
| Plateau scheduler controls (`cooldown`, `min_lr`) | `ReduceLROnPlateau` schedulers for *G* and *D* now support cooldown periods and minimum learning-rate floors. |
| `gradient_clip_val` | Optional global-norm clipping applied after every optimizer step to suppress discriminator-induced spikes. |
| `Training.gpus` | Enables distributed data-parallel training when multiple GPU indices are listed, scaling training efficiently via PyTorch Lightning. |


**Table A4. Supported loss components and configuration parameters.** {#tbl:loss}

| **Loss Type** | **Description** |
|:---------------|:----------------|
| L1 Loss | Pixel-wise reconstruction loss using the L1 norm; maintains global content and brightness consistency. |
| SAM Loss | Spectral Angle Mapper; penalizes angular differences between spectral vectors of predicted and true pixels, preserving spectral fidelity. |
| Perceptual Loss | Feature-space loss using pre-trained VGG19 or LPIPS metrics; improves perceptual quality and texture realism. |
| TV Loss | Total Variation regularizer; encourages spatial smoothness and reduces noise or artifacts. |
| Adversarial Loss | Binary cross-entropy loss on discriminator predictions; drives realism and high-frequency texture generation. |


## Appendix B – Internal Metrics

During training, scalar metrics are continuously logged in **Weights & Biases**. These indicators quantify loss dynamics, adversarial balance, and stability. Table B1 summarises the most relevant internal metrics tracked by *OpenSR-SRGAN*.

**Table B1. Key internal metrics tracked during training and validation.** {#tbl:metrics}

| **Metric** | **Description and Expected Behaviour** |
|:-----------|:--------------------------------------|
| `training/`<br/>`pretrain_phase` | Binary flag indicating whether generator-only warm-up is active. Remains 1 during pretraining and switches to 0 once adversarial learning begins. |
| `discriminator/`<br/>`adversarial_loss` | Binary cross-entropy loss separating real HR from generated SR samples. Decreases below ~0.7 during stable co-training; large oscillations may indicate imbalance. |
| `discriminator/`<br/>`D(y)_prob` | Mean discriminator confidence that ground-truth HR inputs are real. Should rise toward 0.8–1.0 and stay high when *D* is healthy. |
| `discriminator/`<br/>`D(G(x))_prob` | Mean discriminator confidence that generated SR outputs are real. Starts near 0 and climbs toward 0.4–0.6 as *G* improves realism. |
| `generator/content_loss` | Weighted content component of the generator objective (e.g., L1 or spectral loss). Dominant during pretraining; gradually decreases over time. |
| `generator/`<br/>`total_loss` | Full generator objective combining content and adversarial terms. Tracks `content_loss` early, then stabilises once the adversarial weight ramps up. |
| `training/`<br/>`adv_loss_weight` | Current adversarial weight applied to the generator loss. Stays at 0 during pretrain and linearly ramps to its configured maximum value. |
| `validation/`<br/>`DISC_adversarial_loss` | Discriminator loss on validation batches. Should roughly mirror the training curve; strong divergence may signal overfitting or instability. |



## Appendix C – Experimental Configuration and Quantitative Results

This appendix provides detailed configurations, qualitative previews, and quantitative results for two representative experiments with *OpenSR-SRGAN*.


### Experiment 1 – 4× RGB Super-Resolution on SEN2NAIP

In the first experiment, a **residual channel-attention (RCAB)** generator is trained on the SEN2NAIP dataset [@sen2naip].  
The model maps Sentinel-2 RGB-NIR patches at 10 m to NAIP RGB-NIR targets at 2.5 m (4× upscaling).  
The training objective combines **L1 + LPIPS + adversarial** terms, with generator-only warm-up and gradual adversarial ramp-up.  
EMA (β = 0.999) stabilises validation.  

**Hardware:** Dual A100 (DDP, mixed precision).  
**Performance:** ~31 dB PSNR, SSIM$\approx$ 0.8, low SAM, strong perceptual quality.  

Qualitative results show sharper fields, buildings, and roads compared to bicubic upsampling, with minimal spectral distortion (Figure C1).

![False-color visual comparison for 4× RGB SR on SEN2NAIP. Left to right: LR input, model output, HR reference.](figures/rgb_example.png){#fig:exp1}

**Table C1. Configuration summary for the SEN2NAIP RGB experiment.** {#tbl:exp1config}

| **Parameter** | **Setting** |
|:---------------|:------------|
| Dataset | SEN2NAIP (Sentinel-2 → NAIP RGB-NIR, 4×) |
| Generator | RCAB-based SRResNet variant (`block_type=rcab`, 16 blocks, 96 channels) |
| Discriminator | Standard global discriminator (SRGAN-style) |
| Loss composition | L1 (1.0) + Perceptual (0.2) + Adversarial (0.01) |
| Training schedule | Pretrain 150k steps; Ramp 50k steps; EMA β = 0.999 |
| Hardware | Dual A100 (DDP), 16-bit precision |

**Table C2. Validation performance of the SEN2NAIP RGB experiment (4×).** {#tbl:exp1results}

| **Model** | **PSNR↑** | **SSIM↑** | **LPIPS↑** | **SAM↓** |
|:-----------|:----------:|:----------:|:-----------:|:----------:|
| RCAB–SRResNet + Standard Discriminator | 31.45 | 0.81 | 0.82 | 0.069 |


### Experiment 2 – 8× 6-Band SWIR Sentinel-2 Super-Resolution

This experiment targets **six 20 m Sentinel-2 bands** (including SWIR) using a synthetic LR–HR setup:  
the original 20 m image serves as HR, and 160 m downsampled inputs as LR.  
The generator (SRResNet backbone, 32 blocks, 96 channels, scale = 8) is trained with **L1 + SAM + adversarial** losses.  
A PatchGAN discriminator ensures local realism; EMA is disabled.

**Performance:** mid-20 dB PSNR, SSIM $\approx$ 0.7–0.75, low SAM values.  
Figure C2 shows sharper edges and preserved spectral structure relative to bicubic interpolation.

![Visual comparison for 8× multispectral SR (6-band Sentinel-2). Left to right: LR input, model output, HR reference.](figures/swir_example.png){#fig:exp2}

**Table C3. Configuration summary for the 6-band Sentinel-2 experiment.**

| **Parameter** | **Setting** |
|:---------------|:------------|
| Dataset | Sentinel-2 6-band subset (160 m → 20 m, 8× upscaling) |
| Generator | SRResNet backbone (`block_type=res`, 32 blocks, 96 channels, scale = 8) |
| Discriminator | PatchGAN (`n_blocks=4`, patch $\approx$ 70×70) |
| Loss composition | L1 (1.0) + SAM (0.2) + Adversarial (0.005) |
| Training schedule | Pretrain 100k steps; Ramp 40k steps; EMA disabled |
| Hardware | Dual A100 GPU, 32-bit precision |

**Table C4. Validation performance of the 6-band Sentinel-2 experiment (8×).**

| **Model** | **PSNR↑** | **SSIM↑** | **LPIPS↑** | **SAM↓** |
|:-----------|:----------:|:----------:|:-----------:|:----------:|
| SRResNet (6-band) + PatchGAN Discriminator | 26.65 | 0.74 | 0.80 | 0.091 |

