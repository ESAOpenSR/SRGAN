# Training guideline

Super-resolution GANs can be temperamental. This guideline distils lessons learned from training across remote sensing, medical
imaging, microscopy, and consumer photography. Use it as a playbook to stabilise and accelerate your own experiments.

## 1. Start with reconstruction losses

Before introducing the discriminator, pretrain the generator using reconstruction objectives only (L1/L2/SSIM). Set
`Training.Stability.pretrain_g_only: true` and adjust `g_pretrain_steps` based on your dataset size. For high-noise domains
(CT, SAR), extend the pretraining phase to ensure the generator captures signal statistics.

## 2. Normalise thoughtfully

* Compute per-channel statistics that reflect your modality. For CT, clip Hounsfield units; for multispectral data, respect known
  reflectance ranges.
* Align LR and HR normalisation to avoid scale drift.
* When statistics drift between training and deployment, recompute them or enable adaptive histogram matching.

## 3. Ramp in adversarial pressure

Adversarial losses are powerful but destabilising. Use `Training.Losses.adv_warmup` to slowly increase `adv_loss_beta`. Cosine
ramps often feel smoother than linear ramps. Keep an eye on discriminator loss oscillations—if they explode, reduce the ramp
speed or increase `d_update_interval`.

## 4. Choose perceptual channels carefully

Perceptual networks (VGG/LPIPS) were trained on RGB. When working with medical or hyperspectral data:

* Use `perceptual_channels` to select relevant bands (e.g. [0, 1, 2] for RGB composites, [0] for single-contrast MRI).
* Consider training your own feature extractor on modality-specific data and plugging it in via the registry.
* Balance perceptual weights with structural metrics like SSIM or SAM to avoid hallucinating features.

## 5. Monitor gradient statistics

Enable gradient logging via `Training.Logging.log_gradients: true`. Track:

* **Generator gradients** – Spikes may indicate adversarial instability or misconfigured normalisation.
* **Discriminator gradients** – If they vanish, increase `adv_loss_beta` or reduce EMA smoothing.
* **Gradient clipping** – Adjust `Training.Stability.gradient_clip_val` to keep updates within a safe range.

## 6. Use EMA for evaluation

EMA checkpoints often deliver the best validation visuals. Enable `Training.ema` with decay 0.995–0.9999. Swap between raw and
EMA weights during validation to understand their trade-offs.

## 7. Curriculum learning

When datasets contain wide resolution ranges or complex structures, start with smaller crops and gradually increase `crop_size`.
This curriculum helps the generator learn coarse structure before focusing on fine textures.

## 8. Domain-specific augmentations

* **Medical** – Random bias field, intensity scaling, or elastic deformations (using MONAI transforms).
* **Remote sensing** – Radiometric jitter, sun-angle simulation, or cloud masking.
* **Microscopy** – Photobleaching simulation, Poisson noise, or channel dropout.
* **Consumer** – JPEG artefact simulation, colour jitter, random erasing.

Augmentations should mirror the variability seen in deployment environments.

## 9. Evaluate with domain metrics

Beyond PSNR/SSIM, compute modality-aware metrics:

* **Medical** – Structural similarity on organ masks, segmentation overlap, clinical scoring.
* **Remote sensing** – Spectral angle mapper, vegetation index consistency, change detection accuracy.
* **Microscopy** – F1 scores on downstream segmentation or object detection tasks.
* **Consumer** – User studies, NR-IQA scores, or downstream detection accuracy.

## 10. Document everything

Log configuration files, random seeds, dataset hashes, and environment details. Share them alongside results to support
reproducibility across teams and industries.

---

Following these guidelines will help you train robust, high-quality SR models for any domain your organisation cares about.
