# Example results

This page showcases qualitative and quantitative examples from different domains to illustrate what OpenSR GAN Lab can deliver.
Use these as inspiration when designing your own experiments.

## Medical imaging

* **MRI (T1-weighted, ×4):** Generator `rrdb`, perceptual loss LPIPS on channel 0, SAM weight 0.02. Produces crisp cortical
  boundaries and preserves contrast without introducing halo artefacts.
* **CT (lung, ×2):** Generator `rcab`, adversarial ramp over 15k steps, histogram matching enabled. Results maintain HU fidelity
  for radiologist review.

## Remote sensing

* **Sentinel-2 RGB-NIR (×4):** Generator `lka`, discriminator `patchgan`. Spectral angle mapper ensures vegetation indices remain
  stable. Suitable for agritech analytics.
* **SWIR minerals (×3):** Custom generator with grouped convolutions and channel attention. Perceptual loss limited to bands 0–5
  to respect domain-specific features.

## Microscopy

* **Fluorescence confocal (3D, ×2):** 3D RRDB blocks with volumetric tiling. Model trained with total-variation regularisation
  to reduce ringing while enhancing fine structures.
* **Histopathology WSI (×4):** PatchGAN discriminator with feature matching. Results sharpen cellular boundaries and glandular
  textures for pathologist review.

## Consumer photography

* **Compressed JPEG (×4):** ESRGAN baseline with stochastic residual blocks. Removes compression artefacts and restores detail in
  handheld shots.
* **Drone imagery (×2):** RCAB generator with Weights & Biases logging for on-site monitoring. Handles mixed lighting conditions.

## Quantitative benchmarks

| Domain | Dataset | Scale | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Medical | FastMRI knee | ×4 | 30.2 | 0.92 | 0.082 | LPIPS restricted to T1/T2 bands, EMA checkpoint |
| Remote sensing | SEN2NEON RGBNIR | ×4 | 28.7 | 0.89 | 0.115 | Histogram matching + SAM 0.05 |
| Microscopy | DeepZoom fluorescence | ×2 | 32.9 | 0.94 | 0.071 | 3D training with tiling stride 128 |
| Consumer | DIV2K + Flickr2K | ×4 | 29.5 | 0.83 | 0.145 | Stochastic generator ensemble |

> Metrics are indicative and depend on configuration details, normalisation statistics, and training duration. Use them as
> starting points rather than hard baselines.

## Sharing your results

1. Create a pull request adding your experiment description and metrics to this page.
2. Include configuration files, dataset references, and evaluation scripts where possible.
3. If you can share checkpoints, upload them to the Hugging Face Hub and link them here.

Community contributions help expand the portfolio of supported modalities and encourage reproducible research.
