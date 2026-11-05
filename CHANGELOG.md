# Release notes

## 1.1.0 — 2026-11-05

### Pretrained Sen2 SR

- Added the `SEN2-SR` preset for 4× Sentinel-2 RGB/NIR super-resolution, using
  bands B04/B03/B02/B08. Weights, training recipes, and dataset splits are hosted
  on [Hugging Face](https://huggingface.co/simon-donike/SR-GAN).
- Added shuffled-band VGG/local-Gram losses, global relativistic averages,
  normalized lazy R1, and RGB/NIR validation metrics.
- Added a validated EMA-only checkpoint exporter and pinned the pretrained
  configs and weights to an immutable Hugging Face revision.

### Models and prediction

- Added the SwinIR generator and model configuration.
- Added configurable measurement-consistency back-projection. Existing configs
  without a postprocessing section keep histogram matching.
- Fixed inference examples to move the model explicitly with `.to(device)`;
  `map_location` controls checkpoint deserialization, not model placement.

### Hub layout

- Updated loaders for the versioned Hugging Face layout. Model cards, examples,
  checkpoints, and provenance are maintained on the Hub.
- Kept all historical artifact URLs and their contents. `RGB-NIR`, `SWIR`, and
  `SEN2-SR` remain valid preset names.
- Added MIT licensing for the Hub model weights, cards, and authored examples.
  The Python package remains Apache-2.0; archived source notices and sample-data
  terms are preserved.

### Upgrade notes

Install the release with `pip install opensr-srgan==1.1.0`. For SEN2NAIP training,
install `pip install 'opensr-srgan[sen2naip]==1.1.0'`.

SEN2NAIP loading without an explicit split manifest now uses a seeded shuffled
split instead of sequential record ranges. This changes train/validation
membership for those configs. Keep an explicit split manifest when continuing
an existing experiment. The Sen2 SR model card on Hugging Face includes its
exact spatial split.

The Hugging Face checkpoint release remains `v1`; the package version change
does not change the published weights.
