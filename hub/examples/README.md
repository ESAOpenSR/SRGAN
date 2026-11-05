# SR-GAN examples

These small examples use the supported release layouts in this repository.  Install
the matching package revision and download the examples alongside the model files:

```bash
python -m pip install 'opensr-srgan @ git+https://github.com/ESAOpenSR/SRGAN.git@dev'
hf download simon-donike/SR-GAN --include 'examples/*' --local-dir sr-gan-release
cd sr-gan-release/examples
```

Pass `--revision <commit-or-tag>` to every command if you want
`load_inference_model` to resolve a particular Hub revision.  The packaged
presets already pin their supported release by default, so the flag is optional
for the standard release.  The resume script resolves its requested revision to
one immutable commit before downloading its three linked artifacts and records
that commit in its output directory.

## Tensor inference

`infer_tensor.py` accepts a saved PyTorch tensor with shape `BCHW` (or `CHW`) and
saves a `BCHW` tensor.  It also accepts a tuple/list such as `(lr, hr)` and a
dictionary containing a tensor under `lr`, `lr_imgs`, `input`, `image`, or
`x`; in a tuple/list it uses the first tensor, which is conventionally the LR
sample.

```bash
python infer_tensor.py --preset SEN2-SR --device cpu --input lr_b04_b03_b02_b08.pt --output sr.pt
python infer_tensor.py --preset RGB-NIR --device cuda --input lr.pt --output sr.pt
python infer_tensor.py --preset SWIR --device cuda --input swir_lr.pt --output swir_sr.pt
```

For `SEN2-SR` and `RGB-NIR`, channels must be Sentinel-2 `B04, B03, B02, B08`.
The `SWIR` preset expects `B05, B06, B07, B8A, B11, B12` in that order.  The
normal mode calls `predict_step`: raw Sentinel-2 reflectance/DN values in the
usual `0..10000` range produce raw-DN output.  Inputs already in `0..1` are
recognized as normalized reflectance and retain that normalized convention in
the output.  The script does not clip the returned values.  The published
sample batch is a mapping with `lr` and `hr` tensors; passing it directly uses
its normalized `lr` entry.

`--raw` is deliberately a network-only mode.  It calls `model.forward`; supply
the model-ready normalized reflectance values (normally `0..1`) yourself.  Its
saved output remains in that network convention and skips `predict_step`
normalization, denormalization, and post-processing.

`--smoke` creates a small zero tensor to check installation and tensor shapes.
It is not imagery and must not be used to assess image quality.

```bash
python infer_tensor.py --preset SEN2-SR --device cpu --smoke --output smoke_sr.pt
python infer_tensor.py --preset SEN2-SR --input normalized.pt --raw --output normalized_sr.pt
```

## A small GeoTIFF

`infer_geotiff.py` reads one small, already co-registered raster into memory.
It is useful for a cutout, not for a whole Sentinel-2 scene: it does not tile,
overlap, or manage memory for large rasters.

Choose source band indices explicitly with `--bands`; indices are one-based as
in Rasterio/GDAL.  The default is `1,2,...` only when the input raster was
already stacked in the model's required order.

```bash
# A four-band GeoTIFF stacked B04,B03,B02,B08 in bands 1,2,3,4.
python infer_geotiff.py --preset SEN2-SR --input cutout.tif --bands 1,2,3,4 --output cutout_sr.tif

# A six-band GeoTIFF stacked B05,B06,B07,B8A,B11,B12 in bands 3,5,2,1,7,4.
python infer_geotiff.py --preset SWIR --input cutout.tif --bands 3,5,2,1,7,4 --output cutout_swir_sr.tif
```

The output uses float32 values from `predict_step`, retains the source CRS, and
derives its affine transform from the actual output shape.  With raw-DN input,
the output is raw-DN; a `0..1` input retains normalized-reflectance output.
This makes output pixel size the input pixel size divided by the model's actual
scale factor (4 for `SEN2-SR`/`RGB-NIR`, 8 for the legacy `SWIR` preset).

## Continue the Sen2 SR run

The full Sen2 SR checkpoint contains optimizer, scheduler, and EMA state.  The
resume example downloads the canonical training config, split, and checkpoint
from the Hub; it never fetches data itself.  Point `--dataset` to the local
SEN2NAIP/TACO data file expected by the recipe.

```bash
python resume_training.py \
  --dataset /data/SEN2NAIPv2/sen2naipv2-crosssensor.taco \
  --output-dir ./sen2-sr-resume \
  --max-steps 328584
```

The command above only writes a fully resolved `config.yaml` and copies the
downloaded release inputs into the new output directory.  Inspect it before
training, then launch that prepared configuration directly:

```bash
python -m opensr_srgan.train --config ./sen2-sr-resume/config.yaml
```

`--max-steps` is an absolute Lightning optimizer-step limit, so it must exceed
the checkpoint's recorded step (308584) for a continuation to perform updates.
Install the packaged data extra before a resume, then use a fresh output
directory for each attempt:

```bash
python -m pip install 'opensr-srgan[sen2naip] @ git+https://github.com/ESAOpenSR/SRGAN.git@dev'
```

Use `--run` only when preparing and launching in one command against a fresh
output directory.
