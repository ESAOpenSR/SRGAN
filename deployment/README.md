# SRGAN HPC Launcher

This directory contains an installable Slurm launcher for running `opensr-srgan` over `cubo`-staged Sentinel-2 cutouts.

## What it does

- stages Sentinel-2 cutouts with `cubo`
- submits single-patch, grid, or AOI shapefile runs through `sbatch`
- runs SRGAN RGB-NIR, SRGAN SWIR, or both
- optionally stacks RGB-NIR and SWIR outputs into one 10-band GeoTIFF
- writes manifests, logs, metadata, and outputs into one run directory

## Install

```bash
pip install -e .[hpc]
```

This exposes the `srgan-hpc` CLI.

## Example commands

```bash
srgan-hpc validate-config --config deployment/configs/runtime.default.yaml

srgan-hpc submit patch \
  --config deployment/configs/runtime.default.yaml \
  --lat 52.5200 \
  --lon 13.4050 \
  --start-date 2025-07-01 \
  --end-date 2025-07-03 \
  --dry-run

srgan-hpc submit grid \
  --config deployment/configs/runtime.default.yaml \
  --lat1 52.3 --lon1 12.9 \
  --lat2 52.7 --lon2 13.8 \
  --start-date 2025-07-01 \
  --end-date 2025-07-03 \
  --dry-run

srgan-hpc submit aoi \
  --config deployment/configs/runtime.default.yaml \
  --aoi-path /data/berlin_aoi \
  --start-date 2025-07-01 \
  --end-date 2025-07-03 \
  --dry-run
```

## Runtime Modes

- `mode: rgbnir` stages `B04, B03, B02, B08` at 10 m and runs the RGB-NIR preset.
- `mode: swir` stages `B05, B06, B07, B8A, B11, B12` at 20 m and runs the SWIR preset.
- `mode: fused` runs both and writes `fused_sr.tif` with band order `B04, B03, B02, B08, B05, B06, B07, B8A, B11, B12`.

AOI submission accepts either a `.shp` file or a directory containing exactly one `.shp`; sidecar files like `.shx`, `.dbf`, and `.prj` must sit alongside it.
