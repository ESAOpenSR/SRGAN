# HPC inference

The `srgan-hpc` launcher runs SRGAN inference on Slurm clusters for larger Sentinel-2 jobs. It stages cutouts with `cubo`, submits worker jobs with `sbatch`, runs the RGB-NIR preset, the SWIR preset, or both, and writes the run manifests, logs, metadata, and GeoTIFF outputs into one run directory.

This workflow was contributed by GitHub user [@mak8427](https://github.com/mak8427). Thanks for the contribution.

## Install

Install the package with the HPC extras from a checkout of this repository:

```bash
pip install -e .[hpc]
```

This exposes the `srgan-hpc` command.

## Configure a cluster run

Start from the default runtime file and edit it for your cluster:

```bash
cp deployment/configs/runtime.default.yaml deployment/configs/my-cluster.yaml
srgan-hpc validate-config --config deployment/configs/my-cluster.yaml
```

At minimum, update:

* `output_root`: where run folders and outputs should be written.
* `environment`: Python, module, or Conda settings needed by worker jobs.
* `slurm`: partition, GPU, CPU, memory, walltime, account, and QoS settings.
* `mode`: `rgbnir`, `swir`, or `fused`.

The bundled defaults use the published RGB-NIR and SWIR presets. Set `rgbnir.model.config_path`, `rgbnir.model.checkpoint_path`, `swir.model.config_path`, or `swir.model.checkpoint_path` only when you want to run custom weights.

## Submit a quick patch job

Use `--dry-run` first to check the generated Slurm command and run directory without submitting work:

```bash
srgan-hpc submit patch --config deployment/configs/my-cluster.yaml --lat 52.5200 --lon 13.4050 --start-date 2025-07-01 --end-date 2025-07-03 --dry-run
```

When the dry run looks right, repeat the command without `--dry-run` to submit the job.

For larger areas, use `srgan-hpc submit grid` with two bounding coordinates, or `srgan-hpc submit aoi` with a shapefile path. The AOI command accepts either a `.shp` file or a directory containing one shapefile with its sidecar files.

## How it works

The submit command creates a timestamped run directory under `output_root`, writes resolved config and patch manifests, and submits a Slurm job or array. Each worker loads the configured environment, stages the Sentinel-2 cutout, loads the selected SRGAN model, and calls the tiled `opensr-utils` inference pipeline with the configured window size, overlap, batch size, and GPU settings.

After jobs finish, use:

```bash
srgan-hpc status --run-dir /path/to/run
srgan-hpc collect --run-dir /path/to/run --dest /path/to/outputs
```

For bounding-box delivery, `srgan-hpc deliver-bbox` can merge patch outputs and write clipped GeoTIFFs for GIS workflows.
