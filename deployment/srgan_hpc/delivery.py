from __future__ import annotations

from pathlib import Path
from typing import Any

from deployment.srgan_hpc.manifests import read_yaml, write_json
from deployment.srgan_hpc.naming import fused_output_name


def _run_dirs(run_root: Path) -> list[Path]:
    if (run_root / "run_manifest.yaml").exists():
        return [run_root]
    return sorted(path for path in run_root.iterdir() if path.is_dir() and (path / "run_manifest.yaml").exists())


def _date_label(run_dir: Path) -> str:
    manifest = read_yaml(run_dir / "run_manifest.yaml")
    return str(manifest.get("start_date") or run_dir.name)


def _patch_outputs(run_dir: Path, output_name: str) -> list[Path]:
    # Grid/AOI runs automatically collect outputs by moving them from each
    # patch directory into ``collected/<patch_id>``. Support both layouts and
    # prefer the original patch output when both are present (for example after
    # an interrupted or manually copied collection).
    outputs_by_patch = {
        path.parent.name: path
        for path in run_dir.glob(f"collected/*/{output_name}")
    }
    outputs_by_patch.update(
        {
            path.parent.parent.name: path
            for path in run_dir.glob(f"patches/*/outputs/{output_name}")
        }
    )
    outputs = [outputs_by_patch[key] for key in sorted(outputs_by_patch)]
    if not outputs:
        raise FileNotFoundError(f"No {output_name} files found under {run_dir}")
    return outputs


def merge_and_clip_bbox(
    *,
    input_paths: list[Path],
    output_path: Path,
    bbox: tuple[float, float, float, float],
    nodata: int = 0,
) -> Path:
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds, transform_geom

    west, south, east, north = bbox
    sources = [rasterio.open(path) for path in input_paths]
    try:
        crs = sources[0].crs
        if crs is None:
            raise ValueError(f"Input lacks CRS: {input_paths[0]}")
        if any(source.crs != crs for source in sources):
            raise ValueError("All input rasters must share the same CRS")

        dst_bounds = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
        data, transform = merge(sources, bounds=dst_bounds, nodata=nodata)
        bbox_geom = {
            "type": "Polygon",
            "coordinates": [[
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]],
        }
        dst_geom = transform_geom("EPSG:4326", crs, bbox_geom)
        keep = geometry_mask([dst_geom], out_shape=data.shape[-2:], transform=transform, invert=True)
        data[:, ~keep] = nodata

        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=data.shape[-2],
            width=data.shape[-1],
            count=data.shape[0],
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress="ZSTD",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            descriptions = sources[0].descriptions
            if descriptions and len(descriptions) == data.shape[0]:
                for band_index, description in enumerate(descriptions, start=1):
                    if description:
                        dst.set_band_description(band_index, description)
            if sources[0].tags().get("band_names"):
                dst.update_tags(band_names=sources[0].tags()["band_names"])
    finally:
        for source in sources:
            source.close()
    return output_path


def deliver_bbox_outputs(
    *,
    run_root: Path,
    bbox: tuple[float, float, float, float],
    destination: Path | None = None,
    output_name: str = fused_output_name(),
) -> tuple[Path, list[dict[str, Any]]]:
    destination = destination or run_root / "delivery_clipped"
    destination.mkdir(parents=True, exist_ok=True)

    delivered: list[dict[str, Any]] = []
    for run_dir in _run_dirs(run_root):
        date_label = _date_label(run_dir)
        inputs = _patch_outputs(run_dir, output_name)
        output_path = destination / f"{run_dir.name}_{output_name.replace('.tif', '')}_clipped.tif"
        merge_and_clip_bbox(input_paths=inputs, output_path=output_path, bbox=bbox)
        delivered.append(
            {
                "run_dir": str(run_dir),
                "date": date_label,
                "inputs": [str(path) for path in inputs],
                "output": str(output_path),
            }
        )

    write_json(destination / "delivery_manifest.json", {"bbox": list(bbox), "outputs": delivered})
    return destination, delivered
