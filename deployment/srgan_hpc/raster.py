from __future__ import annotations

import os
import re
from math import floor
from pathlib import Path

import numpy as np


def ensure_proj_env() -> None:
    if os.environ.get("PROJ_LIB"):
        return

    from pyproj import datadir

    proj_dir = datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_dir
    os.environ["PROJ_DATA"] = proj_dir


def guess_utm_epsg(lat: float, lon: float) -> int:
    lat = max(min(lat, 84.0), -80.0)
    zone = int(floor((lon + 180.0) / 6.0)) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def parse_epsg(text: str, lat: float, lon: float) -> int:
    match = re.search(r"(\d{4,5})", text)
    if match:
        return int(match.group(1))
    return guess_utm_epsg(lat, lon)


def _as_scalar(value: object) -> float:
    if hasattr(value, "compute"):
        value = value.compute()
    return float(value)


def scale_to_uint16(data: np.ndarray) -> np.ndarray:
    if data.dtype.kind == "f":
        finite_mask = np.isfinite(data)
        safe_data = np.where(finite_mask, data, 0.0)
        if _as_scalar(safe_data.max()) <= 1.1:
            safe_data = safe_data * 10000.0
        return np.clip(safe_data, 0, 10000).astype("uint16")
    return data.astype("uint16")


def compute_centroid_lat_lon(tif_path: Path) -> tuple[float, float]:
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Dataset at {tif_path} lacks a CRS")
        center_row = src.height / 2.0
        center_col = src.width / 2.0
        x, y = src.transform * (center_col, center_row)
        transformed = warp_transform(src.crs, "EPSG:4326", [x], [y])
        lon = transformed[0]
        lat = transformed[1]
        return float(lat[0]), float(lon[0])


def _write_band_names(dataset, band_names: list[str]) -> None:
    if len(band_names) == dataset.count:
        for index, name in enumerate(band_names, start=1):
            dataset.set_band_description(index, name)
    dataset.update_tags(band_names=",".join(band_names))


def compress_geotiff(src_path: Path, dest_path: Path, band_names: list[str] | None = None) -> Path:
    from rasterio.shutil import copy as rio_copy

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    rio_copy(
        src_path,
        dest_path,
        driver="GTiff",
        COMPRESS="ZSTD",
        ZSTD_LEVEL=22,
        PREDICTOR=2,
        TILED=True,
        BLOCKXSIZE=512,
        BLOCKYSIZE=512,
        BIGTIFF="YES",
        NUM_THREADS="ALL_CPUS",
    )
    if band_names is not None:
        import rasterio

        with rasterio.open(dest_path, "r+") as dst:
            _write_band_names(dst, band_names)
    return dest_path


def stack_geotiffs(
    *,
    reference_path: Path,
    secondary_path: Path,
    output_path: Path,
    band_names: list[str],
) -> Path:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(reference_path) as ref, rasterio.open(secondary_path) as secondary:
        profile = ref.profile.copy()
        secondary_data = np.empty((secondary.count, ref.height, ref.width), dtype=ref.dtypes[0])
        for band_index in range(secondary.count):
            reproject(
                source=rasterio.band(secondary, band_index + 1),
                destination=secondary_data[band_index],
                src_transform=secondary.transform,
                src_crs=secondary.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear,
            )

        reference_data = ref.read()
        stacked = np.concatenate([reference_data, secondary_data], axis=0)
        profile.update(count=stacked.shape[0], dtype=stacked.dtype, BIGTIFF="YES")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(stacked)
            _write_band_names(dst, band_names)
    return output_path


def raster_validity_stats(tif_path: Path) -> dict[str, int]:
    import rasterio

    with rasterio.open(tif_path) as src:
        data = src.read(masked=True)

    if data.size == 0:
        return {"total_pixels": 0, "valid_pixels": 0, "nonzero_pixels": 0}

    values = np.asarray(np.ma.getdata(data))
    validity_mask = np.isfinite(values)
    if np.ma.isMaskedArray(data):
        validity_mask &= ~np.ma.getmaskarray(data)

    total_pixels = int(values.shape[-2] * values.shape[-1])
    valid_pixels = int(validity_mask.any(axis=0).sum())
    nonzero_pixels = int((validity_mask & (values != 0)).any(axis=0).sum())
    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nonzero_pixels": nonzero_pixels,
    }
