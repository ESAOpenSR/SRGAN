from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from deployment.srgan_hpc.config import StagingConfig
from deployment.srgan_hpc.manifests import write_json
from deployment.srgan_hpc.raster import ensure_proj_env, parse_epsg, scale_to_uint16

LOGGER = logging.getLogger("srgan-hpc")


class SkipTileError(RuntimeError):
    def __init__(self, reason: str, *, details: dict[str, int] | None = None) -> None:
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


def _valid_pixel_mask(data: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(data)
    return (finite_mask & (data != 0)).any(axis=0)


def _coverage_fraction(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    return float(_valid_pixel_mask(data).mean())


def _center_window(data: np.ndarray) -> np.ndarray:
    height, width = data.shape[-2:]
    row_start = height // 4
    row_end = height - row_start
    col_start = width // 4
    col_end = width - col_start
    return data[..., row_start:row_end, col_start:col_end]


def candidate_coverage_report(data: np.ndarray) -> dict[str, float]:
    return {
        "full_nonzero_fraction": _coverage_fraction(data),
        "center_nonzero_fraction": _coverage_fraction(_center_window(data)),
    }


def order_candidates_by_coverage(data: np.ndarray) -> list[int]:
    reports = [candidate_coverage_report(data[index]) for index in range(data.shape[0])]
    return sorted(
        range(data.shape[0]),
        key=lambda index: (
            reports[index]["center_nonzero_fraction"],
            reports[index]["full_nonzero_fraction"],
        ),
        reverse=True,
    )


def mosaic_candidates_by_valid_pixels(data: np.ndarray, order: list[int]) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError("Expected candidate data with shape (items, bands, y, x)")
    if not order:
        raise ValueError("Candidate order must not be empty")

    mosaic = np.zeros_like(data[order[0]])
    filled = np.zeros(data.shape[-2:], dtype=bool)
    for index in order:
        candidate = data[index]
        valid = _valid_pixel_mask(candidate)
        fill_mask = valid & ~filled
        if not fill_mask.any():
            continue
        mosaic[:, fill_mask] = candidate[:, fill_mask]
        filled |= fill_mask
    return mosaic


def _cube_validity_stats(cube) -> dict[str, int]:
    data = np.asarray(cube.data)
    if data.size == 0:
        return {"total_pixels": 0, "valid_pixels": 0, "nonzero_pixels": 0}

    total_pixels = int(data.shape[-2] * data.shape[-1])
    finite_mask = np.isfinite(data)
    valid_pixels = int(finite_mask.any(axis=0).sum())
    nonzero_pixels = int((finite_mask & (data != 0)).any(axis=0).sum())
    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nonzero_pixels": nonzero_pixels,
    }


def ensure_cube_has_valid_data(cube) -> dict[str, int]:
    stats = _cube_validity_stats(cube)
    if stats["total_pixels"] == 0:
        raise SkipTileError("empty_cube", details=stats)
    if stats["valid_pixels"] == 0:
        raise SkipTileError("all_nan_cube", details=stats)
    if stats["nonzero_pixels"] == 0:
        raise SkipTileError("all_zero_cube", details=stats)
    return stats


def is_retryable_staging_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if status_code == 429 or response_status == 429:
        return True

    message = str(exc).lower()
    markers = [
        "429",
        "too many requests",
        "rate limit",
        "rate-limit",
        "maximum allowed time",
        "request timed out",
        "timeout",
        "temporarily unavailable",
    ]
    return any(marker in message for marker in markers)


def is_rate_limit_error(exc: Exception) -> bool:
    return is_retryable_staging_error(exc)


def create_cube_with_retry(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    config: StagingConfig,
    bands: list[str],
    edge_size: int,
    resolution: int,
):
    try:
        import cubo
        import rioxarray  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'srgan-hpc staging requires optional dependencies. Install with `pip install "opensr-srgan[hpc]"`.'
        ) from exc

    for attempt, delay in enumerate(config.rate_limit_retry_delays_seconds, start=1):
        try:
            return cubo.create(
                lat=latitude,
                lon=longitude,
                collection=config.collection,
                bands=bands,
                start_date=start_date,
                end_date=end_date,
                edge_size=edge_size,
                resolution=resolution,
            )
        except Exception as exc:  # pragma: no cover
            if not (config.retry_on_rate_limit and is_retryable_staging_error(exc)):
                raise
            LOGGER.warning(
                "Retryable cubo staging error for lat=%s lon=%s: %s. Retrying in %s seconds (%s/%s).",
                latitude,
                longitude,
                exc,
                delay,
                attempt,
                len(config.rate_limit_retry_delays_seconds),
            )
            time.sleep(delay)

    return cubo.create(
        lat=latitude,
        lon=longitude,
        collection=config.collection,
        bands=bands,
        start_date=start_date,
        end_date=end_date,
        edge_size=edge_size,
        resolution=resolution,
    )


def _cube_item_label(cube, index: int) -> str:
    if "time" not in getattr(cube, "coords", {}):
        return str(index)
    try:
        return str(cube.coords["time"].values[index])
    except Exception:
        return str(index)


def _select_or_mosaic_time_items(cube, config: StagingConfig):
    diagnostics: dict[str, Any] = {"item_strategy": config.item_strategy}
    if "time" not in cube.dims:
        data = np.asarray(cube.data)
        diagnostics.update(
            {
                "candidate_count": 1,
                "selected_indices": [0],
                "final_full_nonzero_fraction": candidate_coverage_report(data)["full_nonzero_fraction"],
                "final_center_nonzero_fraction": candidate_coverage_report(data)["center_nonzero_fraction"],
            }
        )
        return cube, diagnostics

    cube = cube.transpose("time", "band", "y", "x")
    candidate_count = int(cube.sizes["time"])
    if candidate_count <= 0:
        raise SkipTileError("empty_cube", details={"total_pixels": 0, "valid_pixels": 0, "nonzero_pixels": 0})

    if config.item_strategy == "fixed_index":
        selected_index = config.image_index
        if selected_index >= candidate_count:
            raise IndexError(f"staging.image_index={selected_index} but cube has {candidate_count} items")
        selected = cube.isel(time=selected_index)
        data = np.asarray(selected.data)
        report = candidate_coverage_report(data)
        diagnostics.update(
            {
                "candidate_count": candidate_count,
                "selected_indices": [selected_index],
                "selected_labels": [_cube_item_label(cube, selected_index)],
                "candidate_reports": [{"index": selected_index, "label": _cube_item_label(cube, selected_index), **report}],
                "final_full_nonzero_fraction": report["full_nonzero_fraction"],
                "final_center_nonzero_fraction": report["center_nonzero_fraction"],
            }
        )
        return selected, diagnostics

    data = np.asarray(cube.data)
    order = order_candidates_by_coverage(data)
    candidate_reports = [
        {
            "index": index,
            "label": _cube_item_label(cube, index),
            **candidate_coverage_report(data[index]),
        }
        for index in range(candidate_count)
    ]
    mosaicked = mosaic_candidates_by_valid_pixels(data, order)
    final_report = candidate_coverage_report(mosaicked)
    selected = cube.isel(time=order[0]).copy(data=mosaicked)
    diagnostics.update(
        {
            "candidate_count": candidate_count,
            "selected_indices": order,
            "selected_labels": [_cube_item_label(cube, index) for index in order],
            "candidate_reports": candidate_reports,
            "final_full_nonzero_fraction": final_report["full_nonzero_fraction"],
            "final_center_nonzero_fraction": final_report["center_nonzero_fraction"],
        }
    )

    if final_report["center_nonzero_fraction"] < config.min_center_nonzero_fraction:
        raise SkipTileError(
            "low_center_coverage",
            details={
                "candidate_count": candidate_count,
                "center_nonzero_percent": int(round(final_report["center_nonzero_fraction"] * 100)),
            },
        )
    if final_report["full_nonzero_fraction"] < config.min_full_nonzero_fraction:
        LOGGER.warning(
            "Low full cutout coverage after mosaicking: %.3f < %.3f",
            final_report["full_nonzero_fraction"],
            config.min_full_nonzero_fraction,
        )
    return selected, diagnostics


def stage_cutout(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    config: StagingConfig,
    bands: list[str],
    edge_size: int,
    resolution: int,
    output_path: Path,
    metadata_path: Path | None = None,
) -> Path:
    ensure_proj_env()
    LOGGER.info(
        "staging cubo cutout lat=%s lon=%s start_date=%s end_date=%s output=%s",
        latitude,
        longitude,
        start_date,
        end_date,
        output_path,
    )
    cube = create_cube_with_retry(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        config=config,
        bands=bands,
        edge_size=edge_size,
        resolution=resolution,
    )
    cube, diagnostics = _select_or_mosaic_time_items(cube, config)
    cube = cube.transpose("band", "y", "x")
    stats = ensure_cube_has_valid_data(cube)
    diagnostics["validity_stats"] = stats
    LOGGER.info(
        "validated staged cutout lat=%s lon=%s stats=%s staging=%s",
        latitude,
        longitude,
        stats,
        {
            "item_strategy": diagnostics.get("item_strategy"),
            "candidate_count": diagnostics.get("candidate_count"),
            "selected_indices": diagnostics.get("selected_indices"),
            "final_center_nonzero_fraction": diagnostics.get("final_center_nonzero_fraction"),
            "final_full_nonzero_fraction": diagnostics.get("final_full_nonzero_fraction"),
        },
    )

    epsg_text = str(cube.attrs.get("epsg", "") or cube.coords.get("epsg", ""))
    epsg_code = parse_epsg(epsg_text, latitude, longitude)
    cube = cube.rio.write_crs(epsg_code, inplace=False)

    if config.output_dtype == "uint16":
        cube = cube.copy(data=scale_to_uint16(cube.data))
    cube = cube.rio.write_nodata(config.nodata, encoded=True, inplace=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cube.rio.to_raster(
        output_path,
        compress=config.compression,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    )
    if metadata_path is not None:
        write_json(metadata_path, diagnostics)
    LOGGER.info(
        "wrote staged cutout lat=%s lon=%s output=%s", latitude, longitude, output_path
    )
    return output_path.resolve()
