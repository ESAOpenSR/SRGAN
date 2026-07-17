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


def _point_geometry(longitude: float, latitude: float) -> dict[str, object]:
    return {"type": "Point", "coordinates": [longitude, latitude]}


def _log_label(patch_id: str | None, product_name: str | None) -> str:
    return f"{patch_id or 'patch'} {product_name or 'product'}"


def _format_cloud_cover_range(reports: list[dict[str, Any]]) -> str:
    values: list[float] = []
    for report in reports:
        value = report.get("cloud_cover")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}%"
    return f"{min(values):.3f}-{max(values):.3f}%"


def _search_kwargs_from_config(config: StagingConfig) -> dict[str, Any]:
    search_kwargs: dict[str, Any] = {}
    if config.search_query:
        search_kwargs["query"] = config.search_query
    if config.search_max_items is not None:
        search_kwargs["max_items"] = config.search_max_items
    if config.search_limit is not None:
        search_kwargs["limit"] = config.search_limit
    return search_kwargs


def _auto_select_item_ids(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    config: StagingConfig,
    patch_id: str | None = None,
    product_name: str | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import pystac_client
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'srgan-hpc auto item selection requires pystac-client. Install with `pip install "opensr-srgan[hpc]"`.'
        ) from exc

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    search = catalog.search(
        collections=[config.collection],
        datetime=f"{start_date}/{end_date}",
        intersects=_point_geometry(longitude, latitude),
        query=config.search_query or None,
        max_items=config.auto_select_item_limit,
        limit=config.auto_select_item_limit,
    )
    items = list(search.items())
    if not items:
        raise SkipTileError(
            "no_stac_items",
            details={"latitude": int(latitude * 1_000_000), "longitude": int(longitude * 1_000_000)},
        )

    selected_reports = [
        {
            "id": item.id,
            "tile": item.properties.get("s2:mgrs_tile"),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "datetime": item.properties.get("datetime"),
        }
        for item in items
    ]
    tiles = sorted(
        {
            str(report["tile"])
            for report in selected_reports
            if report.get("tile") is not None
        }
    )
    LOGGER.info(
        "[stac]  %s candidates=%s tiles=%s cloud_range=%s first_item=%s",
        _log_label(patch_id, product_name),
        len(selected_reports),
        ",".join(tiles) if tiles else "n/a",
        _format_cloud_cover_range(selected_reports),
        selected_reports[0]["id"],
    )
    return [item.id for item in items], selected_reports


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
    patch_id: str | None = None,
    product_name: str | None = None,
):
    try:
        import cubo
        import rioxarray  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'srgan-hpc staging requires optional dependencies. Install with `pip install "opensr-srgan[hpc]"`.'
        ) from exc

    search_kwargs = _search_kwargs_from_config(config)
    selected_item_reports: list[dict[str, Any]] = []
    if config.auto_select_item:
        selected_item_ids, selected_item_reports = _auto_select_item_ids(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            config=config,
            patch_id=patch_id,
            product_name=product_name,
        )
        search_kwargs["ids"] = selected_item_ids
        search_kwargs.setdefault("max_items", len(selected_item_ids))
        search_kwargs.setdefault("limit", len(selected_item_ids))

    for attempt, delay in enumerate(config.rate_limit_retry_delays_seconds, start=1):
        try:
            cube = cubo.create(
                lat=latitude,
                lon=longitude,
                collection=config.collection,
                bands=bands,
                start_date=start_date,
                end_date=end_date,
                edge_size=edge_size,
                resolution=resolution,
                **search_kwargs,
            )
            return cube, selected_item_reports
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

    cube = cubo.create(
        lat=latitude,
        lon=longitude,
        collection=config.collection,
        bands=bands,
        start_date=start_date,
        end_date=end_date,
        edge_size=edge_size,
        resolution=resolution,
        **search_kwargs,
    )
    return cube, selected_item_reports


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
    patch_id: str | None = None,
    product_name: str | None = None,
) -> Path:
    ensure_proj_env()
    LOGGER.info(
        "[stage] %s lat=%.6f lon=%.6f date=%s/%s edge=%s res=%sm",
        _log_label(patch_id, product_name),
        latitude,
        longitude,
        start_date,
        end_date,
        edge_size,
        resolution,
    )
    cube, auto_selected_items = create_cube_with_retry(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        config=config,
        bands=bands,
        edge_size=edge_size,
        resolution=resolution,
        patch_id=patch_id,
        product_name=product_name,
    )
    cube, diagnostics = _select_or_mosaic_time_items(cube, config)
    if auto_selected_items:
        diagnostics["auto_selected_items"] = auto_selected_items
    cube = cube.transpose("band", "y", "x")
    stats = ensure_cube_has_valid_data(cube)
    diagnostics["validity_stats"] = stats
    LOGGER.info(
        "[valid] %s full=%.3f center=%.3f nonzero=%s/%s",
        _log_label(patch_id, product_name),
        float(diagnostics.get("final_full_nonzero_fraction") or 0.0),
        float(diagnostics.get("final_center_nonzero_fraction") or 0.0),
        stats["nonzero_pixels"],
        stats["total_pixels"],
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
    try:
        output_label = str(output_path.relative_to(output_path.parent.parent))
    except ValueError:
        output_label = str(output_path)
    LOGGER.info(
        "[write] %s %s", _log_label(patch_id, product_name), output_label
    )
    return output_path.resolve()
