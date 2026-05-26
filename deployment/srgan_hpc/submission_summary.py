from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from deployment.srgan_hpc.config import RuntimeConfig, enabled_product_names
from deployment.srgan_hpc.manifests import read_yaml, write_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "mean": mean(values), "max": max(values)}


def _command_array_range(command: str | None) -> str | None:
    if not command:
        return None
    for token in command.split():
        if token.startswith("--array="):
            return token.split("=", 1)[1]
    return None


def _collect_staging(run_dir: Path) -> dict[str, Any]:
    product_records: dict[str, list[dict[str, Any]]] = {}
    for metadata_path in sorted(run_dir.glob("patches/*/metadata/*_staging.json")):
        product_name = metadata_path.stem.removesuffix("_staging")
        payload = _read_json(metadata_path)
        payload["_patch_id"] = metadata_path.parent.parent.name
        product_records.setdefault(product_name, []).append(payload)

    summary: dict[str, Any] = {}
    for product_name, records in sorted(product_records.items()):
        center = [
            value
            for record in records
            if (value := _as_float(record.get("final_center_nonzero_fraction")))
            is not None
        ]
        full = [
            value
            for record in records
            if (value := _as_float(record.get("final_full_nonzero_fraction")))
            is not None
        ]
        clouds: list[float] = []
        tiles: set[str] = set()
        item_ids: set[str] = set()
        valid_pixels = 0
        nonzero_pixels = 0
        total_pixels = 0
        for record in records:
            for item in record.get("auto_selected_items", []) or []:
                if item.get("id"):
                    item_ids.add(str(item["id"]))
                if item.get("tile"):
                    tiles.add(str(item["tile"]))
                cloud = _as_float(item.get("cloud_cover"))
                if cloud is not None:
                    clouds.append(cloud)
            validity = record.get("validity_stats", {}) or {}
            valid_pixels += int(validity.get("valid_pixels", 0) or 0)
            nonzero_pixels += int(validity.get("nonzero_pixels", 0) or 0)
            total_pixels += int(validity.get("total_pixels", 0) or 0)

        summary[product_name] = {
            "patches": len(records),
            "tiles": sorted(tiles),
            "item_ids": sorted(item_ids),
            "cloud_cover": _stats(clouds),
            "center_nonzero_fraction": _stats(center),
            "full_nonzero_fraction": _stats(full),
            "validity": {
                "valid_pixels": valid_pixels,
                "nonzero_pixels": nonzero_pixels,
                "total_pixels": total_pixels,
            },
        }
    return summary


def build_submission_summary(
    *,
    run_dir: Path,
    config: RuntimeConfig,
    submission: Mapping[str, object],
    request: Mapping[str, object],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    run_manifest = read_yaml(run_dir / "run_manifest.yaml")
    skipped = list(run_manifest.get("skipped", []) or [])
    patch_count = int(run_manifest.get("patch_count", 0) or 0)
    planned_patch_count = int(request.get("planned_patch_count", patch_count) or 0)
    staging = _collect_staging(run_dir)
    warnings: list[str] = []
    if skipped:
        warnings.append(f"{len(skipped)} patch(es) were skipped during staging")
    for product_name, product_summary in staging.items():
        full = product_summary.get("full_nonzero_fraction")
        min_full = full.get("min") if isinstance(full, dict) else None
        if min_full is not None and min_full < config.staging.min_full_nonzero_fraction:
            warnings.append(
                f"{product_name} minimum full coverage {min_full:.3f} is below "
                f"{config.staging.min_full_nonzero_fraction:.3f}"
            )

    command = str(submission.get("command", "")) if submission.get("command") else None
    summary = {
        "run": {
            "run_id": run_manifest.get("run_id"),
            "run_dir": str(run_dir),
            "mode": run_manifest.get("mode"),
            "product_mode": config.mode,
            "start_date": start_date,
            "end_date": end_date,
            "config_path": str(config.config_path) if config.config_path else None,
            "project_name": config.project_name,
            "output_root": str(config.output_root),
        },
        "request": dict(request),
        "patches": {
            "planned": planned_patch_count,
            "submitted": patch_count,
            "skipped": len(skipped),
        },
        "products": {
            "enabled": enabled_product_names(config),
            "bands": {
                "rgbnir": list(config.rgbnir.bands),
                "swir": list(config.swir.bands),
            },
        },
        "staging": staging,
        "slurm": {
            **dict(submission),
            "array": _command_array_range(command),
        },
        "paths": {
            "logs": str(run_dir / "logs"),
            "submission": str(run_dir / "submission"),
            "run_manifest": str(run_dir / "run_manifest.yaml"),
            "resolved_config": str(run_dir / "resolved_config.yaml"),
        },
        "warnings": warnings,
    }
    return summary


def format_submission_summary(summary: Mapping[str, Any]) -> str:
    run = summary["run"]
    patches = summary["patches"]
    slurm = summary["slurm"]
    paths = summary["paths"]
    lines = [
        "",
        "SRGAN submission summary",
        f"Run: {run['run_id']}",
        f"Mode: {run['mode']} | {run['product_mode']}",
        f"Dates: {run['start_date']} to {run['end_date']}",
        f"Run dir: {run['run_dir']}",
        (
            "Patches: "
            f"{patches['submitted']} submitted, {patches['skipped']} skipped "
            f"({patches['planned']} planned)"
        ),
    ]
    if slurm.get("mode") == "dry-run":
        lines.append(f"Slurm: dry-run, array {slurm.get('array') or 'none'}")
    else:
        lines.append(
            f"Slurm: job {slurm.get('job_id', 'unknown')}, "
            f"array {slurm.get('array') or 'none'}"
        )

    staging = summary.get("staging", {})
    if staging:
        lines.append("Staging:")
        for product_name, product_summary in staging.items():
            cloud = product_summary.get("cloud_cover") or {}
            full = product_summary.get("full_nonzero_fraction") or {}
            center = product_summary.get("center_nonzero_fraction") or {}
            validity = product_summary.get("validity") or {}
            cloud_text = "n/a"
            if cloud:
                cloud_text = (
                    f"{cloud['min']:.3f}-{cloud['max']:.3f}% "
                    f"(mean {cloud['mean']:.3f}%)"
                )
            tiles = ",".join(product_summary.get("tiles") or []) or "n/a"
            lines.append(
                f"  {product_name}: patches={product_summary['patches']} "
                f"full_min={full.get('min', 0.0):.3f} "
                f"center_min={center.get('min', 0.0):.3f} "
                f"nonzero={validity.get('nonzero_pixels', 0)}/"
                f"{validity.get('total_pixels', 0)} "
                f"cloud={cloud_text} tiles={tiles}"
            )
    else:
        lines.append("Staging: dry-run or no staging metadata available")

    warnings = list(summary.get("warnings") or [])
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)

    lines.extend(
        [
            "Files:",
            f"  Summary JSON: {paths.get('summary_json')}",
            f"  Summary TXT:  {paths.get('summary_txt')}",
            f"  Logs:         {paths.get('logs')}",
        ]
    )
    return "\n".join(lines)


def write_submission_summary(
    *,
    run_dir: Path,
    config: RuntimeConfig,
    submission: Mapping[str, object],
    request: Mapping[str, object],
    start_date: str,
    end_date: str,
) -> tuple[dict[str, Any], Path, Path]:
    summary = build_submission_summary(
        run_dir=run_dir,
        config=config,
        submission=submission,
        request=request,
        start_date=start_date,
        end_date=end_date,
    )
    summary_json = run_dir / "submission" / "summary.json"
    summary_txt = run_dir / "submission" / "summary.txt"
    summary["paths"]["summary_json"] = str(summary_json)
    summary["paths"]["summary_txt"] = str(summary_txt)
    write_json(summary_json, summary)
    summary_txt.write_text(format_submission_summary(summary) + "\n", encoding="utf-8")
    return summary, summary_json, summary_txt
