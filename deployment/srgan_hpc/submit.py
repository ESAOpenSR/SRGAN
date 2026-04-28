from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from deployment.srgan_hpc.config import (
    RuntimeConfig,
    enabled_product_names,
    get_product_config,
    product_edge_size,
    runtime_config_to_dict,
)
from deployment.srgan_hpc.manifests import new_run_id, write_json, write_yaml
from deployment.srgan_hpc.naming import patch_dir, resolve_run_dir
from deployment.srgan_hpc.patching import Patch
from deployment.srgan_hpc.slurm import SlurmJobSpec, submit_job
from deployment.srgan_hpc.staging import SkipTileError, stage_cutout


def _patch_manifest(
    *,
    patch: Patch,
    run_id: str,
    start_date: str,
    end_date: str,
    config: RuntimeConfig,
    ) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "patch_id": patch.patch_id,
        "patch_index": patch.row_index * patch.column_count + patch.column_index,
        "latitude": patch.latitude,
        "longitude": patch.longitude,
        "edge_size": patch.edge_size,
        "start_date": start_date,
        "end_date": end_date,
        "products": enabled_product_names(config),
        "paths": {
            "run_dir": "../..",
            "inputs": {product: f"inputs/{product}.tif" for product in enabled_product_names(config)},
            "output_dir": "outputs",
            "metadata_dir": "metadata",
        },
        "config": runtime_config_to_dict(config),
    }


def _write_skip_metadata(
    *,
    patch_root: Path,
    patch: Patch,
    reason: str,
    details: dict[str, int],
    start_date: str,
    end_date: str,
) -> None:
    metadata_dir = patch_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        metadata_dir / "skip.json",
        {
            "status": "skipped",
            "reason": reason,
            "details": details,
            "patch_id": patch.patch_id,
            "latitude": patch.latitude,
            "longitude": patch.longitude,
            "start_date": start_date,
            "end_date": end_date,
        },
    )


def _stage_patch_inputs(
    *,
    patch: Patch,
    start_date: str,
    end_date: str,
    config: RuntimeConfig,
    patch_root: Path,
    dry_run: bool,
) -> None:
    for product_name in enabled_product_names(config):
        product = get_product_config(config, product_name)
        input_tif = patch_root / "inputs" / f"{product_name}.tif"
        if dry_run:
            input_tif.parent.mkdir(parents=True, exist_ok=True)
            continue
        stage_cutout(
            latitude=patch.latitude,
            longitude=patch.longitude,
            start_date=start_date,
            end_date=end_date,
            config=config.staging,
            bands=product.bands,
            edge_size=product_edge_size(config, product),
            resolution=product.resolution,
            output_path=input_tif,
        )


def submit_patch_run(
    *,
    config: RuntimeConfig,
    patch: Patch,
    start_date: str,
    end_date: str,
    script_path: Path,
    dry_run: bool = False,
) -> tuple[str, Path, Mapping[str, object]]:
    run_id = new_run_id(config.project_name)
    run_dir = resolve_run_dir(config.output_root, run_id)
    logs_dir = run_dir / "logs"
    patch_root = patch_dir(run_dir, patch.patch_id)

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "resolved_config.yaml", runtime_config_to_dict(config))

    if not dry_run:
        try:
            _stage_patch_inputs(
                patch=patch,
                start_date=start_date,
                end_date=end_date,
                config=config,
                patch_root=patch_root,
                dry_run=False,
            )
        except SkipTileError as exc:
            manifest = _patch_manifest(
                patch=patch,
                run_id=run_id,
                start_date=start_date,
                end_date=end_date,
                config=config,
            )
            manifest["status"] = "skipped"
            manifest["skip_reason"] = exc.reason
            manifest["skip_details"] = exc.details
            manifest_path = patch_root / "manifest.yaml"
            write_yaml(manifest_path, manifest)
            _write_skip_metadata(
                patch_root=patch_root,
                patch=patch,
                reason=exc.reason,
                details=exc.details,
                start_date=start_date,
                end_date=end_date,
            )
            write_yaml(
                run_dir / "run_manifest.yaml",
                {
                    "run_id": run_id,
                    "mode": "patch",
                    "patch_count": 1,
                    "tasks": [],
                    "skipped": [{"patch_id": patch.patch_id, "reason": exc.reason}],
                },
            )
            return run_id, run_dir, {"mode": "skipped", "reason": exc.reason, **exc.details}
    else:
        _stage_patch_inputs(
            patch=patch,
            start_date=start_date,
            end_date=end_date,
            config=config,
            patch_root=patch_root,
            dry_run=True,
        )

    manifest = _patch_manifest(
        patch=patch,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "run_id": run_id,
            "mode": "patch",
            "patch_count": 1,
            "tasks": [{"patch_id": patch.patch_id, "manifest": f"patches/{patch.patch_id}/manifest.yaml"}],
        },
    )
    manifest_path = patch_root / "manifest.yaml"
    write_yaml(manifest_path, manifest)

    spec = SlurmJobSpec(
        job_name=f"srgan_{patch.patch_id}",
        script_path=script_path,
        manifest_path=manifest_path,
        output_path=logs_dir / f"slurm_{patch.patch_id}.out",
        error_path=logs_dir / f"slurm_{patch.patch_id}.err",
        slurm=config.slurm,
        environment=config.environment,
    )
    submission = submit_job(spec, run_dir / "submission", dry_run=dry_run)
    return run_id, run_dir, submission


def _submit_patch_collection(
    *,
    mode: str,
    config: RuntimeConfig,
    patches: list[Patch],
    start_date: str,
    end_date: str,
    script_path: Path,
    dry_run: bool = False,
    aoi_path: Path | None = None,
    aoi_layer: str | None = None,
) -> tuple[str, Path, Mapping[str, object]]:
    run_id = new_run_id(config.project_name)
    run_dir = resolve_run_dir(config.output_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "resolved_config.yaml", runtime_config_to_dict(config))

    tasks: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for patch_index, patch in enumerate(patches):
        patch_root = patch_dir(run_dir, patch.patch_id)
        if not dry_run:
            try:
                _stage_patch_inputs(
                    patch=patch,
                    start_date=start_date,
                    end_date=end_date,
                    config=config,
                    patch_root=patch_root,
                    dry_run=False,
                )
            except SkipTileError as exc:
                manifest = _patch_manifest(
                    patch=patch,
                    run_id=run_id,
                    start_date=start_date,
                    end_date=end_date,
                    config=config,
                )
                manifest["patch_index"] = patch_index
                manifest["status"] = "skipped"
                manifest["skip_reason"] = exc.reason
                manifest["skip_details"] = exc.details
                write_yaml(patch_root / "manifest.yaml", manifest)
                _write_skip_metadata(
                    patch_root=patch_root,
                    patch=patch,
                    reason=exc.reason,
                    details=exc.details,
                    start_date=start_date,
                    end_date=end_date,
                )
                skipped.append({"patch_id": patch.patch_id, "reason": exc.reason, **exc.details})
                continue
        else:
            _stage_patch_inputs(
                patch=patch,
                start_date=start_date,
                end_date=end_date,
                config=config,
                patch_root=patch_root,
                dry_run=True,
            )

        manifest = _patch_manifest(
            patch=patch,
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )
        manifest["patch_index"] = patch_index
        write_yaml(patch_root / "manifest.yaml", manifest)
        tasks.append(manifest)

    run_manifest = {
        "run_id": run_id,
        "mode": mode,
        "patch_count": len(tasks),
        "start_date": start_date,
        "end_date": end_date,
        "skipped_count": len(skipped),
        "tasks": [
            {"patch_id": str(task["patch_id"]), "manifest": f"patches/{task['patch_id']}/manifest.yaml"}
            for task in tasks
        ],
        "skipped": skipped,
    }
    if aoi_path is not None:
        run_manifest["aoi_path"] = str(aoi_path)
    if aoi_layer is not None:
        run_manifest["aoi_layer"] = aoi_layer
    write_yaml(run_dir / "run_manifest.yaml", run_manifest)

    if not tasks:
        return run_id, run_dir, {"mode": "skipped", "reason": "no_submittable_patches", "skipped": len(skipped)}

    spec = SlurmJobSpec(
        job_name=f"srgan_{run_id}",
        script_path=script_path,
        manifest_path=run_dir / "run_manifest.yaml",
        output_path=logs_dir / "slurm_%A_%a.out",
        error_path=logs_dir / "slurm_%A_%a.err",
        slurm=config.slurm,
        environment=config.environment,
        array=f"0-{len(tasks) - 1}" if tasks else None,
    )
    submission = submit_job(spec, run_dir / "submission", dry_run=dry_run)
    return run_id, run_dir, submission


def submit_grid_run(
    *,
    config: RuntimeConfig,
    patches: list[Patch],
    start_date: str,
    end_date: str,
    script_path: Path,
    dry_run: bool = False,
) -> tuple[str, Path, Mapping[str, object]]:
    return _submit_patch_collection(
        mode="grid",
        config=config,
        patches=patches,
        start_date=start_date,
        end_date=end_date,
        script_path=script_path,
        dry_run=dry_run,
    )


def submit_aoi_run(
    *,
    config: RuntimeConfig,
    patches: list[Patch],
    start_date: str,
    end_date: str,
    script_path: Path,
    aoi_path: Path,
    aoi_layer: str | None = None,
    dry_run: bool = False,
) -> tuple[str, Path, Mapping[str, object]]:
    return _submit_patch_collection(
        mode="aoi",
        config=config,
        patches=patches,
        start_date=start_date,
        end_date=end_date,
        script_path=script_path,
        dry_run=dry_run,
        aoi_path=aoi_path,
        aoi_layer=aoi_layer,
    )
