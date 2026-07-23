from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from deployment.srgan_hpc import bundled_slurm_entrypoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="srgan-hpc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--config", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_subparsers = submit_parser.add_subparsers(
        dest="submit_command", required=True
    )

    patch_parser = submit_subparsers.add_parser("patch")
    _add_submit_common_args(patch_parser)
    patch_parser.add_argument("--lat", type=float, required=True)
    patch_parser.add_argument("--lon", type=float, required=True)

    grid_parser = submit_subparsers.add_parser("grid")
    _add_submit_common_args(grid_parser)
    grid_parser.add_argument("--lat1", type=float, required=True)
    grid_parser.add_argument("--lon1", type=float, required=True)
    grid_parser.add_argument("--lat2", type=float, required=True)
    grid_parser.add_argument("--lon2", type=float, required=True)

    aoi_parser = submit_subparsers.add_parser("aoi")
    _add_submit_common_args(aoi_parser)
    aoi_parser.add_argument("--aoi-path")

    run_parser = subparsers.add_parser("run")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)
    task_parser = run_subparsers.add_parser("task")
    task_parser.add_argument("--manifest", required=True)
    task_parser.add_argument("--task-index", type=int)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--run-dir", required=True)
    collect_parser.add_argument("--dest")

    deliver_parser = subparsers.add_parser("deliver-bbox")
    deliver_parser.add_argument("--run-root", required=True)
    deliver_parser.add_argument("--west", type=float, required=True)
    deliver_parser.add_argument("--south", type=float, required=True)
    deliver_parser.add_argument("--east", type=float, required=True)
    deliver_parser.add_argument("--north", type=float, required=True)
    deliver_parser.add_argument("--dest")
    deliver_parser.add_argument("--output-name", default="fused_sr.tif")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-dir", required=True)

    return parser


def _add_submit_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-root",
        help="Override config.output_root for this run without creating a new runtime YAML.",
    )
    parser.add_argument(
        "--project-name",
        help="Override config.project_name for this run without creating a new runtime YAML.",
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--script-path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def _submit_config_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if args.output_root:
        overrides["output_root"] = args.output_root
    if args.project_name:
        overrides["project_name"] = args.project_name
    return overrides


def _load_submit_config(args: argparse.Namespace):
    from deployment.srgan_hpc.config import load_runtime_config

    return load_runtime_config(args.config, overrides=_submit_config_overrides(args))


def _write_and_print_summary(
    *,
    run_dir: Path,
    config,
    submission,
    request: dict[str, object],
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    from deployment.srgan_hpc.submission_summary import (
        format_submission_summary,
        write_submission_summary,
    )

    summary, summary_json, summary_txt = write_submission_summary(
        run_dir=run_dir,
        config=config,
        submission=submission,
        request=request,
        start_date=start_date,
        end_date=end_date,
    )
    print(format_submission_summary(summary))
    return {"json": str(summary_json), "text": str(summary_txt)}


def _resolve_script_path(script_path: str | None) -> Path:
    if script_path is None:
        return bundled_slurm_entrypoint().resolve()
    return Path(script_path).expanduser().resolve()


def _log_multi_cutout_info(logger, patch_count: int, source_name: str) -> None:
    if patch_count <= 1:
        return
    logger.info(
        "%s uses multiple cubo cutouts (%d); cutouts overlap via staging.overlap_meters, "
        "but overlapping SR outputs are not reconciled after inference, so downstream mosaics may show seams "
        "at cutout boundaries",
        source_name,
        patch_count,
    )


def _handle_validate(args: argparse.Namespace) -> int:
    from deployment.srgan_hpc.config import load_runtime_config, patch_resolution

    config = load_runtime_config(args.config)
    print(f"Configuration valid: {config.config_path}")
    return 0


def _handle_submit_patch(args: argparse.Namespace) -> int:
    from deployment.srgan_hpc.logging_utils import configure_logging
    from deployment.srgan_hpc.patching import Patch
    from deployment.srgan_hpc.submit import submit_patch_run

    logger = configure_logging(verbose=args.verbose)
    config = _load_submit_config(args)
    patch = Patch(
        patch_id="patch_000001",
        latitude=args.lat,
        longitude=args.lon,
        edge_size=config.staging.edge_size,
        row_index=0,
        row_count=1,
        column_index=0,
        column_count=1,
    )
    run_id, run_dir, submission = submit_patch_run(
        config=config,
        patch=patch,
        start_date=args.start_date,
        end_date=args.end_date,
        script_path=_resolve_script_path(args.script_path),
        dry_run=args.dry_run,
    )
    logger.info("submitted patch run_id=%s run_dir=%s", run_id, run_dir)
    summary_paths = _write_and_print_summary(
        run_dir=run_dir,
        config=config,
        submission=submission,
        request={
            "type": "patch",
            "lat": args.lat,
            "lon": args.lon,
            "planned_patch_count": 1,
        },
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "submission": submission,
                "summary": summary_paths,
            },
            indent=2,
        )
    )
    return 0


def _handle_submit_grid(args: argparse.Namespace) -> int:
    from deployment.srgan_hpc.config import patch_resolution
    from deployment.srgan_hpc.logging_utils import configure_logging
    from deployment.srgan_hpc.patching import build_patches
    from deployment.srgan_hpc.submit import submit_grid_run

    logger = configure_logging(verbose=args.verbose)
    config = _load_submit_config(args)
    patches = build_patches(
        args.lat1,
        args.lon1,
        args.lat2,
        args.lon2,
        config.staging.edge_size,
        float(patch_resolution(config)),
        config.staging.overlap_meters,
    )
    _log_multi_cutout_info(logger, len(patches), "grid request")
    run_id, run_dir, submission = submit_grid_run(
        config=config,
        patches=patches,
        start_date=args.start_date,
        end_date=args.end_date,
        script_path=_resolve_script_path(args.script_path),
        dry_run=args.dry_run,
    )
    logger.info(
        "submitted grid run_id=%s run_dir=%s patches=%d", run_id, run_dir, len(patches)
    )
    summary_paths = _write_and_print_summary(
        run_dir=run_dir,
        config=config,
        submission=submission,
        request={
            "type": "grid",
            "lat1": args.lat1,
            "lon1": args.lon1,
            "lat2": args.lat2,
            "lon2": args.lon2,
            "planned_patch_count": len(patches),
        },
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "patches": len(patches),
                "submission": submission,
                "summary": summary_paths,
            },
            indent=2,
        )
    )
    return 0


def _handle_submit_aoi(args: argparse.Namespace) -> int:
    from deployment.srgan_hpc.aoi import select_aoi_patches
    from deployment.srgan_hpc.config import patch_resolution
    from deployment.srgan_hpc.logging_utils import configure_logging
    from deployment.srgan_hpc.submit import submit_aoi_run

    logger = configure_logging(verbose=args.verbose)
    config = _load_submit_config(args)
    aoi_path = args.aoi_path or config.aoi.path
    if aoi_path is None:
        raise ValueError("AOI path must be provided via --aoi-path or config.aoi.path")
    selection = select_aoi_patches(
        aoi_path=aoi_path,
        edge_size=config.staging.edge_size,
        resolution_m=float(patch_resolution(config)),
        overlap_meters=config.staging.overlap_meters,
    )
    _log_multi_cutout_info(logger, len(selection.patches), "AOI request")
    run_id, run_dir, submission = submit_aoi_run(
        config=config,
        patches=selection.patches,
        start_date=args.start_date,
        end_date=args.end_date,
        script_path=_resolve_script_path(args.script_path),
        aoi_path=selection.aoi_path,
        dry_run=args.dry_run,
    )
    logger.info(
        "submitted aoi run_id=%s run_dir=%s patches=%d aoi_path=%s",
        run_id,
        run_dir,
        len(selection.patches),
        selection.aoi_path,
    )
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "patches": len(selection.patches),
        "aoi_path": str(selection.aoi_path),
        "submission": submission,
    }
    payload["summary"] = _write_and_print_summary(
        run_dir=run_dir,
        config=config,
        submission=submission,
        request={
            "type": "aoi",
            "aoi_path": str(selection.aoi_path),
            "planned_patch_count": len(selection.patches),
        },
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _handle_run_task(args: argparse.Namespace) -> int:
    from deployment.srgan_hpc.logging_utils import configure_logging
    from deployment.srgan_hpc.run_task import run_task

    configure_logging()
    task_index = args.task_index
    if task_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    output = run_task(Path(args.manifest).resolve(), task_index=task_index)
    print(output if output is not None else "skipped")
    return 0


def _handle_collect(args: argparse.Namespace) -> int:
    from deployment.srgan_hpc.collect import collect_outputs

    destination, copied = collect_outputs(
        Path(args.run_dir).resolve(), Path(args.dest).resolve() if args.dest else None
    )
    print(json.dumps({"destination": str(destination), "copied": copied}, indent=2))
    return 0


def _handle_deliver_bbox(args: argparse.Namespace) -> int:
    from deployment.srgan_hpc.delivery import deliver_bbox_outputs

    destination, delivered = deliver_bbox_outputs(
        run_root=Path(args.run_root).resolve(),
        bbox=(args.west, args.south, args.east, args.north),
        destination=Path(args.dest).resolve() if args.dest else None,
        output_name=args.output_name,
    )
    print(json.dumps({"destination": str(destination), "outputs": delivered}, indent=2))
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    payload = {
        "run_dir": str(run_dir),
        "resolved_config": str(run_dir / "resolved_config.yaml"),
        "run_manifest": str(run_dir / "run_manifest.yaml"),
        "logs_dir": str(run_dir / "logs"),
        "patch_count": (
            len(list((run_dir / "patches").glob("patch_*")))
            if (run_dir / "patches").exists()
            else 0
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate-config":
        return _handle_validate(args)
    if args.command == "submit" and args.submit_command == "patch":
        return _handle_submit_patch(args)
    if args.command == "submit" and args.submit_command == "grid":
        return _handle_submit_grid(args)
    if args.command == "submit" and args.submit_command == "aoi":
        return _handle_submit_aoi(args)
    if args.command == "run" and args.run_command == "task":
        return _handle_run_task(args)
    if args.command == "collect":
        return _handle_collect(args)
    if args.command == "deliver-bbox":
        return _handle_deliver_bbox(args)
    if args.command == "status":
        return _handle_status(args)
    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
