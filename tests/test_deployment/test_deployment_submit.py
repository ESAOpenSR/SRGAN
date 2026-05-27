from __future__ import annotations

from pathlib import Path

from deployment.srgan_hpc.config import RuntimeConfig
from deployment.srgan_hpc.manifests import read_yaml
from deployment.srgan_hpc.patching import Patch
from deployment.srgan_hpc.submit import submit_grid_run, submit_patch_run


def test_submit_patch_dry_run_writes_product_inputs_and_manifest(
    tmp_path: Path,
) -> None:
    config = RuntimeConfig(output_root=tmp_path / "runs", mode="fused")
    patch = Patch(
        patch_id="patch_000001",
        latitude=45.0,
        longitude=9.0,
        edge_size=512,
        row_index=0,
        row_count=1,
        column_index=0,
        column_count=1,
    )

    _, run_dir, submission = submit_patch_run(
        config=config,
        patch=patch,
        start_date="2024-01-01",
        end_date="2024-01-31",
        script_path=Path("/tmp/slurm.sh"),
        dry_run=True,
    )

    manifest_path = run_dir / "patches" / "patch_000001" / "manifest.yaml"
    manifest = read_yaml(manifest_path)
    assert manifest["products"] == ["rgbnir", "swir"]
    assert manifest["paths"]["inputs"] == {
        "rgbnir": "inputs/rgbnir.tif",
        "swir": "inputs/swir.tif",
    }
    assert (run_dir / "patches" / "patch_000001" / "inputs").is_dir()
    assert submission["mode"] == "dry-run"


def test_submit_grid_dry_run_prepares_collect_followup_job(tmp_path: Path) -> None:
    config = RuntimeConfig(output_root=tmp_path / "runs", mode="fused")
    patches = [
        Patch(
            patch_id="patch_000001",
            latitude=45.0,
            longitude=9.0,
            edge_size=512,
            row_index=0,
            row_count=1,
            column_index=0,
            column_count=1,
        ),
        Patch(
            patch_id="patch_000002",
            latitude=45.1,
            longitude=9.1,
            edge_size=512,
            row_index=0,
            row_count=1,
            column_index=1,
            column_count=2,
        ),
    ]

    _, run_dir, submission = submit_grid_run(
        config=config,
        patches=patches,
        start_date="2024-01-01",
        end_date="2024-01-31",
        script_path=Path("/tmp/slurm_task.sh"),
        dry_run=True,
    )

    assert submission["mode"] == "dry-run"
    assert "--array=0-1" in str(submission["command"])
    collect = submission["collect"]
    assert collect["mode"] == "dry-run"
    assert "slurm_collect_entrypoint.sh" in str(collect["command"])
    assert "--array=" not in str(collect["command"])
    assert (run_dir / "submission" / "collect" / "sbatch_command.txt").is_file()
