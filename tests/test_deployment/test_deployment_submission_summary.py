from __future__ import annotations

from pathlib import Path

from deployment.srgan_hpc.config import RuntimeConfig
from deployment.srgan_hpc.manifests import read_yaml, write_json, write_yaml
from deployment.srgan_hpc.submission_summary import write_submission_summary


def test_submission_summary_writes_dry_run_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "srgan_001"
    write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "run_id": "srgan_001",
            "mode": "grid",
            "patch_count": 1,
            "skipped_count": 0,
            "tasks": [{"patch_id": "patch_000001", "manifest": "patches/patch_000001/manifest.yaml"}],
            "skipped": [],
        },
    )
    config = RuntimeConfig(output_root=tmp_path / "runs", project_name="srgan")

    summary, summary_json, summary_txt = write_submission_summary(
        run_dir=run_dir,
        config=config,
        submission={"mode": "dry-run", "command": "sbatch --array=0-0 script manifest"},
        request={"type": "grid", "planned_patch_count": 1},
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert summary["patches"] == {"planned": 1, "submitted": 1, "skipped": 0}
    assert summary["slurm"]["array"] == "0-0"
    assert summary_json.exists()
    assert summary_txt.exists()
    assert "SRGAN submission summary" in summary_txt.read_text(encoding="utf-8")
    assert "Staging: dry-run or no staging metadata available" in summary_txt.read_text(
        encoding="utf-8"
    )


def test_submission_summary_aggregates_staging_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "srgan_001"
    write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "run_id": "srgan_001",
            "mode": "grid",
            "patch_count": 1,
            "skipped_count": 0,
            "tasks": [{"patch_id": "patch_000001", "manifest": "patches/patch_000001/manifest.yaml"}],
            "skipped": [],
        },
    )
    metadata_dir = run_dir / "patches" / "patch_000001" / "metadata"
    write_json(
        metadata_dir / "rgbnir_staging.json",
        {
            "auto_selected_items": [
                {
                    "id": "S2_ITEM",
                    "tile": "33UUU",
                    "cloud_cover": 1.5,
                    "datetime": "2025-01-01T10:00:00Z",
                }
            ],
            "final_center_nonzero_fraction": 1.0,
            "final_full_nonzero_fraction": 0.997,
            "validity_stats": {
                "total_pixels": 100,
                "valid_pixels": 100,
                "nonzero_pixels": 99,
            },
        },
    )
    write_json(
        metadata_dir / "swir_staging.json",
        {
            "auto_selected_items": [
                {
                    "id": "S2_ITEM",
                    "tile": "33UUU",
                    "cloud_cover": 1.5,
                    "datetime": "2025-01-01T10:00:00Z",
                }
            ],
            "final_center_nonzero_fraction": 1.0,
            "final_full_nonzero_fraction": 1.0,
            "validity_stats": {
                "total_pixels": 25,
                "valid_pixels": 25,
                "nonzero_pixels": 25,
            },
        },
    )
    config = RuntimeConfig(output_root=tmp_path / "runs", project_name="srgan")

    summary, summary_json, summary_txt = write_submission_summary(
        run_dir=run_dir,
        config=config,
        submission={"job_id": "12345", "stdout": "Submitted batch job 12345", "stderr": ""},
        request={"type": "grid", "planned_patch_count": 1},
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert summary["staging"]["rgbnir"]["tiles"] == ["33UUU"]
    assert summary["staging"]["rgbnir"]["cloud_cover"]["mean"] == 1.5
    assert summary["staging"]["rgbnir"]["full_nonzero_fraction"]["min"] == 0.997
    assert summary["staging"]["rgbnir"]["validity"]["nonzero_pixels"] == 99
    assert read_yaml(run_dir / "run_manifest.yaml")["patch_count"] == 1
    assert summary_json.exists()
    text = summary_txt.read_text(encoding="utf-8")
    assert "rgbnir: patches=1" in text
    assert "cloud=1.500-1.500%" in text
