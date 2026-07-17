from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import deployment.srgan_hpc.cli as cli_module
from deployment.srgan_hpc.aoi import AoiSelection
from deployment.srgan_hpc.config import (
    EnvironmentConfig,
    RuntimeConfig,
    SlurmConfig,
    get_product_config,
    load_runtime_config,
)
from deployment.srgan_hpc.manifests import read_yaml
from deployment.srgan_hpc.patching import Patch
from deployment.srgan_hpc.slurm import (
    SlurmJobSpec,
    build_sbatch_command,
    parse_job_id,
    submit_job,
)


def test_cli_grid_aoi_and_collect_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        f"output_root: {tmp_path / 'runs'}\nproject_name: cli\nmode: rgbnir\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srgan-hpc",
            "submit",
            "grid",
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "override-runs"),
            "--project-name",
            "cli-grid",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--lat1",
            "45.0",
            "--lon1",
            "9.0",
            "--lat2",
            "45.001",
            "--lon2",
            "9.001",
            "--script-path",
            "/tmp/slurm.sh",
            "--dry-run",
        ],
    )
    assert cli_module.main() == 0
    grid_output = capsys.readouterr().out
    assert '"patches": 1' in grid_output
    assert "override-runs" in grid_output

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
    monkeypatch.setattr(
        "deployment.srgan_hpc.aoi.select_aoi_patches",
        lambda **_kwargs: AoiSelection(
            aoi_path=tmp_path / "area.shp",
            aoi_layer="named-layer",
            geometry=types.SimpleNamespace(),
            patches=[patch],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srgan-hpc",
            "submit",
            "aoi",
            "--config",
            str(config_path),
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--aoi-path",
            str(tmp_path / "area.shp"),
            "--layer",
            "named-layer",
            "--script-path",
            "/tmp/slurm.sh",
            "--dry-run",
        ],
    )
    assert cli_module.main() == 0
    aoi_output = capsys.readouterr().out
    assert '"aoi_layer": "named-layer"' in aoi_output

    run_dir = tmp_path / "collect-run"
    source = run_dir / "patches" / "patch_000001" / "outputs" / "fused_sr.tif"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"output")
    dest = tmp_path / "collected"
    monkeypatch.setattr(
        sys,
        "argv",
        ["srgan-hpc", "collect", "--run-dir", str(run_dir), "--dest", str(dest)],
    )
    assert cli_module.main() == 0
    collect_payload = json.loads(capsys.readouterr().out)
    assert collect_payload["copied"] == 1
    assert Path(collect_payload["destination"]) == dest.resolve()


def test_slurm_command_includes_optional_resources() -> None:
    base = {
        "job_name": "srgan",
        "script_path": Path("/tmp/slurm.sh"),
        "manifest_path": Path("/tmp/manifest.yaml"),
        "output_path": Path("/tmp/out.log"),
        "error_path": Path("/tmp/err.log"),
        "environment": EnvironmentConfig(
            python_executable="/work/envs/srgan/bin/python",
            modules=["cuda", "gdal"],
            conda_env="srgan",
        ),
    }

    command = build_sbatch_command(
        SlurmJobSpec(
            **base,
            slurm=SlurmConfig(
                partition="gpu",
                gres="gpu:a100:1",
                account="proj",
                qos="normal",
                extra_args=["--exclusive"],
            ),
            array="0-3",
        )
    )
    assert "--partition=gpu" in command
    assert "--gres=gpu:a100:1" in command
    assert "--account=proj" in command
    assert "--qos=normal" in command
    assert "--array=0-3" in command
    assert "--exclusive" in command
    assert any("SRGAN_HPC_MODULES=cuda,gdal" in part for part in command)
    assert any("SRGAN_HPC_CONDA_ENV=srgan" in part for part in command)

    gpu_command = build_sbatch_command(
        SlurmJobSpec(
            **base,
            slurm=SlurmConfig(gpu_type="A100", gpus=2),
        )
    )
    assert "--gpus=A100:2" in gpu_command

    plain_gpu_command = build_sbatch_command(
        SlurmJobSpec(
            **base,
            slurm=SlurmConfig(gpus=1),
        )
    )
    assert "--gpus=1" in plain_gpu_command


def test_submit_job_success_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = SlurmJobSpec(
        job_name="srgan",
        script_path=Path("/tmp/slurm.sh"),
        manifest_path=Path("/tmp/manifest.yaml"),
        output_path=Path("/tmp/out.log"),
        error_path=Path("/tmp/err.log"),
        slurm=SlurmConfig(gpus=0),
        environment=EnvironmentConfig(python_executable="python"),
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, check, capture_output, text):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["capture_output"] = capture_output
        captured["text"] = text
        return types.SimpleNamespace(
            stdout="Submitted batch job 12345\n",
            stderr="",
        )

    monkeypatch.setattr("deployment.srgan_hpc.slurm.subprocess.run", fake_run)

    payload = submit_job(spec, tmp_path / "success")

    assert payload["job_id"] == "12345"
    assert captured["cmd"][0] == "sbatch"
    assert (
        json.loads(
            (tmp_path / "success" / "slurm_job_ids.json").read_text(encoding="utf-8")
        )["job_id"]
        == "12345"
    )

    def fake_error(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(
            returncode=7,
            cmd=cmd,
            output="stdout text",
            stderr="stderr text",
        )

    monkeypatch.setattr("deployment.srgan_hpc.slurm.subprocess.run", fake_error)

    with pytest.raises(RuntimeError, match="sbatch submission failed"):
        submit_job(spec, tmp_path / "error")

    error_payload = json.loads(
        (tmp_path / "error" / "slurm_job_ids.json").read_text(encoding="utf-8")
    )
    assert error_payload["mode"] == "error"
    assert error_payload["returncode"] == "7"
    assert error_payload["stderr"] == "stderr text"


def test_parse_job_id_rejects_empty_output() -> None:
    with pytest.raises(ValueError, match="Could not parse"):
        parse_job_id("")


def test_runtime_config_extra_loading_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown product"):
        get_product_config(RuntimeConfig(), "unknown")

    non_mapping = tmp_path / "bad.yaml"
    non_mapping.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected mapping"):
        load_runtime_config(non_mapping)

    model_config = tmp_path / "model.yaml"
    model_checkpoint = tmp_path / "model.ckpt"
    aoi_path = tmp_path / "area.shp"
    model_config.write_text("Model: {}\n", encoding="utf-8")
    model_checkpoint.write_bytes(b"checkpoint")
    aoi_path.touch()
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        """
mode: rgbnir
output_root: runs
aoi:
  path: area.shp
rgbnir:
  config_path: model.yaml
  checkpoint_path: model.ckpt
  model:
    preset: null
    cache_dir: cache
""",
        encoding="utf-8",
    )

    config = load_runtime_config(
        config_path,
        overrides={"staging": {"edge_size": 128}},
    )

    assert config.output_root == (tmp_path / "runs").resolve()
    assert config.aoi.path == str(aoi_path.resolve())
    assert config.rgbnir.model.config_path == str(model_config.resolve())
    assert config.rgbnir.model.checkpoint_path == str(model_checkpoint.resolve())
    assert config.rgbnir.model.cache_dir == str((tmp_path / "cache").resolve())
    assert config.staging.edge_size == 128
