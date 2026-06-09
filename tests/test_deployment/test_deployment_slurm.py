from __future__ import annotations

from pathlib import Path

from deployment.srgan_hpc.config import EnvironmentConfig, SlurmConfig
from deployment.srgan_hpc.slurm import SlurmJobSpec, build_sbatch_command


def test_build_sbatch_command_includes_dependency() -> None:
    spec = SlurmJobSpec(
        job_name="collect",
        script_path=Path("/tmp/collect.sh"),
        manifest_path=Path("/tmp/run_manifest.yaml"),
        output_path=Path("/tmp/out.log"),
        error_path=Path("/tmp/err.log"),
        slurm=SlurmConfig(gpus=0),
        environment=EnvironmentConfig(python_executable="python"),
        dependency="afterok:12345",
    )

    command = build_sbatch_command(spec)

    assert "--dependency=afterok:12345" in command


def test_build_sbatch_command_can_skip_gpu_request() -> None:
    spec = SlurmJobSpec(
        job_name="collect",
        script_path=Path("/tmp/collect.sh"),
        manifest_path=Path("/tmp/run_manifest.yaml"),
        output_path=Path("/tmp/out.log"),
        error_path=Path("/tmp/err.log"),
        slurm=SlurmConfig(gpu_type="A100", gpus=1),
        environment=EnvironmentConfig(python_executable="python"),
        request_gpus=False,
    )

    command = build_sbatch_command(spec)

    assert not any(part.startswith("--gpus=") for part in command)
    assert not any(part.startswith("--gres=") for part in command)
