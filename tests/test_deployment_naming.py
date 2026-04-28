from __future__ import annotations

from pathlib import Path

from deployment.srgan_hpc.naming import (
    fused_output_name,
    patch_dir,
    patch_output_name,
    product_output_name,
    resolve_run_dir,
)


def test_output_name_helpers_format_expected_names() -> None:
    assert patch_output_name(45.1234567, 9.9876543) == "output_SR_image_45.123457_9.987654.tif"
    assert product_output_name("rgbnir") == "rgbnir_sr.tif"
    assert fused_output_name() == "fused_sr.tif"


def test_run_and_patch_directory_helpers_join_paths() -> None:
    output_root = Path("/tmp/srgan-runs")
    run_dir = resolve_run_dir(output_root, "run_001")
    assert run_dir == output_root / "run_001"

    patch_path = patch_dir(run_dir, "patch_0001")
    assert patch_path == output_root / "run_001" / "patches" / "patch_0001"
