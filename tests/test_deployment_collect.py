from __future__ import annotations

from pathlib import Path

from deployment.srgan_hpc.collect import collect_outputs


def test_collect_outputs_preserves_patch_identity_for_duplicate_product_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for patch_id, marker in (("patch_000001", b"first"), ("patch_000002", b"second")):
        output_dir = run_dir / "patches" / patch_id / "outputs"
        output_dir.mkdir(parents=True)
        (output_dir / "fused_sr.tif").write_bytes(marker)

    destination, copied = collect_outputs(run_dir)

    assert destination == run_dir / "collected"
    assert copied == 2
    assert (destination / "patch_000001" / "fused_sr.tif").read_bytes() == b"first"
    assert (destination / "patch_000002" / "fused_sr.tif").read_bytes() == b"second"
