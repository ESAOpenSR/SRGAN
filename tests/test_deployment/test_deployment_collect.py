from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from deployment.srgan_hpc.collect import collect_outputs
from deployment.srgan_hpc.delivery import deliver_bbox_outputs
from deployment.srgan_hpc.manifests import write_yaml


def test_collect_outputs_preserves_patch_identity_for_duplicate_product_names(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    for patch_id, marker in (("patch_000001", b"first"), ("patch_000002", b"second")):
        output_dir = run_dir / "patches" / patch_id / "outputs"
        output_dir.mkdir(parents=True)
        (output_dir / "fused_sr.tif").write_bytes(marker)

    destination, moved = collect_outputs(run_dir)

    assert destination == run_dir / "collected"
    assert moved == 2
    assert (destination / "patch_000001" / "fused_sr.tif").read_bytes() == b"first"
    assert (destination / "patch_000002" / "fused_sr.tif").read_bytes() == b"second"
    assert not (run_dir / "patches" / "patch_000001" / "outputs" / "fused_sr.tif").exists()
    assert not (run_dir / "patches" / "patch_000002" / "outputs" / "fused_sr.tif").exists()


def test_deliver_bbox_reads_outputs_after_automatic_collection(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_path = run_dir / "patches" / "patch_000001" / "outputs" / "fused_sr.tif"
    output_path.parent.mkdir(parents=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint16",
        crs="EPSG:32632",
        transform=from_origin(500000, 100, 10, 10),
        nodata=0,
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype="uint16"))
    write_yaml(run_dir / "run_manifest.yaml", {"start_date": "2025-07-01"})

    collected_dir, moved = collect_outputs(run_dir)
    destination, delivered = deliver_bbox_outputs(
        run_root=run_dir,
        bbox=(9.0, 0.0, 9.001, 0.001),
    )

    collected_output = collected_dir / "patch_000001" / "fused_sr.tif"
    assert moved == 1
    assert collected_output.is_file()
    assert delivered[0]["inputs"] == [str(collected_output)]
    assert (destination / "run_fused_sr_clipped.tif").is_file()
