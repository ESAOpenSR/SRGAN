from __future__ import annotations

import shutil
from pathlib import Path


def collect_outputs(run_dir: Path, destination: Path | None = None) -> tuple[Path, int]:
    destination = destination or run_dir / "collected"
    destination.mkdir(parents=True, exist_ok=True)
    moved = 0
    for tif_path in sorted(run_dir.glob("patches/*/outputs/*.tif")):
        patch_id = tif_path.parent.parent.name
        patch_destination = destination / patch_id
        patch_destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tif_path), patch_destination / tif_path.name)
        moved += 1
    return destination, moved
