from __future__ import annotations

from pathlib import Path

import pytest

from deployment.srgan_hpc.manifests import write_yaml
from deployment.srgan_hpc.run_task import _resolve_patch_manifest_path


def test_resolve_patch_manifest_path_validates_task_index(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.yaml"
    write_yaml(
        manifest_path,
        {
            "tasks": [
                {
                    "patch_id": "patch_000001",
                    "manifest": "patches/patch_000001/manifest.yaml",
                }
            ]
        },
    )

    with pytest.raises(IndexError, match="out of range"):
        _resolve_patch_manifest_path(manifest_path, 1)


def test_resolve_patch_manifest_path_requires_task_index(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.yaml"
    write_yaml(manifest_path, {"tasks": []})

    with pytest.raises(ValueError, match="requires task index"):
        _resolve_patch_manifest_path(manifest_path, None)
