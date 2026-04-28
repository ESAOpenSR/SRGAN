from __future__ import annotations

from pathlib import Path

import pytest

from deployment.srgan_hpc.aoi import resolve_aoi_source_path


def test_resolve_aoi_source_accepts_direct_shapefile(tmp_path: Path) -> None:
    shp_path = tmp_path / "area.shp"
    shp_path.touch()

    assert resolve_aoi_source_path(shp_path) == shp_path.resolve()


def test_resolve_aoi_source_accepts_directory_with_one_shapefile(tmp_path: Path) -> None:
    shp_path = tmp_path / "area.shp"
    shp_path.touch()

    assert resolve_aoi_source_path(tmp_path) == shp_path.resolve()


def test_resolve_aoi_source_rejects_directory_with_multiple_shapefiles(tmp_path: Path) -> None:
    (tmp_path / "a.shp").touch()
    (tmp_path / "b.shp").touch()

    with pytest.raises(ValueError, match="exactly one"):
        resolve_aoi_source_path(tmp_path)
