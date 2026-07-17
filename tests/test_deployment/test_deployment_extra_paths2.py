from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import deployment.srgan_hpc.aoi as aoi_module
import deployment.srgan_hpc.cli as cli_module
import deployment.srgan_hpc.delivery as delivery_module
import deployment.srgan_hpc.run_task as run_task_module
import deployment.srgan_hpc.staging as staging_module
import deployment.srgan_hpc.submit as submit_module
from deployment.srgan_hpc.aoi import (
    load_aoi_geometry,
    patch_footprint,
    resolve_aoi_source_path,
    select_aoi_patches,
)
from deployment.srgan_hpc.config import (
    RuntimeConfig,
    StagingConfig,
    runtime_config_to_dict,
    validate_runtime_config,
)
from deployment.srgan_hpc.delivery import deliver_bbox_outputs
from deployment.srgan_hpc.manifests import read_yaml, write_yaml
from deployment.srgan_hpc.patching import Patch
from deployment.srgan_hpc.raster import (
    _as_scalar,
    compute_centroid_lat_lon,
    ensure_proj_env,
    guess_utm_epsg,
    parse_epsg,
    raster_validity_stats,
    scale_to_uint16,
)
from deployment.srgan_hpc.run_task import run_task
from deployment.srgan_hpc.staging import (
    SkipTileError,
    _select_or_mosaic_time_items,
    ensure_cube_has_valid_data,
    stage_cutout,
)
from deployment.srgan_hpc.submit import submit_grid_run, submit_patch_run


def _write_tif(
    path: Path,
    data: np.ndarray,
    transform,
    *,
    crs: str | None = "EPSG:32632",
    nodata: int | None = 0,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def _write_polygon_shapefile(path: Path) -> None:
    import shapefile
    from pyproj import CRS

    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    try:
        writer.field("name", "C")
        writer.poly([[(0.0, 0.0), (0.02, 0.0), (0.02, 0.02), (0.0, 0.02), (0.0, 0.0)]])
        writer.record("area")
    finally:
        writer.close()
    path.with_suffix(".prj").write_text(CRS.from_epsg(4326).to_wkt(), encoding="utf-8")


def _write_point_shapefile(path: Path) -> None:
    import shapefile
    from pyproj import CRS

    writer = shapefile.Writer(str(path), shapeType=shapefile.POINT)
    try:
        writer.field("name", "C")
        writer.point(0.0, 0.0)
        writer.record("point")
    finally:
        writer.close()
    path.with_suffix(".prj").write_text(CRS.from_epsg(4326).to_wkt(), encoding="utf-8")


def test_aoi_source_and_geometry_extra_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FileNotFoundError, match="AOI path not found"):
        resolve_aoi_source_path(tmp_path / "missing.shp")

    with pytest.raises(ValueError, match="No .shp file"):
        resolve_aoi_source_path(tmp_path)

    text_path = tmp_path / "area.geojson"
    text_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a .shp file"):
        resolve_aoi_source_path(text_path)

    shp_path = tmp_path / "area.shp"
    _write_polygon_shapefile(shp_path)
    resolved_path, geometry = load_aoi_geometry(shp_path)
    assert resolved_path == shp_path.resolve()
    assert geometry.bounds == pytest.approx((0.0, 0.0, 0.02, 0.02))

    missing_prj = tmp_path / "missing_prj.shp"
    missing_prj.touch()
    with pytest.raises(ValueError, match="missing .prj"):
        load_aoi_geometry(missing_prj)

    empty_prj = tmp_path / "empty_prj.shp"
    empty_prj.touch()
    empty_prj.with_suffix(".prj").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty .prj"):
        load_aoi_geometry(empty_prj)

    point_path = tmp_path / "point.shp"
    _write_point_shapefile(point_path)
    with pytest.raises(ValueError, match="polygon geometries"):
        load_aoi_geometry(point_path)

    selection = select_aoi_patches(
        aoi_path=shp_path,
        aoi_layer="layer-name",
        edge_size=64,
        resolution_m=10.0,
        overlap_meters=0.0,
    )
    assert selection.aoi_layer == "layer-name"
    assert selection.patches

    from shapely.geometry import box

    monkeypatch.setattr(
        aoi_module, "load_aoi_geometry", lambda _path: (shp_path, box(0, 0, 1, 1))
    )
    monkeypatch.setattr(aoi_module, "build_patches", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="No SR cutouts intersect"):
        select_aoi_patches(
            aoi_path=shp_path,
            aoi_layer=None,
            edge_size=64,
            resolution_m=10.0,
            overlap_meters=0.0,
        )


def test_patch_footprint_builds_bounds_around_center() -> None:
    patch = Patch(
        patch_id="patch_000001",
        latitude=0.0,
        longitude=0.0,
        edge_size=100,
        row_index=0,
        row_count=1,
        column_index=0,
        column_count=1,
    )

    footprint = patch_footprint(patch, resolution_m=10.0)

    assert footprint.bounds[0] < 0.0
    assert footprint.bounds[1] < 0.0
    assert footprint.bounds[2] > 0.0
    assert footprint.bounds[3] > 0.0


def test_raster_small_helpers_and_validity_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJ_LIB", "/existing/proj")
    ensure_proj_env()
    assert os.environ["PROJ_LIB"] == "/existing/proj"

    monkeypatch.delenv("PROJ_LIB", raising=False)
    monkeypatch.delenv("PROJ_DATA", raising=False)
    ensure_proj_env()
    assert os.environ["PROJ_LIB"]
    assert os.environ["PROJ_DATA"]

    assert guess_utm_epsg(45.0, 9.0) == 32632
    assert guess_utm_epsg(-45.0, 9.0) == 32732
    assert parse_epsg("EPSG:32633", 45.0, 9.0) == 32633
    assert parse_epsg("unknown", 45.0, 9.0) == 32632
    assert _as_scalar(type("Computed", (), {"compute": lambda self: "2.5"})()) == 2.5
    monkeypatch.delenv("PROJ_LIB", raising=False)
    monkeypatch.delenv("PROJ_DATA", raising=False)

    scaled_unit = scale_to_uint16(
        np.array([[[0.0, 0.5, 1.0, np.nan]]], dtype="float32")
    )
    scaled_reflectance = scale_to_uint16(
        np.array([[[-1.0, 50.0, 10001.0]]], dtype="float32")
    )
    scaled_integer = scale_to_uint16(np.array([[[1, 2, 3]]], dtype="int16"))
    assert scaled_unit.tolist() == [[[0, 5000, 10000, 0]]]
    assert scaled_reflectance.tolist() == [[[0, 50, 10000]]]
    assert scaled_integer.dtype == np.dtype("uint16")

    source = tmp_path / "source.tif"
    _write_tif(
        source,
        np.ones((1, 10, 10), dtype="uint16"),
        from_origin(9.0, 46.0, 0.1, 0.1),
        crs="EPSG:4326",
    )
    lat, lon = compute_centroid_lat_lon(source)
    assert lat == pytest.approx(45.5)
    assert lon == pytest.approx(9.5)

    no_crs = tmp_path / "no_crs.tif"
    _write_tif(
        no_crs,
        np.ones((1, 2, 2), dtype="uint16"),
        from_origin(0.0, 0.0, 1.0, 1.0),
        crs=None,
        nodata=None,
    )
    with pytest.raises(ValueError, match="lacks a CRS"):
        compute_centroid_lat_lon(no_crs)

    validity = tmp_path / "validity.tif"
    _write_tif(
        validity,
        np.array([[[0, 1], [2, 0]]], dtype="uint16"),
        from_origin(0.0, 2.0, 1.0, 1.0),
    )
    assert raster_validity_stats(validity) == {
        "total_pixels": 4,
        "valid_pixels": 2,
        "nonzero_pixels": 2,
    }


def _write_patch_output(run_dir: Path, patch_id: str, output_name: str) -> Path:
    output = run_dir / "patches" / patch_id / "outputs" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"patch")
    return output


def test_deliver_bbox_outputs_direct_and_nested_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run_001"
    write_yaml(run_dir / "run_manifest.yaml", {"start_date": "2025-01-01"})
    input_path = _write_patch_output(run_dir, "patch_000001", "custom_sr.tif")
    captured: dict[str, object] = {}

    def fake_merge_and_clip_bbox(*, input_paths, output_path, bbox, nodata=0):
        captured["input_paths"] = input_paths
        captured["bbox"] = bbox
        output_path.write_text("merged", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        delivery_module, "merge_and_clip_bbox", fake_merge_and_clip_bbox
    )
    destination, delivered = deliver_bbox_outputs(
        run_root=run_dir,
        bbox=(9.0, 45.0, 10.0, 46.0),
        output_name="custom_sr.tif",
    )
    assert destination == run_dir / "delivery_clipped"
    assert captured["input_paths"] == [input_path]
    assert delivered[0]["date"] == "2025-01-01"
    manifest = json.loads(
        (destination / "delivery_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bbox"] == [9.0, 45.0, 10.0, 46.0]

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    write_yaml(run_a / "run_manifest.yaml", {})
    write_yaml(run_b / "run_manifest.yaml", {"start_date": "2025-02-01"})
    _write_patch_output(run_a, "patch_000001", "fused_sr.tif")
    _write_patch_output(run_b, "patch_000001", "fused_sr.tif")
    _write_patch_output(run_dir, "patch_000001", "fused_sr.tif")
    nested_destination, nested = deliver_bbox_outputs(
        run_root=tmp_path,
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    assert nested_destination == tmp_path / "delivery_clipped"
    assert [Path(item["run_dir"]).name for item in nested] == [
        "run_001",
        "run_a",
        "run_b",
    ]
    assert nested[0]["date"] == "2025-01-01"
    assert nested[1]["date"] == "run_a"
    assert nested[2]["date"] == "2025-02-01"


def test_deliver_bbox_outputs_rejects_runs_without_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    write_yaml(run_dir / "run_manifest.yaml", {})

    with pytest.raises(FileNotFoundError, match="No fused_sr.tif files"):
        deliver_bbox_outputs(run_root=run_dir, bbox=(0.0, 0.0, 1.0, 1.0))


class FakeCoord:
    def __init__(self, values: list[str]) -> None:
        self.values = values


class FakeRio:
    def __init__(self, cube: "FakeCube") -> None:
        self.cube = cube

    def write_crs(self, epsg_code: int, inplace: bool = False):
        self.cube.written_crs = epsg_code
        return self.cube

    def write_nodata(self, nodata: int, encoded: bool = True, inplace: bool = False):
        self.cube.written_nodata = nodata
        return self.cube

    def to_raster(self, output_path: Path, **kwargs) -> None:
        self.cube.raster_kwargs = kwargs
        output_path.write_bytes(b"raster")


class FakeCube:
    def __init__(
        self,
        data: np.ndarray,
        dims: tuple[str, ...] = ("band", "y", "x"),
        coords: dict[str, object] | None = None,
        attrs: dict[str, object] | None = None,
    ) -> None:
        self.data = np.asarray(data)
        self.dims = dims
        self.coords = coords or {}
        self.attrs = attrs or {}
        self.sizes = {dim: size for dim, size in zip(dims, self.data.shape)}
        self.rio = FakeRio(self)

    def transpose(self, *dims: str):
        axes = [self.dims.index(dim) for dim in dims]
        return FakeCube(
            self.data.transpose(axes), tuple(dims), self.coords, dict(self.attrs)
        )

    def isel(self, **kwargs):
        time_index = kwargs["time"]
        axis = self.dims.index("time")
        dims = tuple(dim for dim in self.dims if dim != "time")
        return FakeCube(
            np.take(self.data, time_index, axis=axis),
            dims,
            self.coords,
            dict(self.attrs),
        )

    def copy(self, data=None):
        return FakeCube(
            self.data.copy() if data is None else data,
            self.dims,
            self.coords,
            dict(self.attrs),
        )


def test_staging_validation_and_selection_extra_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(SkipTileError, match="empty_cube"):
        ensure_cube_has_valid_data(types.SimpleNamespace(data=np.empty((1, 0, 0))))
    with pytest.raises(SkipTileError, match="all_nan_cube"):
        ensure_cube_has_valid_data(
            types.SimpleNamespace(data=np.full((1, 2, 2), np.nan))
        )
    with pytest.raises(SkipTileError, match="all_zero_cube"):
        ensure_cube_has_valid_data(types.SimpleNamespace(data=np.zeros((1, 2, 2))))

    no_time_cube = FakeCube(np.ones((1, 4, 4), dtype="uint16"))
    selected, diagnostics = _select_or_mosaic_time_items(no_time_cube, StagingConfig())
    assert selected is no_time_cube
    assert diagnostics["candidate_count"] == 1

    timed_cube = FakeCube(
        np.stack([np.zeros((1, 4, 4)), np.ones((1, 4, 4))]),
        dims=("time", "band", "y", "x"),
        coords={"time": FakeCoord(["first", "second"])},
    )
    selected, diagnostics = _select_or_mosaic_time_items(
        timed_cube,
        StagingConfig(item_strategy="fixed_index", image_index=1),
    )
    assert np.all(selected.data == 1)
    assert diagnostics["selected_labels"] == ["second"]

    with pytest.raises(IndexError, match="image_index=3"):
        _select_or_mosaic_time_items(
            timed_cube,
            StagingConfig(item_strategy="fixed_index", image_index=3),
        )

    sparse = np.zeros((1, 1, 4, 4), dtype="uint16")
    sparse[0, :, 1:3, 1:3] = 5
    with caplog.at_level("WARNING", logger="srgan-hpc"):
        _select_or_mosaic_time_items(
            FakeCube(sparse, dims=("time", "band", "y", "x")),
            StagingConfig(
                min_center_nonzero_fraction=0.0, min_full_nonzero_fraction=0.8
            ),
        )
    assert "Low full cutout coverage" in caplog.text

    monkeypatch.setattr(staging_module, "ensure_proj_env", lambda: None)
    monkeypatch.setattr(
        staging_module,
        "create_cube_with_retry",
        lambda **_kwargs: (
            FakeCube(np.ones((1, 2, 2), dtype="float32"), attrs={"epsg": "EPSG:32632"}),
            [{"id": "S2_ITEM"}],
        ),
    )
    output_path = tmp_path / "inputs" / "rgbnir.tif"
    metadata_path = tmp_path / "metadata" / "rgbnir.json"
    result = stage_cutout(
        latitude=45.0,
        longitude=9.0,
        start_date="2025-01-01",
        end_date="2025-01-02",
        config=StagingConfig(rate_limit_retry_delays_seconds=[]),
        bands=["B04"],
        edge_size=64,
        resolution=10,
        output_path=output_path,
        metadata_path=metadata_path,
        patch_id="patch_000001",
        product_name="rgbnir",
    )
    assert result == output_path.resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["auto_selected_items"] == [{"id": "S2_ITEM"}]
    assert payload["validity_stats"]["nonzero_pixels"] == 4


def _write_task_manifest(
    tmp_path: Path,
    *,
    mode: str,
    products: list[str],
) -> Path:
    config = RuntimeConfig(output_root=tmp_path / "runs", mode=mode)
    manifest_path = tmp_path / "patches" / "patch_000001" / "manifest.yaml"
    input_paths = {product: f"inputs/{product}.tif" for product in products}
    write_yaml(
        manifest_path,
        {
            "patch_id": "patch_000001",
            "products": products,
            "paths": {
                "inputs": input_paths,
                "output_dir": "outputs",
                "metadata_dir": "metadata",
            },
            "config": runtime_config_to_dict(config),
        },
    )
    for relative_path in input_paths.values():
        input_path = manifest_path.parent / relative_path
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b"input")
    return manifest_path


def test_run_task_skip_single_and_fused_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_task_manifest(
        tmp_path / "skip", mode="rgbnir", products=["rgbnir"]
    )
    monkeypatch.setattr(
        run_task_module,
        "raster_validity_stats",
        lambda _path: {"total_pixels": 4, "valid_pixels": 0, "nonzero_pixels": 0},
    )
    assert run_task(manifest_path) is None
    skip_payload = json.loads(
        (manifest_path.parent / "metadata" / "rgbnir_skip.json").read_text(
            encoding="utf-8"
        )
    )
    assert skip_payload["reason"] == "empty_input_raster"

    manifest_path = _write_task_manifest(
        tmp_path / "single", mode="rgbnir", products=["rgbnir"]
    )
    monkeypatch.setattr(
        run_task_module,
        "raster_validity_stats",
        lambda _path: {"total_pixels": 4, "valid_pixels": 4, "nonzero_pixels": 4},
    )

    def fake_run_inference(*, output_dir, product_name, **_kwargs):
        output = output_dir / f"{product_name}_sr.tif"
        output.write_bytes(b"sr")
        return output

    monkeypatch.setattr(run_task_module, "run_inference", fake_run_inference)
    result = run_task(manifest_path)
    assert result == manifest_path.parent / "outputs" / "rgbnir_sr.tif"
    result_payload = json.loads(
        (manifest_path.parent / "metadata" / "result.json").read_text(encoding="utf-8")
    )
    assert result_payload["status"] == "completed"

    fused_manifest = _write_task_manifest(
        tmp_path / "fused", mode="fused", products=["rgbnir", "swir"]
    )
    captured: dict[str, object] = {}

    def fake_stack_geotiffs(**kwargs):
        captured.update(kwargs)
        kwargs["output_path"].write_bytes(b"fused")
        return kwargs["output_path"]

    monkeypatch.setattr(run_task_module, "stack_geotiffs", fake_stack_geotiffs)
    fused_result = run_task(fused_manifest)
    assert fused_result == fused_manifest.parent / "outputs" / "fused_sr.tif"
    assert captured["band_names"] == [
        "B04",
        "B03",
        "B02",
        "B08",
        "B05",
        "B06",
        "B07",
        "B8A",
        "B11",
        "B12",
    ]

    missing_manifest = _write_task_manifest(
        tmp_path / "missing", mode="fused", products=["rgbnir"]
    )
    with pytest.raises(RuntimeError, match="missing: swir"):
        run_task(missing_manifest)


def test_submit_skip_paths_write_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stage_cutout(**_kwargs):
        raise SkipTileError("all_zero_cube", details={"total_pixels": 4})

    monkeypatch.setattr(submit_module, "stage_cutout", fake_stage_cutout)
    config = RuntimeConfig(output_root=tmp_path / "runs", mode="rgbnir")
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
        start_date="2025-01-01",
        end_date="2025-01-02",
        script_path=Path("/tmp/slurm.sh"),
    )
    assert submission["mode"] == "skipped"
    assert read_yaml(run_dir / "run_manifest.yaml")["tasks"] == []
    assert (
        read_yaml(run_dir / "patches" / "patch_000001" / "manifest.yaml")["status"]
        == "skipped"
    )

    _, grid_dir, grid_submission = submit_grid_run(
        config=config,
        patches=[patch],
        start_date="2025-01-01",
        end_date="2025-01-02",
        script_path=Path("/tmp/slurm.sh"),
    )
    assert grid_submission["reason"] == "no_submittable_patches"
    assert read_yaml(grid_dir / "run_manifest.yaml")["skipped_count"] == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config, tmp: setattr(config.staging, "edge_size", 0), "edge_size"),
        (
            lambda config, tmp: setattr(config.staging, "overlap_meters", -1),
            "overlap_meters",
        ),
        (
            lambda config, tmp: setattr(
                config.staging, "rate_limit_retry_delays_seconds", [0]
            ),
            "rate_limit",
        ),
        (
            lambda config, tmp: setattr(config.staging, "item_strategy", "bad"),
            "item_strategy",
        ),
        (
            lambda config, tmp: setattr(
                config.staging, "min_center_nonzero_fraction", 1.5
            ),
            "min_center",
        ),
        (
            lambda config, tmp: setattr(
                config.staging, "min_full_nonzero_fraction", -0.1
            ),
            "min_full",
        ),
        (
            lambda config, tmp: setattr(config.staging, "auto_select_item_limit", 0),
            "auto_select_item_limit",
        ),
        (
            lambda config, tmp: setattr(config.staging, "search_max_items", 0),
            "search_max_items",
        ),
        (
            lambda config, tmp: setattr(config.staging, "search_limit", 0),
            "search_limit",
        ),
        (
            lambda config, tmp: setattr(config.inference, "window_size", (128,)),
            "window_size",
        ),
        (lambda config, tmp: setattr(config.inference, "batch_size", 0), "batch_size"),
        (lambda config, tmp: setattr(config.slurm, "gpus", -1), "gpus"),
        (lambda config, tmp: setattr(config.slurm, "mem_gb", 0), "mem_gb"),
        (
            lambda config, tmp: setattr(config.slurm, "cpus_per_task", 0),
            "cpus_per_task",
        ),
        (
            lambda config, tmp: setattr(config.aoi, "path", str(tmp / "missing.shp")),
            "AOI path",
        ),
        (lambda config, tmp: setattr(config.rgbnir, "resolution", 0), "resolution"),
        (lambda config, tmp: setattr(config.rgbnir, "factor", 0), "factor"),
        (lambda config, tmp: setattr(config.rgbnir, "bands", []), "bands"),
        (
            lambda config, tmp: (
                setattr(config.rgbnir.model, "preset", None),
                setattr(config.rgbnir.model, "config_path", None),
            ),
            "model",
        ),
        (lambda config, tmp: setattr(config.staging, "edge_size", 3), "whole-number"),
    ],
)
def test_validate_runtime_config_rejects_invalid_values(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    config = RuntimeConfig(mode="rgbnir")
    if message == "whole-number":
        config.mode = "fused"
    mutate(config, tmp_path)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        validate_runtime_config(config)


def test_cli_main_dispatches_common_commands(
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
        sys, "argv", ["srgan-hpc", "validate-config", "--config", str(config_path)]
    )
    assert cli_module.main() == 0
    assert "Configuration valid" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srgan-hpc",
            "submit",
            "patch",
            "--config",
            str(config_path),
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--lat",
            "45.0",
            "--lon",
            "9.0",
            "--script-path",
            "/tmp/slurm.sh",
            "--dry-run",
        ],
    )
    assert cli_module.main() == 0
    assert '"run_id"' in capsys.readouterr().out

    run_dir = tmp_path / "status_run"
    (run_dir / "patches" / "patch_000001").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["srgan-hpc", "status", "--run-dir", str(run_dir)])
    assert cli_module.main() == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["patch_count"] == 1

    called: dict[str, object] = {}
    monkeypatch.setattr(
        "deployment.srgan_hpc.run_task.run_task",
        lambda manifest, task_index=None: called.update(
            {"manifest": manifest, "task_index": task_index}
        )
        or None,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("tasks: []\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["srgan-hpc", "run", "task", "--manifest", str(manifest)]
    )
    assert cli_module.main() == 0
    assert called["task_index"] == 4
    assert capsys.readouterr().out.strip() == "skipped"

    monkeypatch.setattr(
        "deployment.srgan_hpc.delivery.deliver_bbox_outputs",
        lambda **_kwargs: (tmp_path / "delivery", [{"output": "out.tif"}]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srgan-hpc",
            "deliver-bbox",
            "--run-root",
            str(tmp_path),
            "--west",
            "0",
            "--south",
            "0",
            "--east",
            "1",
            "--north",
            "1",
        ],
    )
    assert cli_module.main() == 0
    assert "out.tif" in capsys.readouterr().out
