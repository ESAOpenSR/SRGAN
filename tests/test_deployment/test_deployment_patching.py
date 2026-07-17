from __future__ import annotations

import pytest

from deployment.srgan_hpc.patching import (
    build_patches,
    clamp_center,
    compute_centers,
    meters_to_lat_deg,
    meters_to_lon_deg,
)


def test_patch_degree_helpers_and_clamping() -> None:
    assert meters_to_lat_deg(111_320.0) == pytest.approx(1.0)
    assert meters_to_lon_deg(111_320.0, 0.0) == pytest.approx(1.0)
    assert clamp_center(0.0, -1.0, 1.0, 0.25) == 0.0
    assert clamp_center(-2.0, -1.0, 1.0, 0.25) == -0.75
    assert clamp_center(2.0, -1.0, 1.0, 0.25) == 0.75
    assert clamp_center(5.0, 0.0, 1.0, 2.0) == 0.5


def test_meters_to_lon_deg_rejects_polar_latitudes() -> None:
    with pytest.raises(ValueError, match="Cannot compute longitude"):
        meters_to_lon_deg(10.0, 90.000001)


def test_compute_centers_handles_small_span_and_multiple_steps() -> None:
    assert compute_centers(0.0, 0.1, patch_deg=1.0, step_deg=0.5) == [0.05]

    centers = compute_centers(0.0, 3.0, patch_deg=1.0, step_deg=0.75)

    assert centers == pytest.approx([0.5, 1.25, 2.0, 2.5])


def test_compute_centers_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_centers(0.0, 1.0, patch_deg=0.0, step_deg=1.0)

    with pytest.raises(ValueError, match="must be positive"):
        compute_centers(0.0, 1.0, patch_deg=1.0, step_deg=0.0)


def test_build_patches_assigns_grid_metadata() -> None:
    patches = build_patches(
        lat1=0.0,
        lon1=0.0,
        lat2=0.05,
        lon2=0.05,
        edge_size=100,
        resolution_m=10.0,
        overlap_meters=0.0,
    )

    assert len(patches) > 1
    assert patches[0].patch_id == "patch_000001"
    assert patches[0].row_index == 0
    assert patches[0].column_index == 0
    assert patches[-1].patch_id == f"patch_{len(patches):06d}"
    assert patches[-1].row_count >= 1
    assert patches[-1].column_count >= 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lon1": -179.0, "lon2": 179.5}, "antimeridian"),
        ({"edge_size": 0}, "edge_size"),
        ({"resolution_m": 0.0}, "resolution_m"),
        ({"overlap_meters": 1000.0}, "overlap_meters"),
    ],
)
def test_build_patches_validates_inputs(kwargs: dict[str, float], message: str) -> None:
    params = {
        "lat1": 0.0,
        "lon1": 0.0,
        "lat2": 0.01,
        "lon2": 0.01,
        "edge_size": 100,
        "resolution_m": 10.0,
        "overlap_meters": 0.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_patches(**params)
