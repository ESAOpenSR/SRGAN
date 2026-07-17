from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from deployment.srgan_hpc.config import StagingConfig
from deployment.srgan_hpc.staging import (
    SkipTileError,
    _auto_select_item_ids,
    _format_cloud_cover_range,
    _search_kwargs_from_config,
    _select_or_mosaic_time_items,
    candidate_coverage_report,
    create_cube_with_retry,
    is_rate_limit_error,
    mosaic_candidates_by_valid_pixels,
)
from tests.test_deployment.test_deployment_extra_paths2 import FakeCube


def test_staging_coverage_helpers_handle_empty_and_invalid_inputs() -> None:
    assert candidate_coverage_report(np.empty((1, 0, 0))) == {
        "full_nonzero_fraction": 0.0,
        "center_nonzero_fraction": 0.0,
    }

    with pytest.raises(ValueError, match="Expected candidate data"):
        mosaic_candidates_by_valid_pixels(np.zeros((1, 2, 2)), [0])

    with pytest.raises(ValueError, match="must not be empty"):
        mosaic_candidates_by_valid_pixels(np.zeros((1, 1, 2, 2)), [])


def test_staging_small_private_helpers() -> None:
    assert is_rate_limit_error(RuntimeError("too many requests"))
    assert (
        _format_cloud_cover_range([{"cloud_cover": None}, {"cloud_cover": "bad"}])
        == "n/a"
    )
    assert _format_cloud_cover_range([{"cloud_cover": "1.25"}]) == "1.250%"
    assert (
        _format_cloud_cover_range([{"cloud_cover": 3}, {"cloud_cover": 1}])
        == "1.000-3.000%"
    )

    assert _search_kwargs_from_config(
        StagingConfig(
            search_query={"eo:cloud_cover": {"lt": 10}},
            search_max_items=3,
            search_limit=2,
        )
    ) == {
        "query": {"eo:cloud_cover": {"lt": 10}},
        "max_items": 3,
        "limit": 2,
    }


def test_auto_select_item_ids_skips_when_stac_search_returns_no_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptySearch:
        def items(self) -> list[object]:
            return []

    class FakeCatalog:
        def search(self, **_kwargs):
            return EmptySearch()

    monkeypatch.setitem(
        sys.modules,
        "pystac_client",
        types.SimpleNamespace(
            Client=types.SimpleNamespace(open=lambda _url: FakeCatalog())
        ),
    )

    with pytest.raises(SkipTileError, match="no_stac_items") as exc_info:
        _auto_select_item_ids(
            latitude=45.0,
            longitude=9.0,
            start_date="2025-01-01",
            end_date="2025-01-02",
            config=StagingConfig(auto_select_item=True),
        )

    assert exc_info.value.details == {"latitude": 45_000_000, "longitude": 9_000_000}


def test_create_cube_with_retry_returns_from_retry_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    cube = object()

    def fake_create(**kwargs):
        captured.update(kwargs)
        return cube

    monkeypatch.setitem(sys.modules, "cubo", types.SimpleNamespace(create=fake_create))
    monkeypatch.setitem(sys.modules, "rioxarray", types.SimpleNamespace())

    result, reports = create_cube_with_retry(
        latitude=45.0,
        longitude=9.0,
        start_date="2025-01-01",
        end_date="2025-01-02",
        config=StagingConfig(rate_limit_retry_delays_seconds=[1]),
        bands=["B04"],
        edge_size=64,
        resolution=10,
    )

    assert result is cube
    assert reports == []
    assert captured["bands"] == ["B04"]
    assert captured["edge_size"] == 64


def test_select_or_mosaic_time_items_empty_and_low_center_paths() -> None:
    with pytest.raises(SkipTileError, match="empty_cube"):
        _select_or_mosaic_time_items(
            FakeCube(np.empty((0, 1, 4, 4)), dims=("time", "band", "y", "x")),
            StagingConfig(),
        )

    with pytest.raises(SkipTileError, match="low_center_coverage") as exc_info:
        _select_or_mosaic_time_items(
            FakeCube(np.zeros((1, 1, 4, 4)), dims=("time", "band", "y", "x")),
            StagingConfig(min_center_nonzero_fraction=0.5),
        )

    assert exc_info.value.details == {
        "candidate_count": 1,
        "center_nonzero_percent": 0,
    }


def test_select_or_mosaic_time_items_falls_back_to_index_label() -> None:
    class BrokenCoord:
        @property
        def values(self):
            raise RuntimeError("cannot read labels")

    data = np.ones((1, 1, 4, 4), dtype="uint16")
    _, diagnostics = _select_or_mosaic_time_items(
        FakeCube(
            data,
            dims=("time", "band", "y", "x"),
            coords={"time": BrokenCoord()},
        ),
        StagingConfig(item_strategy="fixed_index", image_index=0),
    )

    assert diagnostics["selected_labels"] == ["0"]
