from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from deployment.srgan_hpc.config import StagingConfig
from deployment.srgan_hpc.staging import (
    _auto_select_item_ids,
    candidate_coverage_report,
    create_cube_with_retry,
    is_retryable_staging_error,
    mosaic_candidates_by_valid_pixels,
    order_candidates_by_coverage,
)


class ResponseError(Exception):
    def __init__(self, status_code: int) -> None:
        self.response = type("Response", (), {"status_code": status_code})()
        super().__init__(f"HTTP {status_code}")


class FakeItem:
    def __init__(self, item_id: str, tile: str, cloud_cover: float) -> None:
        self.id = item_id
        self.properties = {
            "s2:mgrs_tile": tile,
            "eo:cloud_cover": cloud_cover,
            "datetime": "2024-08-13T10:15:59Z",
        }


def install_fake_stac(monkeypatch: pytest.MonkeyPatch, items: list[FakeItem]) -> None:
    class FakeSearch:
        def items(self) -> list[FakeItem]:
            return items

    class FakeCatalog:
        def search(self, **_kwargs):
            return FakeSearch()

    client = types.SimpleNamespace(open=lambda _url: FakeCatalog())
    monkeypatch.setitem(sys.modules, "pystac_client", types.SimpleNamespace(Client=client))


def test_retryable_staging_error_detects_rate_limit_status() -> None:
    assert is_retryable_staging_error(ResponseError(429))


def test_retryable_staging_error_detects_planetary_computer_timeout() -> None:
    error = RuntimeError(
        "The request exceeded the maximum allowed time, please try again."
    )

    assert is_retryable_staging_error(error)


def test_retryable_staging_error_rejects_unrelated_errors() -> None:
    assert not is_retryable_staging_error(RuntimeError("invalid asset href"))


def test_mosaic_valid_strategy_prefers_center_covered_candidate() -> None:
    data = np.zeros((2, 1, 8, 8), dtype="uint16")
    data[0, :, :, 7:] = 10
    data[1, :, :, :] = 20

    order = order_candidates_by_coverage(data)
    mosaic = mosaic_candidates_by_valid_pixels(data, order)

    assert order == [1, 0]
    assert np.all(mosaic == 20)


def test_mosaic_valid_strategy_patches_missing_pixels_from_second_candidate() -> None:
    data = np.zeros((2, 1, 8, 8), dtype="uint16")
    data[0, :, :, :4] = 10
    data[1, :, :, 4:] = 20

    mosaic = mosaic_candidates_by_valid_pixels(data, [0, 1])

    assert np.all(mosaic[:, :, :4] == 10)
    assert np.all(mosaic[:, :, 4:] == 20)
    assert candidate_coverage_report(mosaic)["full_nonzero_fraction"] == pytest.approx(1.0)


def test_auto_select_item_ids_returns_all_stac_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_stac(
        monkeypatch,
        [
            FakeItem("item-a", "32UPA", 1.2),
            FakeItem("item-b", "32UPB", 0.3),
            FakeItem("item-c", "33UUU", 2.4),
        ],
    )

    item_ids, reports = _auto_select_item_ids(
        latitude=50.1,
        longitude=15.1,
        start_date="2024-08-12",
        end_date="2024-08-14",
        config=StagingConfig(auto_select_item=True, auto_select_item_limit=3),
        patch_id="patch_000001",
        product_name="rgbnir",
    )

    assert item_ids == ["item-a", "item-b", "item-c"]
    assert [report["tile"] for report in reports] == ["32UPA", "32UPB", "33UUU"]


def test_create_cube_with_retry_passes_all_auto_selected_ids_to_cubo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_stac(
        monkeypatch,
        [
            FakeItem("item-a", "32UPA", 1.2),
            FakeItem("item-b", "32UPB", 0.3),
        ],
    )
    captured_kwargs: dict[str, object] = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "cubo", types.SimpleNamespace(create=fake_create))
    monkeypatch.setitem(sys.modules, "rioxarray", types.SimpleNamespace())

    cube, reports = create_cube_with_retry(
        latitude=50.1,
        longitude=15.1,
        start_date="2024-08-12",
        end_date="2024-08-14",
        config=StagingConfig(
            auto_select_item=True,
            auto_select_item_limit=2,
            rate_limit_retry_delays_seconds=[],
        ),
        bands=["B04", "B03"],
        edge_size=4096,
        resolution=10,
        patch_id="patch_000001",
        product_name="rgbnir",
    )

    assert cube is not None
    assert captured_kwargs["ids"] == ["item-a", "item-b"]
    assert captured_kwargs["max_items"] == 2
    assert captured_kwargs["limit"] == 2
    assert [report["id"] for report in reports] == ["item-a", "item-b"]
