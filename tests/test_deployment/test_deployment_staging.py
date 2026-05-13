from __future__ import annotations

import numpy as np
import pytest

from deployment.srgan_hpc.staging import (
    candidate_coverage_report,
    is_retryable_staging_error,
    mosaic_candidates_by_valid_pixels,
    order_candidates_by_coverage,
)


class ResponseError(Exception):
    def __init__(self, status_code: int) -> None:
        self.response = type("Response", (), {"status_code": status_code})()
        super().__init__(f"HTTP {status_code}")


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
