from __future__ import annotations

from deployment.srgan_hpc.staging import is_retryable_staging_error


class ResponseError(Exception):
    def __init__(self, status_code: int) -> None:
        self.response = type("Response", (), {"status_code": status_code})()
        super().__init__(f"HTTP {status_code}")


def test_retryable_staging_error_detects_rate_limit_status() -> None:
    assert is_retryable_staging_error(ResponseError(429))


def test_retryable_staging_error_detects_planetary_computer_timeout() -> None:
    error = RuntimeError("The request exceeded the maximum allowed time, please try again.")

    assert is_retryable_staging_error(error)


def test_retryable_staging_error_rejects_unrelated_errors() -> None:
    assert not is_retryable_staging_error(RuntimeError("invalid asset href"))
