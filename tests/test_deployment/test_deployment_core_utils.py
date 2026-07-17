from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from deployment.srgan_hpc.checkpoint import resolve_checkpoint_path, sha256sum
from deployment.srgan_hpc.logging_utils import configure_logging
from deployment.srgan_hpc.metadata import write_software_metadata


def test_resolve_checkpoint_path_accepts_none_and_existing_file(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")

    assert resolve_checkpoint_path(None) is None
    assert resolve_checkpoint_path(str(checkpoint)) == checkpoint.resolve()


def test_resolve_checkpoint_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_checkpoint_path(str(tmp_path / "missing.ckpt"))


def test_sha256sum_streams_file_contents(tmp_path: Path) -> None:
    payload = b"abc" + (b"0123456789" * 200_000)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(payload)

    assert sha256sum(checkpoint) == hashlib.sha256(payload).hexdigest()


def test_configure_logging_sets_stream_and_file_handlers(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "run.log"

    logger = configure_logging(log_path=log_path, verbose=True)
    logger.info("hello logging")
    for handler in logger.handlers:
        handler.flush()

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2
    assert "hello logging" in log_path.read_text(encoding="utf-8")


def test_configure_logging_replaces_existing_handlers() -> None:
    logger = configure_logging(verbose=False)
    first_handlers = list(logger.handlers)

    logger = configure_logging(verbose=False)

    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert logger.handlers != first_handlers


def test_write_software_metadata_includes_environment_and_extra_fields(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata" / "software.json"

    write_software_metadata(metadata_path, extra={"run_id": "srgan_001"})

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "srgan_001"
    assert "python_version" in payload
    assert "platform" in payload
