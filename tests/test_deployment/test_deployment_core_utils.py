from __future__ import annotations

import json
import logging
from pathlib import Path

from deployment.srgan_hpc.logging_utils import configure_logging
from deployment.srgan_hpc.metadata import write_software_metadata


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
