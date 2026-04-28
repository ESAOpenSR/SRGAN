from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def resolve_checkpoint_path(configured_path: str | None) -> Path | None:
    if configured_path is None:
        return None
    path = Path(configured_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def sha256sum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
