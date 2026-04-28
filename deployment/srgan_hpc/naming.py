from __future__ import annotations

from pathlib import Path


def patch_output_name(latitude: float, longitude: float) -> str:
    return f"output_SR_image_{latitude:.6f}_{longitude:.6f}.tif"


def product_output_name(product_name: str) -> str:
    return f"{product_name}_sr.tif"


def fused_output_name() -> str:
    return "fused_sr.tif"


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    return output_root / run_id


def patch_dir(run_dir: Path, patch_id: str) -> Path:
    return run_dir / "patches" / patch_id
