from __future__ import annotations

from pathlib import Path

import pytest

from deployment.srgan_hpc.config import (
    RuntimeConfig,
    enabled_product_names,
    load_runtime_config,
    patch_resolution,
    product_edge_size,
)


def test_default_config_enables_fused_products() -> None:
    config = load_runtime_config(Path("deployment/configs/runtime.default.yaml"))

    assert config.mode == "fused"
    assert enabled_product_names(config) == ["rgbnir", "swir"]
    assert patch_resolution(config) == 10
    assert product_edge_size(config, config.rgbnir) == 4096
    assert product_edge_size(config, config.swir) == 2048


def test_single_product_modes_select_expected_products() -> None:
    rgbnir = RuntimeConfig(mode="rgbnir")
    swir = RuntimeConfig(mode="swir")

    assert enabled_product_names(rgbnir) == ["rgbnir"]
    assert enabled_product_names(swir) == ["swir"]
    assert patch_resolution(swir) == 20


def test_invalid_mode_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("mode: bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mode must be one of"):
        load_runtime_config(config_path)
