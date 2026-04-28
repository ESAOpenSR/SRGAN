from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


DEFAULT_RATE_LIMIT_RETRY_DELAYS_SECONDS = [15, 30, 60, 120, 120, 120]
RuntimeMode = Literal["rgbnir", "swir", "fused"]


@dataclass(slots=True)
class EnvironmentConfig:
    python_executable: str = "python"
    modules: list[str] = field(default_factory=list)
    conda_env: str | None = None


@dataclass(slots=True)
class ModelSourceConfig:
    preset: str | None
    config_path: str | None = None
    checkpoint_path: str | None = None
    cache_dir: str | None = None


@dataclass(slots=True)
class ProductConfig:
    bands: list[str]
    resolution: int
    factor: int
    model: ModelSourceConfig


@dataclass(slots=True)
class AoiConfig:
    path: str | None = None
    layer: str | None = None


@dataclass(slots=True)
class StagingConfig:
    collection: str = "sentinel-2-l2a"
    image_index: int = 0
    edge_size: int = 4096
    nodata: int = 0
    output_dtype: str = "uint16"
    compression: str = "deflate"
    overlap_meters: float = 128.0
    retry_on_rate_limit: bool = True
    rate_limit_retry_delays_seconds: list[int] = field(
        default_factory=lambda: list(DEFAULT_RATE_LIMIT_RETRY_DELAYS_SECONDS)
    )


@dataclass(slots=True)
class InferenceConfig:
    window_size: tuple[int, int] = (128, 128)
    batch_size: int = 16
    overlap: int = 12
    eliminate_border_px: int = 2
    gpus: int | list[int] = 0
    save_preview: bool = False


@dataclass(slots=True)
class SlurmConfig:
    partition: str | None = None
    gpu_type: str | None = None
    gpus: int = 1
    gres: str | None = None
    cpus_per_task: int = 8
    mem_gb: int = 128
    time: str = "01:00:00"
    account: str | None = None
    qos: str | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeConfig:
    project_name: str = "srgan"
    output_root: Path = Path("runs")
    mode: RuntimeMode = "fused"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    aoi: AoiConfig = field(default_factory=AoiConfig)
    staging: StagingConfig = field(default_factory=StagingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rgbnir: ProductConfig = field(
        default_factory=lambda: ProductConfig(
            bands=["B04", "B03", "B02", "B08"],
            resolution=10,
            factor=4,
            model=ModelSourceConfig(preset="RGB-NIR"),
        )
    )
    swir: ProductConfig = field(
        default_factory=lambda: ProductConfig(
            bands=["B05", "B06", "B07", "B8A", "B11", "B12"],
            resolution=20,
            factor=8,
            model=ModelSourceConfig(preset="SWIR"),
        )
    )
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    config_path: Path | None = None


def enabled_product_names(config: RuntimeConfig) -> list[str]:
    if config.mode == "rgbnir":
        return ["rgbnir"]
    if config.mode == "swir":
        return ["swir"]
    return ["rgbnir", "swir"]


def get_product_config(config: RuntimeConfig, product_name: str) -> ProductConfig:
    if product_name == "rgbnir":
        return config.rgbnir
    if product_name == "swir":
        return config.swir
    raise ValueError(f"Unknown product: {product_name}")


def patch_resolution(config: RuntimeConfig) -> int:
    products = [get_product_config(config, name) for name in enabled_product_names(config)]
    return min(product.resolution for product in products)


def product_edge_size(config: RuntimeConfig, product: ProductConfig) -> int:
    selected_resolution = patch_resolution(config)
    edge_size = config.staging.edge_size * selected_resolution / product.resolution
    rounded = int(round(edge_size))
    if rounded <= 0 or abs(edge_size - rounded) > 1e-6:
        raise ValueError(
            "staging.edge_size must describe a whole-number cutout at every enabled product resolution"
        )
    return rounded


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _model_source_from_mapping(data: dict[str, Any], default_preset: str) -> ModelSourceConfig:
    return ModelSourceConfig(
        preset=data.get("preset", default_preset),
        config_path=data.get("config_path"),
        checkpoint_path=data.get("checkpoint_path"),
        cache_dir=data.get("cache_dir"),
    )


def _product_from_mapping(data: dict[str, Any], default: ProductConfig) -> ProductConfig:
    model_data = data.get("model", {})
    legacy_config_path = data.get("config_path")
    legacy_checkpoint_path = data.get("checkpoint_path")
    if legacy_config_path is not None or legacy_checkpoint_path is not None:
        model_data = {
            **model_data,
            "config_path": legacy_config_path or model_data.get("config_path"),
            "checkpoint_path": legacy_checkpoint_path or model_data.get("checkpoint_path"),
        }
    return ProductConfig(
        bands=list(data.get("bands", default.bands)),
        resolution=int(data.get("resolution", default.resolution)),
        factor=int(data.get("factor", default.factor)),
        model=_model_source_from_mapping(model_data, default.model.preset or ""),
    )


def _runtime_from_mapping(data: dict[str, Any]) -> RuntimeConfig:
    defaults = RuntimeConfig()
    environment = EnvironmentConfig(**data.get("environment", {}))
    aoi = AoiConfig(**data.get("aoi", {}))
    staging = StagingConfig(**data.get("staging", {}))
    inference_data = dict(data.get("inference", {}))
    if "window_size" in inference_data:
        inference_data["window_size"] = tuple(inference_data["window_size"])
    inference = InferenceConfig(**inference_data)
    slurm = SlurmConfig(**data.get("slurm", {}))
    return RuntimeConfig(
        project_name=data.get("project_name", defaults.project_name),
        output_root=Path(data.get("output_root", "runs")),
        mode=data.get("mode", defaults.mode),
        environment=environment,
        aoi=aoi,
        staging=staging,
        inference=inference,
        rgbnir=_product_from_mapping(data.get("rgbnir", {}), defaults.rgbnir),
        swir=_product_from_mapping(data.get("swir", {}), defaults.swir),
        slurm=slurm,
    )


def _resolve_optional_path(value: str | None, base_dir: Path, *, must_exist: bool = False) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return str(path)


def _resolve_product_paths(product: ProductConfig, base_dir: Path) -> None:
    product.model.config_path = _resolve_optional_path(product.model.config_path, base_dir, must_exist=True)
    product.model.checkpoint_path = _resolve_optional_path(product.model.checkpoint_path, base_dir, must_exist=True)
    product.model.cache_dir = _resolve_optional_path(product.model.cache_dir, base_dir)


def load_runtime_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> RuntimeConfig:
    path = Path(config_path).expanduser().resolve()
    data = _read_yaml(path)
    if overrides:
        data = _merge(data, overrides)
    config = _runtime_from_mapping(data)
    config.config_path = path
    base_dir = path.parent

    if not config.output_root.is_absolute():
        config.output_root = (base_dir / config.output_root).resolve()

    if config.aoi.path is not None:
        config.aoi.path = _resolve_optional_path(config.aoi.path, base_dir)

    _resolve_product_paths(config.rgbnir, base_dir)
    _resolve_product_paths(config.swir, base_dir)
    validate_runtime_config(config)
    return config


def validate_runtime_config(config: RuntimeConfig) -> None:
    if config.mode not in {"rgbnir", "swir", "fused"}:
        raise ValueError("mode must be one of: rgbnir, swir, fused")
    if config.staging.edge_size <= 0:
        raise ValueError("staging.edge_size must be positive")
    if config.staging.overlap_meters < 0:
        raise ValueError("staging.overlap_meters must be non-negative")
    if any(delay <= 0 for delay in config.staging.rate_limit_retry_delays_seconds):
        raise ValueError("staging.rate_limit_retry_delays_seconds must contain positive integers")
    if len(config.inference.window_size) != 2 or min(config.inference.window_size) <= 0:
        raise ValueError("inference.window_size must contain two positive integers")
    if config.inference.batch_size <= 0:
        raise ValueError("inference.batch_size must be positive")
    if config.slurm.gpus < 0:
        raise ValueError("slurm.gpus must be non-negative")
    if config.slurm.mem_gb <= 0:
        raise ValueError("slurm.mem_gb must be positive")
    if config.slurm.cpus_per_task <= 0:
        raise ValueError("slurm.cpus_per_task must be positive")
    if config.aoi.path is not None and not Path(config.aoi.path).exists():
        raise FileNotFoundError(f"AOI path not found: {config.aoi.path}")

    for product_name in enabled_product_names(config):
        product = get_product_config(config, product_name)
        if product.resolution <= 0:
            raise ValueError(f"{product_name}.resolution must be positive")
        if product.factor <= 0:
            raise ValueError(f"{product_name}.factor must be positive")
        if not product.bands:
            raise ValueError(f"{product_name}.bands must not be empty")
        if product.model.preset is None and product.model.config_path is None:
            raise ValueError(f"{product_name}.model must define a preset or config_path")
        product_edge_size(config, product)


def runtime_config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_root"] = str(config.output_root)
    data["config_path"] = str(config.config_path) if config.config_path else None
    data["inference"]["window_size"] = list(config.inference.window_size)
    return data
