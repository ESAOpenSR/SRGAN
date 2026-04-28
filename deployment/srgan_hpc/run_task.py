from __future__ import annotations

import logging
from pathlib import Path

from deployment.srgan_hpc.config import InferenceConfig, ProductConfig
from deployment.srgan_hpc.inference import run_inference
from deployment.srgan_hpc.manifests import read_yaml, write_json
from deployment.srgan_hpc.metadata import write_software_metadata
from deployment.srgan_hpc.naming import fused_output_name
from deployment.srgan_hpc.raster import raster_validity_stats, stack_geotiffs


LOGGER = logging.getLogger("srgan-hpc")


def _resolve_patch_manifest(manifest_path: Path, task_index: int | None) -> dict:
    manifest = read_yaml(manifest_path)
    if "tasks" not in manifest:
        return manifest
    if task_index is None:
        raise ValueError("Array task manifest requires task index")
    task_entries = manifest["tasks"]
    task = task_entries[task_index]
    return read_yaml((manifest_path.parent / Path(task["manifest"])).resolve())


def _resolve_manifest_local_path(manifest_path: Path, relative_path: str) -> Path:
    return (manifest_path.parent / relative_path).resolve()


def _product_from_dict(data: dict) -> ProductConfig:
    from deployment.srgan_hpc.config import ModelSourceConfig

    model = data["model"]
    return ProductConfig(
        bands=list(data["bands"]),
        resolution=int(data["resolution"]),
        factor=int(data["factor"]),
        model=ModelSourceConfig(
            preset=model.get("preset"),
            config_path=model.get("config_path"),
            checkpoint_path=model.get("checkpoint_path"),
            cache_dir=model.get("cache_dir"),
        ),
    )


def _inference_from_dict(data: dict) -> InferenceConfig:
    return InferenceConfig(
        window_size=tuple(data["window_size"]),
        batch_size=data.get("batch_size", 16),
        overlap=data["overlap"],
        eliminate_border_px=data["eliminate_border_px"],
        gpus=data["gpus"],
        save_preview=data["save_preview"],
    )


def _skip_empty_product(metadata_dir: Path, product_name: str, input_tif: Path, validity: dict[str, int]) -> None:
    write_json(
        metadata_dir / f"{product_name}_skip.json",
        {
            "status": "skipped",
            "reason": "empty_input_raster",
            "product": product_name,
            "input_tif": str(input_tif),
            **validity,
        },
    )


def run_task(manifest_path: Path, task_index: int | None = None) -> Path | None:
    manifest_path = manifest_path.resolve()
    root_manifest = read_yaml(manifest_path)
    if "tasks" in root_manifest:
        if task_index is None:
            raise ValueError("Array task manifest requires task index")
        manifest_path = (manifest_path.parent / Path(root_manifest["tasks"][task_index]["manifest"])).resolve()
    manifest = _resolve_patch_manifest(manifest_path, None)
    config = manifest["config"]
    output_dir = _resolve_manifest_local_path(manifest_path, manifest["paths"]["output_dir"])
    metadata_dir = _resolve_manifest_local_path(manifest_path, manifest["paths"]["metadata_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    patch_id = str(manifest.get("patch_id", "unknown"))
    inference = _inference_from_dict(config["inference"])

    outputs: dict[str, str] = {}
    band_names: list[str] = []
    LOGGER.info("starting worker task patch_id=%s manifest=%s mode=%s", patch_id, manifest_path, config["mode"])

    for product_name in manifest["products"]:
        product = _product_from_dict(config[product_name])
        input_tif = _resolve_manifest_local_path(manifest_path, manifest["paths"]["inputs"][product_name])
        validity = raster_validity_stats(input_tif)
        if validity["valid_pixels"] == 0 or validity["nonzero_pixels"] == 0:
            LOGGER.info("skipping patch_id=%s product=%s because input is empty stats=%s", patch_id, product_name, validity)
            _skip_empty_product(metadata_dir, product_name, input_tif, validity)
            write_software_metadata(metadata_dir / "software_env.json")
            return None

        outputs[product_name] = str(
            run_inference(
                input_tif=input_tif,
                output_dir=output_dir,
                product_name=product_name,
                product=product,
                inference=inference,
            )
        )
        band_names.extend(product.bands)

    final_output: Path
    if config["mode"] == "fused":
        final_output = output_dir / fused_output_name()
        stack_geotiffs(
            reference_path=Path(outputs["rgbnir"]),
            secondary_path=Path(outputs["swir"]),
            output_path=final_output,
            band_names=band_names,
        )
    else:
        final_output = Path(next(iter(outputs.values())))

    write_json(
        metadata_dir / "result.json",
        {
            "status": "completed",
            "mode": config["mode"],
            "outputs": outputs,
            "final_output": str(final_output),
            "band_names": band_names,
        },
    )
    write_software_metadata(metadata_dir / "software_env.json")
    LOGGER.info("completed worker task patch_id=%s output=%s", patch_id, final_output)
    return final_output
