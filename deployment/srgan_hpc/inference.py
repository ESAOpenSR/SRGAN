from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch

from deployment.srgan_hpc.config import InferenceConfig, ProductConfig
from deployment.srgan_hpc.naming import product_output_name
from deployment.srgan_hpc.raster import compress_geotiff


LOGGER = logging.getLogger("srgan-hpc")


def _build_runner(opensr_utils: Any, inference: InferenceConfig):
    class ConfigurableLargeFileProcessing(opensr_utils.large_file_processing):
        def __init__(self, *args: Any, batch_size: int, **kwargs: Any) -> None:
            self._srgan_hpc_batch_size = batch_size
            super().__init__(*args, **kwargs)

        def create_datamodule(self) -> None:
            from opensr_utils.data_utils.datamodule import PredictionDataModule

            dm = PredictionDataModule(
                input_type=self.input_type,
                root=self.root,
                windows=self.image_meta["image_windows"],
                lr_file_dict=self.image_meta["lr_file_dict"],
                prefetch_factor=2,
                batch_size=self._srgan_hpc_batch_size,
                num_workers=4,
            )
            dm.setup()
            self._log(
                f"Created PredictionDataModule with {len(dm.dataset)} patches "
                f"(batch_size={self._srgan_hpc_batch_size})."
            )
            self.datamodule = dm

    return ConfigurableLargeFileProcessing


def load_srgan_model(product: ProductConfig):
    from opensr_srgan import load_from_config, load_inference_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_location = torch.device(device)
    if product.model.config_path is not None:
        return load_from_config(
            product.model.config_path,
            product.model.checkpoint_path,
            map_location=map_location,
            mode="eval",
        ).to(device), device
    if product.model.preset is None:
        raise ValueError("Product model source must define preset or config_path")
    return load_inference_model(
        product.model.preset,
        cache_dir=product.model.cache_dir,
        map_location=map_location,
    ).to(device), device


def run_inference(
    *,
    input_tif: Path,
    output_dir: Path,
    product_name: str,
    product: ProductConfig,
    inference: InferenceConfig,
) -> Path:
    try:
        import opensr_utils
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "srgan-hpc inference requires opensr-utils. Install with `pip install \"opensr-srgan[hpc]\"`."
        ) from exc

    model, device = load_srgan_model(product)
    LOGGER.info(
        "running SRGAN product=%s input_tif=%s device=%s bands=%s factor=%s",
        product_name,
        input_tif,
        device,
        product.bands,
        product.factor,
    )
    runner_cls = _build_runner(opensr_utils, inference)
    runner = runner_cls(
        root=str(input_tif),
        model=model,
        window_size=tuple(inference.window_size),
        batch_size=inference.batch_size,
        factor=product.factor,
        overlap=inference.overlap,
        eliminate_border_px=inference.eliminate_border_px,
        device=device,
        gpus=cast(Any, inference.gpus),
        save_preview=inference.save_preview,
        debug=False,
    )
    runner.start_super_resolution()

    final_sr_path = getattr(runner, "final_sr_path", None)
    if final_sr_path is None:
        final_sr_path = output_dir / "sr.tif"
    else:
        final_sr_path = Path(final_sr_path)

    if not final_sr_path.exists():
        raise FileNotFoundError(f"Expected SR output at {final_sr_path}")

    compressed_path = output_dir / product_output_name(product_name)
    LOGGER.info("compressing SR GeoTIFF product=%s input=%s output=%s", product_name, final_sr_path, compressed_path)
    compress_geotiff(final_sr_path, compressed_path, band_names=product.bands)
    final_sr_path.unlink(missing_ok=True)
    return compressed_path
