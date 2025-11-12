# inference.py
"""
End-to-end inference utilities for Sentinel-2 Super-Resolution with SRGAN.

This module provides:
  - A helper to build and load an SRGAN model from a configuration and checkpoint.
  - A convenience wrapper to run patch-based Sentinel-2 inference using
    the `opensr_utils.large_file_processing` interface.
  - A simple `main()` entry point for standalone command-line runs.

Typical usage
-------------
>>> from opensr_srgan.inference import run_sen2_inference
>>> run_sen2_inference(
...     sen2_path="/data/S2A_MSIL2A_EXAMPLE.SAFE",
...     config_path="configs/config_20m.yaml",
...     ckpt_path="checkpoints/srgan-20m-6band/last.ckpt",
...     gpus=[0]
... )
"""
import os
from pathlib import Path

import torch

from .model.SRGAN import SRGAN_model


def load_model(config_path=None, ckpt_path=None, device=None):
    """Instantiate an SRGAN model and optionally load pretrained weights.

    This helper is safe to call from tests or scripts. It builds the model from
    a provided configuration file, loads weights from a checkpoint if available,
    and transfers the model to the selected device.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML configuration file describing the generator/discriminator setup.
    ckpt_path : str or Path, optional
        Path to a Lightning checkpoint or raw PyTorch state dictionary.
    device : str or torch.device, optional
        Device on which to place the model. Defaults to `"cuda"` if available.

    Returns
    -------
    model : SRGAN_model
        The loaded and ready-to-infer SRGAN Lightning module.
    device : str
        The device string used for model placement (e.g., `"cuda"` or `"cpu"`).

    Notes
    -----
    - Automatically switches the model to evaluation mode.
    - Tries to use the Lightning checkpoint API first; falls back to a raw `state_dict`
      for compatibility with manually saved checkpoints.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SRGAN_model(config_file_path=config_path).eval().to(device)

    if ckpt_path:
        # Try Lightning API first (without 'strict'); fall back to raw state_dict
        try:
            model = (
                SRGAN_model.load_from_checkpoint(ckpt_path, map_location=device)
                .eval()
                .to(device)
            )
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
            state = state.get("state_dict", state)
            model.load_state_dict(state, strict=False)

    return model, device


def run_sen2_inference(
    sen2_path=None,
    config_path=None,
    ckpt_path=None,
    gpus=None,
    window_size=(128, 128),
    factor=4,
    overlap=12,
    eliminate_border_px=2,
    save_preview=False,
    debug=False,
):
    """Run super-resolution inference on a Sentinel-2 SAFE product.

    This function prepares the SRGAN model, launches patch-based processing of
    large Sentinel-2 tiles, and orchestrates the full inference pipeline using
    the `opensr_utils.large_file_processing` backend.
    WARNING: only works for RGB-NIR as of now.

    Parameters
    ----------
    sen2_path : str or Path, optional
        Path to the Sentinel-2 `.SAFE` folder to process.
    config_path : str or Path, optional
        Path to the model configuration YAML file.
    ckpt_path : str or Path, optional
        Path to the model checkpoint file.
    gpus : list[int], optional
        GPU IDs to use for inference (e.g., `[0, 1]`). If `None`, automatically
        selects the first GPU if available.
    window_size : tuple(int, int), default=(128, 128)
        Size of each input patch (in low-resolution pixels) processed per pass.
    factor : int, default=4
        Super-resolution scaling factor.
    overlap : int, default=12
        Overlap (in LR pixels) between inference windows to reduce edge artifacts.
    eliminate_border_px : int, default=2
        Number of border pixels to remove from each patch during blending.
    save_preview : bool, default=False
        If True, saves a small visual preview of the reconstructed image.
    debug : bool, default=False
        Enables verbose debug logs for troubleshooting.

    Returns
    -------
    sr_object : opensr_utils.large_file_processing
        The inference handler object that manages tiling, batching, and output writing.

    Notes
    -----
    - Automatically sets `CUDA_VISIBLE_DEVICES` when `gpus` is provided.
    - Uses the first GPU by default when running on CUDA-enabled systems.
    - The returned object can be further inspected or extended post-run.
    """
    if gpus is not None and len(gpus) > 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, gpus)))

    model, device = load_model(config_path=config_path, ckpt_path=ckpt_path)

    import opensr_utils

    sr_object = opensr_utils.large_file_processing(
        root=sen2_path,
        model=model,
        window_size=window_size,
        factor=factor,
        overlap=overlap,
        eliminate_border_px=eliminate_border_px,
        device=device,
        gpus=gpus if gpus is not None else ([0] if device == "cuda" else []),
        save_preview=save_preview,
        debug=debug,
    )
    sr_object.start_super_resolution()
    return sr_object


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------
def main():
    """Example standalone entry point for testing Sentinel-2 inference.

    Sets up example paths for a sample SAFE product, model config, and checkpoint,
    then runs the super-resolution pipeline on GPU 0.

    This function is primarily intended for smoke testing and manual debugging.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # ---- Define placeholders ----
    sen2_path = Path(__file__).resolve().parent / "data" / "S2A_MSIL2A_EXAMPLE.SAFE"
    config_path = Path(__file__).resolve().parent / "configs" / "config_20m.yaml"
    ckpt_path = "checkpoints/srgan-20m-6band/last.ckpt"
    gpus = [0]

    run_sen2_inference(
        sen2_path=str(sen2_path),
        config_path=str(config_path),
        ckpt_path=ckpt_path,
        gpus=gpus,
    )


if __name__ == "__main__":
    main()
