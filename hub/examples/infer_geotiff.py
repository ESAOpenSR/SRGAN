# SPDX-License-Identifier: MIT
"""Super-resolve one small, already co-registered GeoTIFF cutout in memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine

from opensr_srgan import load_inference_model


_BAND_ORDERS = {
    "SEN2-SR": ("B04", "B03", "B02", "B08"),
    "RGB-NIR": ("B04", "B03", "B02", "B08"),
    "SWIR": ("B05", "B06", "B07", "B8A", "B11", "B12"),
}


def _parse_band_indices(value: str | None, *, count: int) -> tuple[int, ...]:
    if value is None:
        return tuple(range(1, count + 1))
    try:
        indices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--bands must be comma-separated positive integers.") from error
    if len(indices) != count or any(index < 1 for index in indices):
        raise argparse.ArgumentTypeError(f"--bands must contain exactly {count} positive indices.")
    return indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("SEN2-SR", "RGB-NIR", "SWIR"), default="SEN2-SR")
    parser.add_argument("--input", required=True, type=Path, help="Small input GeoTIFF.")
    parser.add_argument("--output", required=True, type=Path, help="Output GeoTIFF.")
    parser.add_argument(
        "--bands",
        help="One-based input indices in model order, for example 1,2,3,4. Defaults to 1..C.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument("--revision", help="Optional Hugging Face commit, tag, or branch to resolve.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    model = load_inference_model(
        args.preset, map_location=device, revision=args.revision
    ).to(device)
    model.eval()
    channels = int(model.config.Model.in_bands)
    scale = int(model.config.Generator.scaling_factor)
    indices = _parse_band_indices(args.bands, count=channels)
    band_names = _BAND_ORDERS[args.preset]

    with rasterio.open(args.input) as source:
        if max(indices) > source.count:
            raise ValueError(f"{args.input} has {source.count} bands but --bands requests {indices}.")
        # This intentionally reads the full cutout. Use a tiled pipeline for large scenes.
        lr = source.read(indexes=indices).astype(np.float32, copy=False)
        profile = source.profile.copy()
        source_transform = source.transform

    if not np.isfinite(lr).all():
        raise ValueError("Input GeoTIFF contains NaN or infinite values in the selected bands.")
    normalized_input = bool(np.isfinite(lr).any() and np.nanmin(lr) >= 0 and np.nanmax(lr) <= 1.5)

    with torch.inference_mode():
        sr = model.predict_step(torch.from_numpy(lr).unsqueeze(0))[0]
    sr_data = sr.numpy().astype(np.float32, copy=False)
    height, width = sr_data.shape[-2:]
    transform = source_transform * Affine.scale(lr.shape[-1] / width, lr.shape[-2] / height)

    profile.update(
        driver="GTiff",
        count=sr_data.shape[0],
        height=height,
        width=width,
        dtype="float32",
        transform=transform,
        compress="deflate",
        predictor=3,
    )
    for key in ("blockxsize", "blockysize", "tiled", "nodata"):
        profile.pop(key, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output, "w", **profile) as destination:
        destination.write(sr_data)
        for index, name in enumerate(band_names, start=1):
            destination.set_band_description(index, name)
        destination.update_tags(
            preset=args.preset,
            input_band_indices=",".join(map(str, indices)),
            band_order=",".join(band_names),
            sr_scale_factor=str(scale),
            value_convention=(
                "normalized reflectance from predict_step"
                if normalized_input
                else "raw DN/reflectance from predict_step"
            ),
        )
    print(
        f"Saved {args.output}: {lr.shape[-2:]} -> {sr_data.shape[-2:]}, "
        f"scale={scale}, source bands {indices} as {','.join(band_names)}."
    )


if __name__ == "__main__":
    main()
