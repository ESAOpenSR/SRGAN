from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from deployment.srgan_hpc.raster import compress_geotiff, stack_geotiffs


def _write_tif(path: Path, data: np.ndarray, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:32632",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data)


def test_stack_geotiffs_reprojects_to_reference_grid_and_writes_band_names(tmp_path: Path) -> None:
    reference = tmp_path / "rgbnir.tif"
    secondary = tmp_path / "swir.tif"
    output = tmp_path / "fused.tif"
    ref_data = np.ones((4, 8, 8), dtype="uint16")
    secondary_data = np.full((6, 4, 4), 2, dtype="uint16")

    _write_tif(reference, ref_data, from_origin(500000, 5100000, 2.5, 2.5))
    _write_tif(secondary, secondary_data, from_origin(500000, 5100000, 5.0, 5.0))

    stack_geotiffs(
        reference_path=reference,
        secondary_path=secondary,
        output_path=output,
        band_names=["B04", "B03", "B02", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"],
    )

    with rasterio.open(output) as src:
        assert src.count == 10
        assert src.width == 8
        assert src.height == 8
        assert src.transform == from_origin(500000, 5100000, 2.5, 2.5)
        assert src.descriptions == ("B04", "B03", "B02", "B08", "B05", "B06", "B07", "B8A", "B11", "B12")


def test_compress_geotiff_writes_band_names(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    output = tmp_path / "rgbnir_sr.tif"
    _write_tif(source, np.ones((4, 8, 8), dtype="uint16"), from_origin(500000, 5100000, 2.5, 2.5))

    compress_geotiff(source, output, band_names=["B04", "B03", "B02", "B08"])

    with rasterio.open(output) as src:
        assert src.descriptions == ("B04", "B03", "B02", "B08")
        assert src.tags()["band_names"] == "B04,B03,B02,B08"
