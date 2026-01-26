"""Raster IO helpers for NDVI analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio


RasterData = Tuple[np.ndarray, dict]


def read_raster_band(path: Path | str) -> RasterData:
    """Read a single-band raster into memory and return data + profile."""
    with rasterio.open(path) as dataset:
        data = dataset.read(1)
        profile = dataset.profile.copy()
    return data, profile


def write_ndvi_geotiff(path: Path | str, ndvi: np.ndarray, profile: dict) -> None:
    """Write NDVI array to GeoTIFF using the provided profile."""
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
    )
    with rasterio.open(path, "w", **output_profile) as dataset:
        dataset.write(ndvi.astype(np.float32), 1)
