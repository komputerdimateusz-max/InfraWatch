"""Raster resampling utilities."""

from __future__ import annotations

import numpy as np
from rasterio.warp import Resampling, reproject


def resample_to_match(
    src_arr: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
    method: str = "nearest",
) -> np.ndarray:
    """Resample src_arr to match destination profile."""
    resampling = Resampling[method]
    dst_arr = np.empty((dst_profile["height"], dst_profile["width"]), dtype=src_arr.dtype)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_profile.get("transform"),
        src_crs=src_profile.get("crs"),
        dst_transform=dst_profile.get("transform"),
        dst_crs=dst_profile.get("crs"),
        resampling=resampling,
    )
    return dst_arr
