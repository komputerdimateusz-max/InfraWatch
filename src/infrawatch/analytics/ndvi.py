"""NDVI math utilities."""

from __future__ import annotations

import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI from red and near-infrared arrays.

    NDVI = (NIR - RED) / (NIR + RED)
    Where the denominator is zero, the output is NaN.
    """
    red_array = np.asarray(red, dtype=np.float32)
    nir_array = np.asarray(nir, dtype=np.float32)
    denominator = nir_array + red_array
    numerator = nir_array - red_array
    ndvi = np.full_like(denominator, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=ndvi, where=denominator != 0)
    return ndvi
