"""Sentinel-2 Scene Classification Layer utilities."""

from __future__ import annotations

import numpy as np


EXCLUDED_CLASSES = {3, 8, 9, 10, 11}
WATER_CLASS = 6


def scl_valid_mask(scl: np.ndarray, *, exclude_water: bool = True) -> np.ndarray:
    """Return a boolean mask of valid pixels given an SCL raster.

    Excludes cloud shadow (3), clouds (8, 9, 10), cirrus (10), snow (11), and
    optionally water (6).
    """
    excluded = set(EXCLUDED_CLASSES)
    if exclude_water:
        excluded.add(WATER_CLASS)
    return ~np.isin(scl, list(excluded))
