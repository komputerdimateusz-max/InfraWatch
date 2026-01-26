"""CRS helpers for consistent axis ordering and safe transforms."""

from __future__ import annotations

from typing import Any

from pyproj import CRS, Transformer


def normalize_crs(value: Any) -> CRS | None:
    """Normalize CRS input to a pyproj CRS, or None if unavailable."""
    if value is None:
        return None
    return CRS.from_user_input(value)


def to_crs_transformer(source: Any, target: Any) -> Transformer | None:
    """Build a Transformer with always_xy for consistent lon/lat ordering."""
    source_crs = normalize_crs(source)
    target_crs = normalize_crs(target)
    if source_crs is None or target_crs is None:
        return None
    if source_crs == target_crs:
        return None
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def transform_bounds_always_xy(
    source: Any,
    target: Any,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Transform bounds using explicit x/y axis order for reliable lon/lat results."""
    source_crs = normalize_crs(source)
    target_crs = normalize_crs(target)
    if source_crs is None or target_crs is None:
        raise ValueError("CRS required to transform bounds.")
    if source_crs == target_crs:
        return bounds

    left, bottom, right, top = bounds
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    minx, miny = transformer.transform(left, bottom)
    maxx, maxy = transformer.transform(right, top)
    return (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))
