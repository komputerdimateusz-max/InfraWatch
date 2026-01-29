"""Reusable GeoJSON helpers for line-based scoring."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from pyproj import CRS
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from infrawatch.utils.crs import normalize_crs, to_crs_transformer

LINE_GEOMETRY_TYPES = ("LineString", "MultiLineString")
DEFAULT_LINE_CRS = CRS.from_epsg(4326)


def iter_coordinates(coordinates: Any) -> Iterable[Sequence[float]]:
    if not isinstance(coordinates, (list, tuple)):
        return
    if coordinates and isinstance(coordinates[0], (int, float)):
        yield coordinates
    else:
        for item in coordinates:
            yield from iter_coordinates(item)


def normalize_geojson_coordinates(geometry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(geometry)
    if "coordinates" in geometry:
        normalized["coordinates"] = _coordinates_to_lists(geometry["coordinates"])
    return normalized


def _coordinates_to_lists(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_coordinates_to_lists(item) for item in value]
    return value


def iter_line_features(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        features = [{"type": "Feature", "geometry": payload, "properties": {}}]

    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") in LINE_GEOMETRY_TYPES:
            yield feature


def detect_geojson_crs(payload: dict[str, Any], fallback: CRS | None = None) -> CRS:
    crs_payload = payload.get("crs")
    if isinstance(crs_payload, dict):
        props = crs_payload.get("properties") or {}
        name = props.get("name")
        if name:
            detected = normalize_crs(name)
            if detected:
                return detected
    return fallback or DEFAULT_LINE_CRS


def transform_feature_collection(
    payload: dict[str, Any],
    source_crs: Any,
    target_crs: Any,
) -> dict[str, Any]:
    if normalize_crs(source_crs) == normalize_crs(target_crs):
        return payload
    transformer = to_crs_transformer(source_crs, target_crs)
    if transformer is None:
        return payload

    transformed_features = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        projected = shapely_transform(transformer.transform, geom)
        transformed_feature = dict(feature)
        transformed_feature["geometry"] = mapping(projected)
        transformed_features.append(transformed_feature)
    return {"type": "FeatureCollection", "features": transformed_features}


def validate_line_feature_collection(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    valid_features: list[dict[str, Any]] = []

    def is_finite_number(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return np.isfinite(value)
        return False

    for idx, feature in enumerate(payload.get("features", []), start=1):
        geometry = feature.get("geometry")
        if not geometry or not isinstance(geometry, dict):
            warnings.append(f"Feature {idx} missing geometry.")
            continue
        geometry_type = geometry.get("type")
        if geometry_type not in LINE_GEOMETRY_TYPES:
            warnings.append(f"Feature {idx} skipped (unsupported geometry type).")
            continue
        coords = geometry.get("coordinates")
        if coords is None:
            warnings.append(f"Feature {idx} missing coordinates.")
            continue
        has_coords = False
        invalid_coord = False
        for coord in iter_coordinates(coords):
            if coord and len(coord) >= 2:
                has_coords = True
                if not (is_finite_number(coord[0]) and is_finite_number(coord[1])):
                    invalid_coord = True
                    break
        if not has_coords:
            warnings.append(f"Feature {idx} has empty coordinates.")
            continue
        if invalid_coord:
            warnings.append(f"Feature {idx} has invalid coordinate values.")
            continue
        normalized = dict(feature)
        normalized["geometry"] = normalize_geojson_coordinates(geometry)
        valid_features.append(normalized)

    return {"type": "FeatureCollection", "features": valid_features}, warnings


def bounds_from_feature_collection(payload: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bounds: tuple[float, float, float, float] | None = None
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        if bounds is None:
            bounds = geom.bounds
        else:
            minx, miny, maxx, maxy = bounds
            geom_bounds = geom.bounds
            bounds = (
                min(minx, geom_bounds[0]),
                min(miny, geom_bounds[1]),
                max(maxx, geom_bounds[2]),
                max(maxy, geom_bounds[3]),
            )
    return bounds
