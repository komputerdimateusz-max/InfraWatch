"""Traction corridor risk scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.errors import WindowError
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from infrawatch.utils.crs import normalize_crs, to_crs_transformer

def _load_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_line_features(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("type") == "FeatureCollection":
        return list(payload.get("features", []))
    if payload.get("type") == "Feature":
        return [payload]
    return [{"type": "Feature", "geometry": payload, "properties": {}}]


def _risk_category(score: float) -> str:
    if score < 34:
        return "LOW"
    if score < 67:
        return "MED"
    return "HIGH"


def _risk_score(mean_ndvi: float, p90_ndvi: float, pct_above_0_6: float) -> float:
    if np.isnan(mean_ndvi) or np.isnan(p90_ndvi) or np.isnan(pct_above_0_6):
        return 100.0
    mean_norm = np.clip((mean_ndvi + 1.0) / 2.0, 0.0, 1.0)
    p90_norm = np.clip((p90_ndvi + 1.0) / 2.0, 0.0, 1.0)
    pct_norm = np.clip(pct_above_0_6, 0.0, 1.0)
    vegetation_score = 0.5 * mean_norm + 0.3 * p90_norm + 0.2 * pct_norm
    return float(np.clip((1.0 - vegetation_score) * 100.0, 0.0, 100.0))


def _detect_geojson_crs(payload: dict[str, Any]) -> CRS | None:
    crs_payload = payload.get("crs")
    if isinstance(crs_payload, dict):
        props = crs_payload.get("properties") or {}
        name = props.get("name")
        if name:
            return normalize_crs(name)
    return None


def reproject_shapely_geom(geom, src_crs: Any, dst_crs: Any):
    transformer = to_crs_transformer(src_crs, dst_crs)
    if transformer is None:
        return geom
    return shapely_transform(transformer.transform, geom)


def _no_data_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "mean_ndvi": float("nan"),
        "p90_ndvi": float("nan"),
        "pct_above_0_6": float("nan"),
        "data_status": "NO_DATA",
    }


def _sample_ndvi_for_line_dataset(
    dataset: rasterio.io.DatasetReader,
    line_geom,
    buffer_m: float,
    *,
    line_crs: Any = None,
) -> dict[str, Any]:
    if line_geom is None or line_geom.is_empty:
        return _no_data_stats()

    ndvi_crs = dataset.crs
    ndvi_geom = reproject_shapely_geom(line_geom, line_crs, ndvi_crs)
    buffered = ndvi_geom.buffer(buffer_m) if buffer_m > 0 else ndvi_geom
    if buffered.is_empty:
        return _no_data_stats()

    bounds = buffered.bounds
    try:
        window = from_bounds(*bounds, transform=dataset.transform)
        window = window.intersection(Window(0, 0, dataset.width, dataset.height))
        # Round to pixel grid so raster reads and masks align exactly.
        window = window.round_offsets().round_lengths()
    except WindowError:
        return _no_data_stats()

    if window.width < 1 or window.height < 1:
        return _no_data_stats()

    data = dataset.read(1, window=window, masked=True).astype(np.float32)
    if data.size == 0:
        return _no_data_stats()
    mask = rasterize(
        [(buffered, 1)],
        out_shape=data.shape,
        transform=dataset.window_transform(window),
        fill=0,
        dtype="uint8",
    ).astype(bool)

    if data.mask is np.ma.nomask:
        combined_mask = ~mask
    else:
        combined_mask = np.logical_or(data.mask, ~mask)
    values = np.ma.array(data, mask=combined_mask)
    values = np.ma.masked_invalid(values)

    if values.count() == 0:
        return _no_data_stats()

    mean_ndvi = float(values.mean())
    p90_ndvi = float(np.percentile(values.compressed(), 90))
    pct_above = float(np.mean(values.compressed() > 0.6))
    return {
        "count": int(values.count()),
        "mean_ndvi": mean_ndvi,
        "p90_ndvi": p90_ndvi,
        "pct_above_0_6": pct_above,
        "data_status": "OK",
    }


def sample_ndvi_for_line(
    ndvi_path: Path | str,
    line_geom,
    buffer_m: float,
    step_m: float | None = None,
    *,
    line_crs: Any = None,
) -> dict[str, Any]:
    """Sample NDVI values within a buffered line corridor using windowed reads."""
    _ = step_m
    ndvi_path = Path(ndvi_path)
    with rasterio.open(ndvi_path) as dataset:
        return _sample_ndvi_for_line_dataset(dataset, line_geom, buffer_m, line_crs=line_crs)


def score_traction_segments(
    ndvi_path: Path | str,
    lines_path: Path | str,
    *,
    buffer_m: float = 20.0,
    lines_crs: Any = None,
) -> list[dict[str, Any]]:
    """Score traction risk for each LineString feature in a GeoJSON."""
    ndvi_path = Path(ndvi_path)
    lines_path = Path(lines_path)

    lines_payload = _load_geojson(lines_path)
    features = _iter_line_features(lines_payload)

    results: list[dict[str, Any]] = []

    resolved_lines_crs = normalize_crs(lines_crs) or _detect_geojson_crs(lines_payload)

    with rasterio.open(ndvi_path) as dataset:
        for idx, feature in enumerate(features, start=1):
            geom = shape(feature.get("geometry"))
            stats = _sample_ndvi_for_line_dataset(
                dataset,
                geom,
                buffer_m,
                line_crs=resolved_lines_crs or dataset.crs,
            )
            mean_ndvi = stats["mean_ndvi"]
            p90_ndvi = stats["p90_ndvi"]
            pct_above = stats["pct_above_0_6"]

            risk_score = _risk_score(mean_ndvi, p90_ndvi, pct_above)
            risk_category = _risk_category(risk_score)
            results.append(
                {
                    "segment_id": idx,
                    "feature_id": feature.get("id"),
                    "mean_ndvi": mean_ndvi,
                    "p90_ndvi": p90_ndvi,
                    "pct_above_0_6": pct_above,
                    "risk_score": risk_score,
                    "risk_category": risk_category,
                    "buffer_m": buffer_m,
                    "data_status": stats["data_status"],
                    "sample_count": stats["count"],
                }
            )

    return results
