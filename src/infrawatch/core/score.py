"""Shared risk scoring logic for traction corridors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.errors import WindowError
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from infrawatch.core.geo import iter_line_features
from infrawatch.utils.crs import normalize_crs, to_crs_transformer

NDVI_THRESHOLD_DEFAULT = 0.6
RISK_LOW_MAX = 34.0
RISK_MED_MAX = 67.0
RISK_SCORE_NO_DATA = 100.0
MEAN_WEIGHT = 0.5
P90_WEIGHT = 0.3
PCT_ABOVE_WEIGHT = 0.2
P90_PERCENTILE = 90.0


def risk_category(score: float) -> str:
    if score < RISK_LOW_MAX:
        return "LOW"
    if score < RISK_MED_MAX:
        return "MED"
    return "HIGH"


def risk_score(mean_ndvi: float, p90_ndvi: float, pct_above_threshold: float) -> float:
    if np.isnan(mean_ndvi) or np.isnan(p90_ndvi) or np.isnan(pct_above_threshold):
        return RISK_SCORE_NO_DATA
    mean_norm = np.clip((mean_ndvi + 1.0) / 2.0, 0.0, 1.0)
    p90_norm = np.clip((p90_ndvi + 1.0) / 2.0, 0.0, 1.0)
    pct_norm = np.clip(pct_above_threshold, 0.0, 1.0)
    vegetation_score = (
        MEAN_WEIGHT * mean_norm + P90_WEIGHT * p90_norm + PCT_ABOVE_WEIGHT * pct_norm
    )
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


def _no_data_stats(reason: str | None = None) -> dict[str, Any]:
    return {
        "count": 0,
        "mean_ndvi": float("nan"),
        "p90_ndvi": float("nan"),
        "pct_above_threshold": float("nan"),
        "data_status": "NO_DATA",
        "data_status_detail": reason,
    }


def _bounds_overlap(
    bounds: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
) -> bool:
    return not (
        bounds[2] <= other[0]
        or bounds[0] >= other[2]
        or bounds[3] <= other[1]
        or bounds[1] >= other[3]
    )


def _safe_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    return (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))


def _safe_window_from_bounds(
    bounds: tuple[float, float, float, float],
    dataset: rasterio.io.DatasetReader,
) -> Window | None:
    try:
        window = from_bounds(*bounds, transform=dataset.transform)
        window = window.intersection(Window(0, 0, dataset.width, dataset.height))
        window = window.round_offsets().round_lengths()
    except WindowError:
        return None
    if window.width < 1 or window.height < 1:
        return None
    return window


def _sample_ndvi_for_line_dataset(
    dataset: rasterio.io.DatasetReader,
    line_geom,
    buffer_m: float,
    *,
    line_crs: Any = None,
    ndvi_threshold: float = NDVI_THRESHOLD_DEFAULT,
) -> dict[str, Any]:
    if line_geom is None or line_geom.is_empty:
        return _no_data_stats("Empty geometry")

    ndvi_crs = dataset.crs
    ndvi_geom = reproject_shapely_geom(line_geom, line_crs, ndvi_crs)
    buffered = ndvi_geom.buffer(buffer_m) if buffer_m > 0 else ndvi_geom
    if buffered.is_empty:
        return _no_data_stats("Empty geometry after buffering")

    bounds = _safe_bounds(buffered.bounds)
    if not _bounds_overlap(bounds, dataset.bounds):
        return _no_data_stats("Geometry outside NDVI extent")

    window = _safe_window_from_bounds(bounds, dataset)
    if window is None:
        return _no_data_stats("Geometry outside NDVI extent")

    data = dataset.read(1, window=window, masked=True).astype(np.float32)
    if data.size == 0:
        return _no_data_stats("No NDVI samples in window")
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
        return _no_data_stats("No NDVI samples in buffer")

    mean_ndvi = float(values.mean())
    p90_ndvi = float(np.percentile(values.compressed(), P90_PERCENTILE))
    pct_above = float(np.mean(values.compressed() > ndvi_threshold))
    return {
        "count": int(values.count()),
        "mean_ndvi": mean_ndvi,
        "p90_ndvi": p90_ndvi,
        "pct_above_threshold": pct_above,
        "data_status": "OK",
    }


def sample_ndvi_for_line(
    ndvi_path: Path | str,
    line_geom,
    buffer_m: float,
    step_m: float | None = None,
    *,
    line_crs: Any = None,
    ndvi_threshold: float = NDVI_THRESHOLD_DEFAULT,
) -> dict[str, Any]:
    """Sample NDVI values within a buffered line corridor using windowed reads."""
    _ = step_m
    ndvi_path = Path(ndvi_path)
    with rasterio.open(ndvi_path) as dataset:
        return _sample_ndvi_for_line_dataset(
            dataset,
            line_geom,
            buffer_m,
            line_crs=line_crs,
            ndvi_threshold=ndvi_threshold,
        )


def _load_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_traction_segments(
    ndvi_path: Path | str,
    lines_path: Path | str,
    *,
    buffer_m: float = 20.0,
    lines_crs: Any = None,
    ndvi_threshold: float = NDVI_THRESHOLD_DEFAULT,
) -> list[dict[str, Any]]:
    """Score traction risk for each LineString feature in a GeoJSON."""
    ndvi_path = Path(ndvi_path)
    lines_path = Path(lines_path)

    lines_payload = _load_geojson(lines_path)
    features = list(iter_line_features(lines_payload))

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
                ndvi_threshold=ndvi_threshold,
            )
            mean_ndvi = stats["mean_ndvi"]
            p90_ndvi = stats["p90_ndvi"]
            pct_above = stats["pct_above_threshold"]

            score = risk_score(mean_ndvi, p90_ndvi, pct_above)
            category = risk_category(score)
            results.append(
                {
                    "segment_id": idx,
                    "feature_id": feature.get("id"),
                    "mean_ndvi": mean_ndvi,
                    "p90_ndvi": p90_ndvi,
                    "pct_above_0_6": pct_above,
                    "risk_score": score,
                    "risk_category": category,
                    "buffer_m": buffer_m,
                    "data_status": stats["data_status"],
                    "sample_count": stats["count"],
                }
            )

    return results


@dataclass
class ScoreMeta:
    ndvi_path: str | None
    ndvi_crs: str | None
    line_count: int


def score_feature_collection(
    feature_collection: dict[str, Any],
    ndvi_path: Path | str | None,
    *,
    buffer_m: float,
    ndvi_threshold: float,
) -> tuple[list[dict[str, Any]], ScoreMeta]:
    """Score a GeoJSON FeatureCollection using an NDVI raster (optional)."""
    features = list(iter_line_features(feature_collection))
    line_count = len(features)
    if line_count == 0:
        return [], ScoreMeta(ndvi_path=None, ndvi_crs=None, line_count=0)

    lines_crs = normalize_crs(_detect_geojson_crs(feature_collection)) or CRS.from_epsg(4326)

    if ndvi_path is None:
        results = []
        for idx in range(1, line_count + 1):
            stats = _no_data_stats("NDVI missing")
            score = risk_score(stats["mean_ndvi"], stats["p90_ndvi"], stats["pct_above_threshold"])
            results.append(
                {
                    "segment_id": idx,
                    "feature_id": features[idx - 1].get("id"),
                    "mean_ndvi": stats["mean_ndvi"],
                    "p90_ndvi": stats["p90_ndvi"],
                    "pct_above_0_6": stats["pct_above_threshold"],
                    "risk_score": score,
                    "risk_category": risk_category(score),
                    "buffer_m": buffer_m,
                    "data_status": stats["data_status"],
                    "sample_count": stats["count"],
                }
            )
        return results, ScoreMeta(ndvi_path=None, ndvi_crs=None, line_count=line_count)

    ndvi_path = Path(ndvi_path)
    with rasterio.open(ndvi_path) as dataset:
        ndvi_crs = dataset.crs.to_string() if dataset.crs else None
        results: list[dict[str, Any]] = []
        for idx, feature in enumerate(features, start=1):
            geom = shape(feature.get("geometry"))
            stats = _sample_ndvi_for_line_dataset(
                dataset,
                geom,
                buffer_m,
                line_crs=lines_crs,
                ndvi_threshold=ndvi_threshold,
            )
            mean_ndvi = stats["mean_ndvi"]
            p90_ndvi = stats["p90_ndvi"]
            pct_above = stats["pct_above_threshold"]
            score = risk_score(mean_ndvi, p90_ndvi, pct_above)
            results.append(
                {
                    "segment_id": idx,
                    "feature_id": feature.get("id"),
                    "mean_ndvi": mean_ndvi,
                    "p90_ndvi": p90_ndvi,
                    "pct_above_0_6": pct_above,
                    "risk_score": score,
                    "risk_category": risk_category(score),
                    "buffer_m": buffer_m,
                    "data_status": stats["data_status"],
                    "sample_count": stats["count"],
                }
            )

    return results, ScoreMeta(
        ndvi_path=str(ndvi_path),
        ndvi_crs=ndvi_crs,
        line_count=line_count,
    )


def build_trend_rows(
    feature_collection: dict[str, Any],
    ndvi_paths_by_date: dict[str, Path | None],
    *,
    buffer_m: float,
    ndvi_threshold: float,
) -> list[dict[str, Any]]:
    """Score the same segments across dates, returning a trend table."""
    trend_rows: list[dict[str, Any]] = []
    for date_label, ndvi_path in ndvi_paths_by_date.items():
        results, _ = score_feature_collection(
            feature_collection,
            ndvi_path,
            buffer_m=buffer_m,
            ndvi_threshold=ndvi_threshold,
        )
        for row in results:
            trend_rows.append(
                {
                    "date": date_label,
                    "segment_id": row["segment_id"],
                    "mean_ndvi": row["mean_ndvi"],
                    "risk_score": row["risk_score"],
                    "risk_category": row["risk_category"],
                    "data_status": row["data_status"],
                }
            )
    return trend_rows
