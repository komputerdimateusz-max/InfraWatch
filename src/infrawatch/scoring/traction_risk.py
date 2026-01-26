"""Traction corridor risk scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape


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


def score_traction_segments(
    ndvi_path: Path | str,
    lines_path: Path | str,
    *,
    buffer_m: float = 20.0,
) -> list[dict[str, Any]]:
    """Score traction risk for each LineString feature in a GeoJSON."""
    ndvi_path = Path(ndvi_path)
    lines_path = Path(lines_path)

    lines_payload = _load_geojson(lines_path)
    features = _iter_line_features(lines_payload)

    results: list[dict[str, Any]] = []

    with rasterio.open(ndvi_path) as dataset:
        ndvi = dataset.read(1).astype(np.float32)
        transform = dataset.transform

        for idx, feature in enumerate(features, start=1):
            geom = shape(feature.get("geometry"))
            buffered = geom.buffer(buffer_m)
            mask = rasterize(
                [(buffered, 1)],
                out_shape=ndvi.shape,
                transform=transform,
                fill=0,
                dtype="uint8",
            )
            values = ndvi[mask == 1]
            values = values[~np.isnan(values)]
            if values.size == 0:
                mean_ndvi = float("nan")
                p90_ndvi = float("nan")
                pct_above = float("nan")
            else:
                mean_ndvi = float(np.mean(values))
                p90_ndvi = float(np.percentile(values, 90))
                pct_above = float(np.mean(values > 0.6))

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
                }
            )

    return results
