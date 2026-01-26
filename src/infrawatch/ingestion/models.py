from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class IngestionRequest:
    bbox: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    date_from: date
    date_to: date
    max_scenes: int = 3
    max_cloud_cover: float | None = None


@dataclass(frozen=True)
class SceneSummary:
    product_id: str
    title: str
    acquisition_datetime: datetime
    cloud_cover: float | None
    bbox: tuple[float, float, float, float]
    footprint: Any | None
    assets: dict[str, str]  # e.g. {"B04": "s3://eodata/...", "B08": "...", "SCL": "..."}


@dataclass(frozen=True)
class SceneMetadata:
    product_id: str
    title: str
    acquisition_datetime: datetime
    cloud_cover: float | None
    bbox: tuple[float, float, float, float]
    footprint: Any | None
    source_endpoint: str
    source_query: dict[str, Any]
    downloaded_assets: dict[str, str] | None = None  # band -> local path (optional, set by CLI)
