from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Iterable, Sequence

RAW_S2_SUBDIR = Path("raw") / "s2"
PRODUCT_ID_ALLOWED = re.compile(r"[^A-Za-z0-9._-]+")


def validate_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("BBox must have four values: min_lon min_lat max_lon max_lat.")
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("BBox coordinates must satisfy min < max for lon/lat.")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be within [-180, 180].")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be within [-90, 90].")
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def normalize_date_folder(value: date | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc).date()
    return value.strftime("%Y%m%d")


def sanitize_product_id(product_id: str) -> str:
    cleaned = PRODUCT_ID_ALLOWED.sub("_", product_id.strip())
    return cleaned or "unknown_product"


def raw_s2_root(data_dir: Path) -> Path:
    return data_dir / RAW_S2_SUBDIR


def scene_date_dir(data_dir: Path, acquisition_dt: datetime) -> Path:
    return raw_s2_root(data_dir) / normalize_date_folder(acquisition_dt)


def scene_dir(data_dir: Path, acquisition_dt: datetime, product_id: str) -> Path:
    safe_id = sanitize_product_id(product_id)
    return scene_date_dir(data_dir, acquisition_dt) / safe_id


def metadata_path(scene_dir_path: Path) -> Path:
    return scene_dir_path / "metadata.json"


def select_asset_filename(url: str, fallback: str) -> str:
    filename = Path(url).name
    return filename or fallback


def as_bbox_list(bbox: Iterable[float]) -> list[float]:
    return [float(value) for value in bbox]
