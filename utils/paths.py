from __future__ import annotations

from pathlib import Path

from .models import SceneSummary


def as_bbox_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def select_asset_filename(href: str, fallback: str) -> str:
    # Prefer filename from href if present
    tail = href.rsplit("/", 1)[-1]
    if tail and "." in tail:
        return tail
    return fallback


def scene_cache_dir(base_dir: Path, scene: SceneSummary) -> Path:
    ymd = scene.acquisition_datetime.strftime("%Y%m%d")
    return base_dir / ymd / scene.product_id
