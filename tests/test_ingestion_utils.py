from datetime import datetime, timezone
from pathlib import Path
import pytest

from infrawatch.ingestion.models import SceneSummary
from infrawatch.ingestion.paths import (
    normalize_date_folder,
    sanitize_product_id,
    scene_dir,
    validate_bbox,
)


def test_validate_bbox_happy_path():
    bbox = validate_bbox((14.0, 52.0, 15.5, 53.1))
    assert bbox == (14.0, 52.0, 15.5, 53.1)


def test_validate_bbox_invalid_order():
    with pytest.raises(ValueError):
        validate_bbox((15.0, 52.0, 14.0, 53.1))


def test_normalize_date_folder_utc():
    dt = datetime(2025, 6, 1, 12, 30, tzinfo=timezone.utc)
    assert normalize_date_folder(dt) == "20250601"


def test_sanitize_product_id():
    assert sanitize_product_id("S2A/ABC") == "S2A_ABC"


def test_scene_dir_uses_sanitized_product_id(tmp_path: Path):
    dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    output = scene_dir(tmp_path, dt, "S2A/ABC")
    assert output == tmp_path / "raw" / "s2" / "20250601" / "S2A_ABC"


def test_scene_sorting_by_datetime_and_id():
    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    scenes = [
        SceneSummary(
            product_id="B",
            title="B",
            acquisition_datetime=base,
            cloud_cover=None,
            bbox=(0.0, 0.0, 1.0, 1.0),
            footprint=None,
            download_url="https://example.com/b",
        ),
        SceneSummary(
            product_id="A",
            title="A",
            acquisition_datetime=base,
            cloud_cover=None,
            bbox=(0.0, 0.0, 1.0, 1.0),
            footprint=None,
            download_url="https://example.com/a",
        ),
        SceneSummary(
            product_id="C",
            title="C",
            acquisition_datetime=base.replace(day=2),
            cloud_cover=None,
            bbox=(0.0, 0.0, 1.0, 1.0),
            footprint=None,
            download_url="https://example.com/c",
        ),
    ]
    sorted_scenes = sorted(scenes, key=lambda scene: (scene.acquisition_datetime, scene.product_id))
    assert [scene.product_id for scene in sorted_scenes] == ["A", "B", "C"]
