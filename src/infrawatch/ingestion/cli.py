from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from infrawatch.config import load_config
from infrawatch.logging_conf import setup_logging

from .copernicus_client import CopernicusAuth, CopernicusClient, scene_metadata
from .models import IngestionRequest
from .paths import scene_cache_dir

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(description="InfraWatch Sentinel-2 ingestion (MVP).")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--max-cloud-cover", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    cfg = load_config()
    setup_logging(cfg.log_level)

    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)

    request = IngestionRequest(
        bbox=(args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]),
        date_from=date_from,
        date_to=date_to,
        max_scenes=args.max_scenes,
        max_cloud_cover=args.max_cloud_cover,
    )

    auth = None
    import os

    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    if username and password:
        auth = CopernicusAuth(username=username, password=password)

    client = CopernicusClient(auth=auth)

    scenes, query = client.search(request)

    base_out = cfg.data_dir

    logger.info("Saving data under %s", base_out)

    print("Planned Sentinel-2 scenes:")
    for i, scene in enumerate(scenes, start=1):
        scene_dir = scene_cache_dir(base_out, scene)
        print(f"{i:02d}. {scene.acquisition_datetime.isoformat()} | {scene.product_id}")
        print(f"    -> {scene_dir}")

    if args.dry_run:
        print("Dry-run completed. To enable downloads, set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in your environment.")
        return 0

    if not client.has_credentials():
        raise RuntimeError("Missing COPERNICUS_USERNAME/COPERNICUS_PASSWORD. Downloads require credentials.")

    for scene in scenes:
        scene_dir = scene_cache_dir(base_out, scene)
        meta_path = scene_dir / "metadata.json"

        # Cache check: if metadata exists, assume scene already processed
        if meta_path.exists():
            logger.info("Skipping cached scene %s", scene.product_id)
            continue

        downloaded = client.download_scene(scene, scene_dir)
        downloaded_assets = {band: str(path) for band, path in downloaded.items()}

        meta = scene_metadata(scene, client.endpoint, query)
        meta_dict = {
            "product_id": meta.product_id,
            "title": meta.title,
            "acquisition_datetime": meta.acquisition_datetime.isoformat(),
            "cloud_cover": meta.cloud_cover,
            "bbox": meta.bbox,
            "footprint": meta.footprint,
            "source_endpoint": meta.source_endpoint,
            "source_query": meta.source_query,
            "downloaded_assets": downloaded_assets,
        }
        _write_json(meta_path, meta_dict)

        for band, path in downloaded.items():
            logger.info("Downloaded %s -> %s", band, path)

    return 0
