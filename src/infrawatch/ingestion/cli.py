from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Sequence

from infrawatch.config import load_config

from .copernicus_client import CopernicusAuth, CopernicusClient, scene_metadata
from .models import IngestionRequest, SceneSummary
from .paths import metadata_path, raw_s2_root, scene_dir, validate_bbox

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InfraWatch Sentinel-2 ingestion (Copernicus).")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box coordinates in lon/lat.",
    )
    parser.add_argument("--date-from", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--date-to", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--max-scenes", type=int, required=True, help="Maximum scenes to fetch.")
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=None,
        help="Optional max cloud cover percentage (0-100).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List downloads without fetching.")
    return parser.parse_args(argv)


def build_request(args: argparse.Namespace) -> IngestionRequest:
    bbox = validate_bbox(args.bbox)
    return IngestionRequest(
        bbox=bbox,
        dateFrom=args.date_from,
        dateTo=args.date_to,
        maxScenes=args.max_scenes,
        maxCloudCover=args.max_cloud_cover,
        dryRun=args.dry_run,
    )


def print_plan(scenes: list[SceneSummary], data_root: Path) -> None:
    print("Planned Sentinel-2 scenes:")
    for idx, scene in enumerate(scenes, start=1):
        out_dir = scene_dir(data_root, scene.acquisition_datetime, scene.product_id)
        print(f"{idx:02d}. {scene.acquisition_datetime.isoformat()} | {scene.product_id}")
        print(f"    -> {out_dir}")


def write_metadata(metadata_path_value: Path, metadata: dict) -> None:
    metadata_path_value.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def get_auth_from_env() -> CopernicusAuth | None:
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    if username and password:
        return CopernicusAuth(username=username, password=password)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    logging.basicConfig(level=cfg.log_level)

    request = build_request(args)
    auth = get_auth_from_env()
    client = CopernicusClient(auth=auth)

    scenes, query = client.search(request)
    data_root = raw_s2_root(cfg.data_dir)
    print_plan(scenes, cfg.data_dir)

    if request.dry_run:
        if not auth:
            print(
                "Dry-run completed. To enable downloads, set COPERNICUS_USERNAME and "
                "COPERNICUS_PASSWORD in your environment."
            )
        return 0

    if not auth:
        raise RuntimeError(
            "COPERNICUS_USERNAME and COPERNICUS_PASSWORD are required for downloads. "
            "Run with --dry-run to preview available scenes."
        )

    logger.info("Saving data under %s", data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        scene_path = scene_dir(cfg.data_dir, scene.acquisition_datetime, scene.product_id)
        meta_path = metadata_path(scene_path)
        if meta_path.exists():
            logger.info("Skipping cached scene %s", scene.product_id)
            continue
        asset_path = client.download_scene(scene, scene_path)
        meta = scene_metadata(scene, client.endpoint, query).model_dump(mode="json")
        meta["local_asset"] = str(asset_path)
        write_metadata(meta_path, meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
