from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from dotenv import load_dotenv

from infrawatch.analytics.ndvi import compute_ndvi
from infrawatch.analytics.raster_io import read_raster_band, write_ndvi_geotiff
from infrawatch.analytics.resample import resample_to_match
from infrawatch.analytics.scl import EXCLUDED_CLASSES, WATER_CLASS, scl_valid_mask
from infrawatch.config import load_config
from infrawatch.logging_conf import setup_logging

from .copernicus_client import (
    CopernicusAuth,
    CopernicusClient,
    DOWNLOAD_RETRIES,
    ODATA_BASE,
    STAC_ENDPOINT,
    build_search_payload,
    parse_datetime,
    post_json,
)
from .models import IngestionRequest, SceneSummary
from .paths import metadata_path, normalize_date_folder, scene_dir, select_asset_filename, validate_bbox


logger = logging.getLogger(__name__)

DEFAULT_MAX_DAYS = 10
DEFAULT_ASSETS = ("B04", "B08", "SCL")
DEFAULT_DATA_DIR = Path(os.getenv("EO_DATA_DIR", "C:/InfraWatch/satellite_data"))
SEARCH_LIMIT_FLOOR = 200
MISSING_CLOUD_COVER = 9999.0

ASSET_KEY_LOOKUP = {
    "B04": ["B04", "b04", "red", "B04_10m", "B04_20m"],
    "B08": ["B08", "b08", "nir", "B08_10m", "B08_20m"],
    "SCL": ["SCL", "scl", "SCL_20m", "SCL20", "classification"],
}


@dataclass(frozen=True)
class ScenePlan:
    scene: SceneSummary
    scene_dir: Path
    asset_hrefs: dict[str, str]
    asset_paths: dict[str, Path]


def parse_assets(raw: str) -> tuple[str, ...]:
    assets = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not assets:
        raise ValueError("Assets list cannot be empty.")
    unknown = [asset for asset in assets if asset not in ASSET_KEY_LOOKUP]
    if unknown:
        raise ValueError(f"Unsupported asset(s): {unknown}. Supported: {list(ASSET_KEY_LOOKUP.keys())}")
    required = {"B04", "B08", "SCL"}
    if not required.issubset(set(assets)):
        raise ValueError("NDVI requires assets B04, B08, and SCL to be downloaded.")
    return tuple(assets)


def select_assets(assets: Mapping[str, Any], requested_assets: Iterable[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for asset in requested_assets:
        keys = ASSET_KEY_LOOKUP.get(asset, [])
        href = None
        for key in keys:
            entry = assets.get(key)
            if entry and isinstance(entry, Mapping) and "href" in entry:
                href = str(entry["href"])
                break
        if not href:
            raise RuntimeError(
                f"Missing required asset {asset}. Available STAC asset keys: {list(assets.keys())}"
            )
        selected[asset] = href
    return selected


def scene_from_feature(feature: Mapping[str, Any], requested_assets: Iterable[str]) -> SceneSummary:
    props = feature.get("properties", {})
    dt_raw = props.get("datetime") or props.get("start_datetime")
    if not dt_raw:
        raise RuntimeError("Scene is missing acquisition datetime.")
    acquisition_dt = parse_datetime(dt_raw)

    product_id = props.get("productIdentifier") or feature.get("id") or "unknown"
    title = props.get("title") or product_id
    cloud_cover = props.get("eo:cloud_cover")

    bbox = feature.get("bbox") or []
    if len(bbox) != 4:
        raise RuntimeError("Scene is missing a valid bbox.")

    assets = select_assets(feature.get("assets", {}), requested_assets)

    return SceneSummary(
        product_id=product_id,
        title=title,
        acquisition_datetime=acquisition_dt,
        cloud_cover=cloud_cover,
        bbox=tuple(float(value) for value in bbox),
        footprint=feature.get("geometry"),
        assets=assets,
    )


def search_scenes(
    client: CopernicusClient,
    request: IngestionRequest,
    requested_assets: Iterable[str],
    search_limit: int,
) -> tuple[list[SceneSummary], Mapping[str, Any]]:
    payload = build_search_payload(request)
    payload["limit"] = search_limit
    response = post_json(client.endpoint, payload, auth=None)
    features = response.get("features", [])
    if not features:
        raise RuntimeError("No Sentinel-2 L2A scenes found for the requested window.")
    scenes = [scene_from_feature(feature, requested_assets) for feature in features]
    scenes_sorted = sorted(scenes, key=lambda scene: (scene.acquisition_datetime, scene.product_id))

    matched = response.get("context", {}).get("matched")
    if matched and matched > search_limit:
        logger.warning("STAC search matched %s scenes; limit=%s may truncate results.", matched, search_limit)

    return scenes_sorted, payload


def select_best_scenes_by_day(scenes: Iterable[SceneSummary], max_days: int) -> list[SceneSummary]:
    grouped: dict[str, list[SceneSummary]] = {}
    for scene in scenes:
        day_key = normalize_date_folder(scene.acquisition_datetime)
        grouped.setdefault(day_key, []).append(scene)

    best_per_day: list[SceneSummary] = []
    for day_key in sorted(grouped.keys()):
        day_scenes = grouped[day_key]
        day_scenes_sorted = sorted(
            day_scenes,
            key=lambda s: (
                s.cloud_cover if s.cloud_cover is not None else MISSING_CLOUD_COVER,
                s.acquisition_datetime,
                s.product_id,
            ),
        )
        best_per_day.append(day_scenes_sorted[0])

    return best_per_day[:max_days]


def build_plan(data_dir: Path, scenes: Iterable[SceneSummary]) -> list[ScenePlan]:
    plans: list[ScenePlan] = []
    for scene in scenes:
        target_dir = scene_dir(data_dir, scene.acquisition_datetime, scene.product_id)
        asset_paths: dict[str, Path] = {}
        for asset_name, href in scene.assets.items():
            filename = select_asset_filename(href, f"{scene.product_id}_{asset_name}.bin")
            asset_paths[asset_name] = target_dir / filename
        plans.append(
            ScenePlan(
                scene=scene,
                scene_dir=target_dir,
                asset_hrefs=dict(scene.assets),
                asset_paths=asset_paths,
            )
        )
    return plans


def plan_overview(plans: Iterable[ScenePlan]) -> str:
    lines = ["Planned Sentinel-2 time-series downloads:"]
    for idx, plan in enumerate(plans, start=1):
        day_key = normalize_date_folder(plan.scene.acquisition_datetime)
        lines.append(f"{idx:02d}. {day_key} | {plan.scene.product_id}")
        lines.append(f"    -> {plan.scene_dir}")
        for asset_name, asset_path in plan.asset_paths.items():
            lines.append(f"       - {asset_name}: {asset_path.name}")
    return "\n".join(lines)


def should_skip_scene(plan: ScenePlan, force: bool) -> bool:
    if force:
        return False
    meta = metadata_path(plan.scene_dir)
    if not meta.exists():
        return False
    for path in plan.asset_paths.values():
        if not path.exists():
            return False
    ndvi_path = plan.scene_dir / "ndvi.tif"
    return ndvi_path.exists()


def compute_ndvi_with_mask(
    red_path: Path,
    nir_path: Path,
    scl_path: Path,
    output_path: Path,
    *,
    mask_water: bool,
) -> str | None:
    red, red_profile = read_raster_band(red_path)
    nir, nir_profile = read_raster_band(nir_path)

    if red.shape != nir.shape:
        raise ValueError(f"Band shapes do not match: B04={red.shape}, B08={nir.shape}")
    if red_profile.get("transform") != nir_profile.get("transform"):
        raise ValueError("Band georeferencing does not match between B04 and B08")

    ndvi = compute_ndvi(red, nir)

    scl, scl_profile = read_raster_band(scl_path)
    scl_resampled = resample_to_match(scl, scl_profile, red_profile, method="nearest")
    valid_mask = scl_valid_mask(scl_resampled, exclude_water=mask_water)
    ndvi = ndvi.astype(np.float32, copy=True)
    ndvi[~valid_mask] = np.nan
    write_ndvi_geotiff(output_path, ndvi, red_profile)

    crs_value = red_profile.get("crs")
    return str(crs_value) if crs_value else None


def write_metadata(
    plan: ScenePlan,
    query: Mapping[str, Any],
    downloaded_assets: Mapping[str, Path],
    crs: str | None,
    *,
    mask_water: bool,
) -> None:
    meta = {
        "product_id": plan.scene.product_id,
        "title": plan.scene.title,
        "acquisition_datetime": plan.scene.acquisition_datetime.isoformat(),
        "cloud_cover": plan.scene.cloud_cover,
        "bbox": plan.scene.bbox,
        "footprint": plan.scene.footprint,
        "crs": crs,
        "source_endpoints": {
            "stac": STAC_ENDPOINT,
            "odata": ODATA_BASE,
        },
        "source_query": dict(query),
        "selected_assets": dict(plan.asset_hrefs),
        "downloaded_assets": {name: str(path) for name, path in downloaded_assets.items()},
        "ndvi_path": str(plan.scene_dir / "ndvi.tif"),
        "scl_excluded_classes": sorted(EXCLUDED_CLASSES),
        "mask_water": mask_water,
        "water_class": WATER_CLASS,
        "download_retries": DOWNLOAD_RETRIES,
    }
    plan.scene_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metadata_path(plan.scene_dir)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Sentinel-2 L2A time-series with NDVI outputs.")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--max-cloud-cover", type=float, default=None)
    parser.add_argument("--max-days", type=int, default=DEFAULT_MAX_DAYS)
    parser.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading.")
    parser.add_argument("--strict", action="store_true", help="Fail fast on the first asset error.")
    parser.add_argument(
        "--include-water",
        action="store_true",
        help="Include water pixels in the NDVI output (default masks water).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    cfg = load_config()
    setup_logging(cfg.log_level)

    args = parse_args(argv)
    bbox = validate_bbox(args.bbox)
    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)
    requested_assets = parse_assets(args.assets)
    data_dir = args.data_dir.expanduser().resolve()
    mask_water = not args.include_water

    auth = None
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    if username and password:
        auth = CopernicusAuth(username=username, password=password)

    client = CopernicusClient(auth=auth)
    search_limit = max(args.max_days * 20, SEARCH_LIMIT_FLOOR)
    request = IngestionRequest(
        bbox=bbox,
        date_from=date_from,
        date_to=date_to,
        max_scenes=search_limit,
        max_cloud_cover=args.max_cloud_cover,
    )

    scenes, query = search_scenes(client, request, requested_assets, search_limit=search_limit)
    selected = select_best_scenes_by_day(scenes, args.max_days)
    plans = build_plan(data_dir, selected)

    print(plan_overview(plans))

    if args.dry_run:
        print("Dry-run completed. No downloads executed.")
        return 0

    if not client.has_credentials():
        raise RuntimeError("Missing COPERNICUS_USERNAME/COPERNICUS_PASSWORD. Downloads require credentials.")

    for plan in plans:
        if should_skip_scene(plan, args.force):
            logger.info("Skipping cached scene %s", plan.scene.product_id)
            continue

        try:
            downloaded_assets: dict[str, Path] = {}
            for asset_name, href in plan.asset_hrefs.items():
                output_path = plan.asset_paths[asset_name]
                if output_path.exists() and not args.force:
                    logger.info("Using cached %s -> %s", asset_name, output_path)
                    downloaded_assets[asset_name] = output_path
                    continue
                path = client.download_asset(
                    asset_name,
                    href,
                    plan.scene_dir,
                    product_identifier=plan.scene.product_id,
                    retries=DOWNLOAD_RETRIES,
                )
                downloaded_assets[asset_name] = path

            ndvi_path = plan.scene_dir / "ndvi.tif"
            if ndvi_path.exists() and not args.force:
                logger.info("NDVI already computed -> %s", ndvi_path)
                crs = None
            else:
                crs = compute_ndvi_with_mask(
                    red_path=downloaded_assets["B04"],
                    nir_path=downloaded_assets["B08"],
                    scl_path=downloaded_assets["SCL"],
                    output_path=ndvi_path,
                    mask_water=mask_water,
                )

            write_metadata(plan, query, downloaded_assets, crs, mask_water=mask_water)
        except Exception as exc:  # noqa: BLE001 - controlled CLI flow
            logger.error("Failed to process scene %s: %s", plan.scene.product_id, exc, exc_info=True)
            if args.strict:
                raise
            continue

    return 0
