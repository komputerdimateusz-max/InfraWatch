from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import base64
import json
import logging
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import IngestionRequest, SceneMetadata, SceneSummary
from .paths import as_bbox_list, select_asset_filename

# Copernicus Data Space Ecosystem (CDSE) STAC API v1
STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/search"
# CDSE STAC collection id for Sentinel-2 Level-2A
COLLECTION_ID = "sentinel-2-l2a"
DOWNLOAD_TIMEOUT = 120

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopernicusAuth:
    username: str
    password: str

    def header(self) -> str:
        token = f"{self.username}:{self.password}".encode("utf-8")
        return "Basic " + base64.b64encode(token).decode("ascii")


class CopernicusClient:
    def __init__(self, auth: CopernicusAuth | None = None, endpoint: str = STAC_ENDPOINT) -> None:
        self.auth = auth
        self.endpoint = endpoint

    def has_credentials(self) -> bool:
        return self.auth is not None

    def search(self, request: IngestionRequest) -> tuple[list[SceneSummary], Mapping[str, Any]]:
        payload = build_search_payload(request)
        response = post_json(self.endpoint, payload, auth=self.auth)
        features = response.get("features", [])
        if not features:
            raise RuntimeError("No Sentinel-2 L2A scenes found for the requested window.")
        scenes = [scene_from_feature(feature) for feature in features]
        sorted_scenes = sorted(scenes, key=lambda scene: (scene.acquisition_datetime, scene.product_id))
        return sorted_scenes[: request.max_scenes], payload

    def download_scene(self, scene: SceneSummary, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = select_asset_filename(scene.download_url, f"{scene.product_id}.zip")
        output_path = destination_dir / filename
        logger.info("Downloading %s -> %s", scene.product_id, output_path)
        headers = {}
        if self.auth:
            headers["Authorization"] = self.auth.header()
        request = Request(scene.download_url, headers=headers)
        try:
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                with output_path.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
        except HTTPError as exc:
            raise RuntimeError(http_error_message(exc, scene.download_url)) from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while downloading {scene.download_url}: {exc}") from exc
        return output_path


def build_search_payload(request: IngestionRequest) -> dict[str, Any]:
    datetime_filter = f"{request.date_from.isoformat()}T00:00:00Z/{request.date_to.isoformat()}T23:59:59Z"
    payload: dict[str, Any] = {
        "collections": [COLLECTION_ID],
        "bbox": as_bbox_list(request.bbox),
        "datetime": datetime_filter,
        "limit": request.max_scenes,
    }
    if request.max_cloud_cover is not None:
        payload["query"] = {"eo:cloud_cover": {"lte": request.max_cloud_cover}}
    return payload


def post_json(endpoint: str, payload: Mapping[str, Any], auth: CopernicusAuth | None = None) -> Mapping[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth.header()
    request = Request(endpoint, data=data, headers=headers)
    try:
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(http_error_message(exc, endpoint)) from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while contacting {endpoint}: {exc}") from exc


def scene_from_feature(feature: Mapping[str, Any]) -> SceneSummary:
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
    download_url = select_download_url(feature.get("assets", {}))
    return SceneSummary(
        product_id=product_id,
        title=title,
        acquisition_datetime=acquisition_dt,
        cloud_cover=cloud_cover,
        bbox=tuple(float(value) for value in bbox),
        footprint=feature.get("geometry"),
        download_url=download_url,
    )


def select_download_url(assets: Mapping[str, Any]) -> str:
    for key in ("download", "data", "product", "analytic", "visual"):
        asset = assets.get(key)
        if asset and "href" in asset:
            return asset["href"]
    for asset in assets.values():
        if isinstance(asset, Mapping) and "href" in asset:
            return asset["href"]
    raise RuntimeError("No downloadable asset found in STAC item.")


def parse_datetime(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)


def scene_metadata(scene: SceneSummary, endpoint: str, query: Mapping[str, Any]) -> SceneMetadata:
    return SceneMetadata(
        product_id=scene.product_id,
        title=scene.title,
        acquisition_datetime=scene.acquisition_datetime,
        cloud_cover=scene.cloud_cover,
        bbox=scene.bbox,
        footprint=scene.footprint,
        source_endpoint=endpoint,
        source_query=query,
    )


def http_error_message(error: HTTPError, url: str) -> str:
    if error.code in {401, 403}:
        return (
            f"Authentication failed when contacting {url}. "
            "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD env vars."
        )
    return f"HTTP {error.code} error while contacting {url}: {error.reason}"
