from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import base64
import json
import logging
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .models import IngestionRequest, SceneMetadata, SceneSummary
from .paths import as_bbox_list, select_asset_filename

# CDSE STAC API v1
STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION_ID = "sentinel-2-l2a"
DOWNLOAD_TIMEOUT = 120

# CDSE Auth + OData (for downloading files via Nodes())
IDENTITY_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopernicusAuth:
    username: str
    password: str

    def basic_header(self) -> str:
        token = f"{self.username}:{self.password}".encode("utf-8")
        return "Basic " + base64.b64encode(token).decode("ascii")


class CopernicusClient:
    def __init__(self, auth: CopernicusAuth | None = None, endpoint: str = STAC_ENDPOINT) -> None:
        self.auth = auth
        self.endpoint = endpoint
        self._access_token: str | None = None

    def has_credentials(self) -> bool:
        return self.auth is not None

    def search(self, request: IngestionRequest) -> tuple[list[SceneSummary], Mapping[str, Any]]:
        payload = build_search_payload(request)
        response = post_json(self.endpoint, payload, auth=None)
        features = response.get("features", [])
        if not features:
            raise RuntimeError("No Sentinel-2 L2A scenes found for the requested window.")
        scenes = [scene_from_feature(feature) for feature in features]
        sorted_scenes = sorted(scenes, key=lambda scene: (scene.acquisition_datetime, scene.product_id))
        return sorted_scenes[: request.max_scenes], payload

    def _get_access_token(self) -> str:
        if self._access_token:
            return self._access_token
        if not self.auth:
            raise RuntimeError(
                "Missing credentials. Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD to enable downloads."
            )
        data = (
            "client_id=cdse-public"
            "&grant_type=password"
            f"&username={quote(self.auth.username)}"
            f"&password={quote(self.auth.password)}"
        ).encode("utf-8")
        req = Request(
            IDENTITY_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        token = payload.get("access_token")
        if not token:
            raise RuntimeError("Could not obtain access token from CDSE.")
        self._access_token = token
        return token

    def _odata_product_uuid_for_name(self, product_safe_name: str) -> str:
        token = self._get_access_token()
        filter_expr = f"Name eq '{product_safe_name}'"
        qs = urlencode({"$filter": filter_expr})
        url = f"{ODATA_BASE}/Products?{qs}"

        req = Request(url, headers={"Authorization": f"Bearer {token}"})
        with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        items = payload.get("value", [])
        if not items:
            raise RuntimeError(f"OData: product not found for Name='{product_safe_name}'.")
        return str(items[0]["Id"])

    def _download_via_odata_nodes(self, s3_href: str, destination_dir: Path) -> Path:
        rel = s3_href[len("s3://eodata/") :]
        parts = rel.split("/")

        safe_idx = next(i for i, p in enumerate(parts) if p.endswith(".SAFE"))
        product_safe_name = parts[safe_idx]
        inside_parts = parts[safe_idx + 1 :]

        product_uuid = self._odata_product_uuid_for_name(product_safe_name)

        def node(seg: str) -> str:
            return f"Nodes({quote(seg, safe='-_.()')})"

        nodes_chain = "/".join([node(product_safe_name)] + [node(p) for p in inside_parts])
        url = f"{ODATA_BASE}/Products({product_uuid})/{nodes_chain}/$value"

        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        destination_dir.mkdir(parents=True, exist_ok=True)
        output_path = destination_dir / inside_parts[-1]

        logger.info("Downloading via OData Nodes -> %s", output_path)

        req = Request(url, headers=headers)
        with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            with output_path.open("wb") as handle:
                while chunk := resp.read(1024 * 1024):
                    handle.write(chunk)

        return output_path

    def download_scene(self, scene: SceneSummary, destination_dir: Path) -> Path:
        if scene.download_url.startswith("s3://eodata/"):
            return self._download_via_odata_nodes(scene.download_url, destination_dir)

        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = select_asset_filename(scene.download_url, f"{scene.product_id}.bin")
        output_path = destination_dir / filename

        headers = {}
        if self.auth:
            headers["Authorization"] = self.auth.basic_header()

        req = Request(scene.download_url, headers=headers)
        with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            with output_path.open("wb") as handle:
                while chunk := resp.read(1024 * 1024):
                    handle.write(chunk)

        return output_path


def build_search_payload(request: IngestionRequest) -> dict[str, Any]:
    return {
        "collections": [COLLECTION_ID],
        "bbox": as_bbox_list(request.bbox),
        "datetime": f"{request.date_from.isoformat()}T00:00:00Z/{request.date_to.isoformat()}T23:59:59Z",
        "limit": request.max_scenes,
    }


def post_json(endpoint: str, payload: Mapping[str, Any], auth: CopernicusAuth | None = None) -> Mapping[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth.basic_header()
    req = Request(endpoint, data=data, headers=headers)
    with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def scene_from_feature(feature: Mapping[str, Any]) -> SceneSummary:
    props = feature.get("properties", {})
    dt_raw = props.get("datetime") or props.get("start_datetime")
    acquisition_dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))

    product_id = props.get("productIdentifier") or feature.get("id") or "unknown"
    title = props.get("title") or product_id
    cloud_cover = props.get("eo:cloud_cover")
    bbox = tuple(feature.get("bbox", []))

    assets = feature.get("assets", {})
    logger.info("STAC assets keys: %s", list(assets.keys()))

    download_url = select_download_url(assets)

    return SceneSummary(
        product_id=product_id,
        title=title,
        acquisition_datetime=acquisition_dt,
        cloud_cover=cloud_cover,
        bbox=bbox,
        footprint=feature.get("geometry"),
        download_url=download_url,
    )


def select_download_url(assets: Mapping[str, Any]) -> str:
    for key in ("B04", "B08", "SCL", "download", "data"):
        asset = assets.get(key)
        if asset and "href" in asset:
            return asset["href"]
    for asset in assets.values():
        if isinstance(asset, Mapping) and "href" in asset:
            return asset["href"]
    raise RuntimeError("No downloadable asset found in STAC item.")
