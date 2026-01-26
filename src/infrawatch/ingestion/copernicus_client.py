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
        response = post_json(self.endpoint, payload, auth=None)  # STAC search works without auth
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
        try:
            with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(http_error_message(exc, IDENTITY_TOKEN_URL)) from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while contacting {IDENTITY_TOKEN_URL}: {exc}") from exc

        token = payload.get("access_token")
        if not token:
            raise RuntimeError("Could not obtain access token from CDSE (missing access_token in response).")
        self._access_token = token
        return token

    def _odata_product_uuid_for_name(self, product_safe_name: str) -> str:
        token = self._get_access_token()
        filter_expr = f"Name eq '{product_safe_name}'"
        qs = urlencode({"$filter": filter_expr})
        url = f"{ODATA_BASE}/Products?{qs}"

        req = Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(http_error_message(exc, url)) from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while contacting {url}: {exc}") from exc

        items = payload.get("value", [])
        if not items:
            raise RuntimeError(f"OData: product not found for Name='{product_safe_name}'.")
        product_id = items[0].get("Id")
        if not product_id:
            raise RuntimeError("OData: product record is missing Id.")
        return str(product_id)

    def _download_via_odata_nodes(self, s3_href: str, destination_dir: Path) -> Path:
        """
        Download a single file from a product given an s3://eodata/... href via OData Nodes().
        """
        if not s3_href.startswith("s3://eodata/"):
            raise RuntimeError(f"Expected s3://eodata/ href, got: {s3_href}")

        rel = s3_href[len("s3://eodata/") :]
        parts = rel.split("/")

        safe_idx = None
        for i, p in enumerate(parts):
            if p.endswith(".SAFE"):
                safe_idx = i
                break
        if safe_idx is None:
            raise RuntimeError(f"Could not locate .SAFE in s3 path: {s3_href}")

        product_safe_name = parts[safe_idx]
        inside_parts = parts[safe_idx + 1 :]
        if not inside_parts:
            raise RuntimeError(f"No file path inside product for: {s3_href}")

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

        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                with output_path.open("wb") as handle:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            return output_path
        except HTTPError as exc:
            raise RuntimeError(http_error_message(exc, url)) from exc
        except URLError as exc:
            raise RuntimeError(f"Network error while downloading {url}: {exc}") from exc

    def download_scene(self, scene: SceneSummary, destination_dir: Path) -> dict[str, Path]:
        """
        Download all requested assets for a scene.
        Returns: mapping band -> local file path
        """
        downloaded: dict[str, Path] = {}
        for band, href in scene.assets.items():
            # Choose filename by href; if it lacks extension, fallback to band
            filename_fallback = f"{scene.product_id}_{band}.bin"
            _ = select_asset_filename(href, filename_fallback)  # keep behavior consistent even if not used
            path = self._download_via_odata_nodes(href, destination_dir) if href.startswith("s3://eodata/") else None
            if path is None:
                # Fallback HTTPS download (rare); keep minimal
                destination_dir.mkdir(parents=True, exist_ok=True)
                fname = select_asset_filename(href, filename_fallback)
                output_path = destination_dir / fname
                headers: dict[str, str] = {}
                if self.auth:
                    headers["Authorization"] = self.auth.basic_header()
                req = Request(href, headers=headers)
                try:
                    with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                        with output_path.open("wb") as handle:
                            while True:
                                chunk = resp.read(1024 * 1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                except HTTPError as exc:
                    raise RuntimeError(http_error_message(exc, href)) from exc
                except URLError as exc:
                    raise RuntimeError(f"Network error while downloading {href}: {exc}") from exc
                path = output_path

            downloaded[band] = path
        return downloaded


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
        headers["Authorization"] = auth.basic_header()
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

    assets = select_ndvi_assets(feature.get("assets", {}))

    return SceneSummary(
        product_id=product_id,
        title=title,
        acquisition_datetime=acquisition_dt,
        cloud_cover=cloud_cover,
        bbox=tuple(float(value) for value in bbox),
        footprint=feature.get("geometry"),
        assets=assets,
    )


def select_ndvi_assets(assets: Mapping[str, Any]) -> dict[str, str]:
    """
    Select the minimum assets required to compute NDVI: B04 (red), B08 (nir), and SCL.
    STAC implementations may differ in exact keys; we try a few common patterns.
    """
    def pick(keys: list[str]) -> str | None:
        for k in keys:
            asset = assets.get(k)
            if asset and isinstance(asset, Mapping) and "href" in asset:
                return str(asset["href"])
        return None

    b04 = pick(["B04", "b04", "red", "B04_10m", "B04_20m"])
    b08 = pick(["B08", "b08", "nir", "B08_10m", "B08_20m"])
    scl = pick(["SCL", "scl", "SCL_20m", "SCL20", "classification"])

    missing = [name for name, val in (("B04", b04), ("B08", b08), ("SCL", scl)) if not val]
    if missing:
        raise RuntimeError(
            f"Missing required NDVI assets {missing}. Available STAC asset keys: {list(assets.keys())}"
        )

    return {"B04": b04, "B08": b08, "SCL": scl}  # type: ignore[return-value]


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
        source_query=dict(query),
    )


def http_error_message(error: HTTPError, url: str) -> str:
    if error.code in {401, 403}:
        return (
            f"Authentication failed when contacting {url}. "
            "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD env vars."
        )
    return f"HTTP {error.code} error while contacting {url}: {error.reason}"
