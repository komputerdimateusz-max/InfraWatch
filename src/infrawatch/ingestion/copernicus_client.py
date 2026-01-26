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
        """
        Resolve OData product UUID by querying:
          Products?$filter=Name eq '<PRODUCT>.SAFE'

        IMPORTANT: The $filter value must be URL-encoded (spaces, quotes).
        """
        token = self._get_access_token()

        # Build the filter expression and encode it properly
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
        Convert:
          s3://eodata/.../<PRODUCT>.SAFE/<path/to/file>
        into:
          /Products(<uuid>)/Nodes(<PRODUCT>.SAFE)/Nodes(...)/$value

        Then download the file using Bearer token.
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
        filename = inside_parts[-1]
        output_path = destination_dir / filename

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

    def download_scene(self, scene: SceneSummary, destination_dir: Path) -> Path:
        if scene.download_url.startswith("s3://eodata/"):
            return self._download_via_odata_nodes(scene.download_url, destination_dir)

        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = select_asset_filename(scene.download_url, f"{scene.product_id}.bin")
        output_path = destination_dir / filename

        logger.info("Downloading %s -> %s", scene.product_id, output_path)
        headers: dict[str, str] = {}
        if self.auth:
            headers["Authorization"] = self.auth.basic_header()

        request = Request(scene.download_url, headers=headers)
        try:
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                with output_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
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
