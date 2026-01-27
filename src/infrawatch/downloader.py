"""NDVI downloader helpers for Sentinel-2 data sources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import rasterio

try:
    from pystac_client import Client
except ImportError:  # pragma: no cover - optional dependency
    Client = None

CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"


@dataclass
class Scene:
    scene_id: str
    date: datetime
    cloud_cover: float | None
    tile_id: str | None
    assets: dict[str, str]
    backend: str
    preview: str | None


@dataclass
class SearchParams:
    bbox: tuple[float, float, float, float]
    date_range: tuple[date, date]
    cloud_max: float
    backend: str


class DownloaderError(RuntimeError):
    """Raised when a download or search fails."""


def search_scenes(
    aoi: tuple[float, float, float, float],
    date_range: tuple[date, date],
    cloud_max: float,
    backend: str,
) -> list[Scene]:
    params = SearchParams(bbox=aoi, date_range=date_range, cloud_max=cloud_max, backend=backend)
    if backend == "Copernicus Data Space (CDSE)":
        return _search_cdse(params)
    if backend == "AWS Open Data (Earth Search)":
        return _search_earth_search(params)
    raise DownloaderError(f"Unsupported backend: {backend}")


def download_scene(scene: Scene, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    return derive_or_fetch_ndvi(scene, target_dir)


def derive_or_fetch_ndvi(scene: Scene, target_dir: Path) -> Path:
    ndvi_path = target_dir / "ndvi.tif"
    if ndvi_path.exists():
        return ndvi_path

    red_asset = _resolve_asset(scene.assets, ["B04", "red", "B4"])
    nir_asset = _resolve_asset(scene.assets, ["B08", "nir", "B8"])
    if not red_asset or not nir_asset:
        raise DownloaderError(
            "Unable to locate red/NIR assets for NDVI. "
            "Ensure the selected scene includes B04 and B08 bands."
        )

    red_path = target_dir / "B04.tif"
    nir_path = target_dir / "B08.tif"

    _download_asset(scene.backend, red_asset, red_path)
    _download_asset(scene.backend, nir_asset, nir_path)

    _compute_ndvi(red_path, nir_path, ndvi_path)
    return ndvi_path


def _resolve_asset(assets: dict[str, str], keys: Iterable[str]) -> str | None:
    for key in keys:
        href = assets.get(key)
        if href:
            return href
    return None


def _search_cdse(params: SearchParams) -> list[Scene]:
    _require_pystac_client()
    client = Client.open(CDSE_STAC_URL)
    time_range = f"{params.date_range[0].isoformat()}/{params.date_range[1].isoformat()}"
    search = client.search(
        collections=["SENTINEL-2-L2A"],
        bbox=list(params.bbox),
        datetime=time_range,
        query={"eo:cloud_cover": {"lte": params.cloud_max}},
        max_items=50,
    )
    items = list(search.get_items())
    return _items_to_scenes(items, backend=params.backend)


def _search_earth_search(params: SearchParams) -> list[Scene]:
    _require_pystac_client()
    client = Client.open(EARTH_SEARCH_URL)
    time_range = f"{params.date_range[0].isoformat()}/{params.date_range[1].isoformat()}"
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=list(params.bbox),
        datetime=time_range,
        query={"eo:cloud_cover": {"lte": params.cloud_max}},
        max_items=50,
    )
    items = list(search.get_items())
    return _items_to_scenes(items, backend=params.backend)


def _items_to_scenes(items: Iterable, backend: str) -> list[Scene]:
    scenes: list[Scene] = []
    for item in items:
        assets = {key: asset.href for key, asset in item.assets.items()}
        cloud_cover = None
        if item.properties:
            cloud_cover = item.properties.get("eo:cloud_cover")
        tile_id = item.properties.get("s2:mgrs_tile") if item.properties else None
        preview = None
        for key in ("thumbnail", "visual", "rendered_preview"):
            asset = item.assets.get(key)
            if asset:
                preview = asset.href
                break
        scenes.append(
            Scene(
                scene_id=item.id,
                date=item.datetime or datetime.utcnow(),
                cloud_cover=cloud_cover,
                tile_id=tile_id,
                assets=assets,
                backend=backend,
                preview=preview,
            )
        )
    scenes.sort(key=lambda s: s.date, reverse=True)
    return scenes


def _download_asset(backend: str, href: str, target_path: Path) -> None:
    headers = {}
    if backend == "Copernicus Data Space (CDSE)":
        token = _get_cdse_token()
        if token is None:
            raise DownloaderError(
                "CDSE credentials missing. Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env."
            )
        headers["Authorization"] = f"Bearer {token}"

    with requests.get(href, headers=headers, stream=True, timeout=180) as response:
        if response.status_code != 200:
            raise DownloaderError(
                f"Download failed ({response.status_code}). "
                "If using CDSE, confirm credentials and permissions."
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as target_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target_file.write(chunk)


def _get_cdse_token() -> str | None:
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    if not username or not password:
        return None

    token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )
    payload = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    response = requests.post(token_url, data=payload, timeout=60)
    if response.status_code != 200:
        raise DownloaderError(
            "Failed to authenticate with CDSE. Verify COPERNICUS_USERNAME and COPERNICUS_PASSWORD."
        )
    token = response.json().get("access_token")
    if not token:
        raise DownloaderError("CDSE authentication did not return an access token.")
    return token


def _compute_ndvi(red_path: Path, nir_path: Path, ndvi_path: Path) -> None:
    with rasterio.open(red_path) as red_ds, rasterio.open(nir_path) as nir_ds:
        red = red_ds.read(1).astype("float32")
        nir = nir_ds.read(1).astype("float32")
        red_nodata = red_ds.nodata
        nir_nodata = nir_ds.nodata

        mask = np.zeros_like(red, dtype=bool)
        if red_nodata is not None:
            mask |= red == red_nodata
        if nir_nodata is not None:
            mask |= nir == nir_nodata

        denom = nir + red
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / denom
        ndvi = np.where(denom == 0, np.nan, ndvi)
        ndvi = np.where(mask, np.nan, ndvi)

        profile = red_ds.profile.copy()
        profile.update(dtype="float32", count=1, compress="deflate", nodata=-9999.0)
        ndvi_out = np.where(np.isnan(ndvi), profile["nodata"], ndvi).astype("float32")

        ndvi_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(ndvi_path, "w", **profile) as dst:
            dst.write(ndvi_out, 1)


def _require_pystac_client() -> None:
    if Client is None:
        raise RuntimeError(
            "pystac-client is required for the NDVI downloader. "
            "Install it with `pip install pystac-client`."
        )
