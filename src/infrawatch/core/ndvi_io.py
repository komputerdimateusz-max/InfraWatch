"""NDVI discovery and metadata helpers shared across apps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from infrawatch.utils.crs import transform_bounds_always_xy

DATE_PATTERN = re.compile(r"(20\d{6})")
DATE_FOLDER_PATTERN = re.compile(r"^\d{8}$")
DEFAULT_NDVI_BASE_DIR = Path(r"C:\InfraWatch\satellite_data\raw\s2")
NDVI_FILENAME = "ndvi.tif"


@dataclass
class NdviDetection:
    path: Path | None
    scene_date: str | None
    scene_folder: Path | None


@dataclass
class NdviCandidate:
    path: Path
    size: int
    mtime: float
    reason: str
    msil2a: bool
    has_ndvi: bool
    ndvi_prefix: bool


@dataclass
class NdviScanMeta:
    selected_path: str | None
    candidates_count: int
    reason: str | None
    file_size: int | None
    mtime: float | None
    candidates: list[str]
    warning: str | None


@dataclass
class NdviInventory:
    inventory: dict[str, str]
    dates_sorted: list[str]
    meta: dict[str, NdviScanMeta]
    warnings: list[str]


def format_date_label(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return date_str


def parse_date_label(date_str: str) -> str | None:
    if DATE_FOLDER_PATTERN.match(date_str):
        return date_str
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError:
        return None


def _rank_candidate(candidate: NdviCandidate) -> tuple[int, int, int, int, float]:
    return (
        1 if candidate.msil2a else 0,
        1 if candidate.has_ndvi else 0,
        1 if candidate.ndvi_prefix else 0,
        candidate.size,
        candidate.mtime,
    )


def _candidate_from_path(path: Path, reason: str) -> NdviCandidate:
    stat = path.stat()
    name_lower = path.name.lower()
    path_lower = str(path).lower()
    return NdviCandidate(
        path=path,
        size=stat.st_size,
        mtime=stat.st_mtime,
        reason=reason,
        msil2a="msil2a" in path_lower,
        has_ndvi="ndvi" in name_lower,
        ndvi_prefix=path.name.upper().startswith("NDVI_"),
    )


def _find_ndvi_candidates(date_dir: Path) -> list[NdviCandidate]:
    tif_paths = [
        path
        for path in date_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    primary = [path for path in tif_paths if path.name.lower() == NDVI_FILENAME]
    if primary:
        return [_candidate_from_path(path, "primary_exact") for path in primary]

    named_ndvi = [path for path in tif_paths if "ndvi" in path.name.lower()]
    if named_ndvi:
        return [_candidate_from_path(path, "fallback_ndvi_name") for path in named_ndvi]

    if tif_paths:
        return [_candidate_from_path(path, "fallback_any_tif") for path in tif_paths]
    return []


def _select_ndvi_candidate(candidates: list[NdviCandidate]) -> tuple[NdviCandidate | None, str | None]:
    if not candidates:
        return None, None

    sorted_candidates = sorted(candidates, key=_rank_candidate, reverse=True)
    valid_candidates: list[NdviCandidate] = []
    import rasterio

    for candidate in sorted_candidates:
        try:
            with rasterio.open(candidate.path):
                valid_candidates.append(candidate)
        except Exception:  # noqa: BLE001
            continue

    if not valid_candidates:
        return None, None

    valid_candidates.sort(key=_rank_candidate, reverse=True)
    selected = valid_candidates[0]
    top_rank = _rank_candidate(selected)
    ambiguous = [cand for cand in valid_candidates if _rank_candidate(cand) == top_rank]
    warning = None
    if len(ambiguous) > 1:
        warning = (
            "Multiple NDVI candidates matched equally; selected best match. "
            f"Candidates: {', '.join(str(c.path) for c in valid_candidates)}"
        )
    return selected, warning


@lru_cache(maxsize=16)
def scan_ndvi_inventory(base_dir: str | Path) -> NdviInventory:
    base_path = Path(base_dir).expanduser()
    if not base_path.exists():
        return NdviInventory({}, [], {}, [f"Base directory not found: {base_path}"])

    date_dirs = [
        path
        for path in base_path.iterdir()
        if path.is_dir() and DATE_FOLDER_PATTERN.match(path.name)
    ]
    inventory: dict[str, str] = {}
    meta: dict[str, NdviScanMeta] = {}
    warnings: list[str] = []

    for date_dir in sorted(date_dirs, key=lambda p: p.name, reverse=True):
        candidates = _find_ndvi_candidates(date_dir)
        selected, warning = _select_ndvi_candidate(candidates)
        if warning:
            warnings.append(f"{date_dir.name}: {warning}")
        if not selected:
            meta[date_dir.name] = NdviScanMeta(
                selected_path=None,
                candidates_count=len(candidates),
                reason=None,
                file_size=None,
                mtime=None,
                candidates=[str(c.path) for c in candidates],
                warning=warning,
            )
            continue

        inventory[date_dir.name] = str(selected.path)
        meta[date_dir.name] = NdviScanMeta(
            selected_path=str(selected.path),
            candidates_count=len(candidates),
            reason=selected.reason,
            file_size=selected.size,
            mtime=selected.mtime,
            candidates=[str(c.path) for c in candidates],
            warning=warning,
        )

    dates_sorted = sorted(inventory.keys(), reverse=True)
    return NdviInventory(inventory, dates_sorted, meta, warnings)


def detect_latest_ndvi(base_dir: Path) -> NdviDetection:
    candidates = list(base_dir.rglob(NDVI_FILENAME))
    if not candidates:
        return NdviDetection(path=None, scene_date=None, scene_folder=None)

    dated: list[tuple[datetime, Path]] = []
    undated: list[Path] = []
    for path in candidates:
        parsed = _parse_scene_date(path)
        if parsed:
            dated.append((parsed, path))
        else:
            undated.append(path)

    if dated:
        dated.sort(key=lambda item: item[0])
        scene_date, selected = dated[-1]
        return NdviDetection(
            path=selected,
            scene_date=scene_date.strftime("%Y-%m-%d"),
            scene_folder=selected.parent,
        )

    selected = max(undated, key=lambda item: item.stat().st_mtime)
    return NdviDetection(
        path=selected,
        scene_date=datetime.fromtimestamp(selected.stat().st_mtime).strftime("%Y-%m-%d"),
        scene_folder=selected.parent,
    )


def _parse_scene_date(path: Path) -> datetime | None:
    for part in path.parts:
        match = DATE_PATTERN.search(part)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d")
            except ValueError:
                continue
    return None


def resolve_ndvi_path_for_date(
    date_label: str,
    inventory: NdviInventory,
) -> tuple[Path | None, str | None]:
    date_key = parse_date_label(date_label)
    if date_key is None:
        return None, f"Invalid date format: {date_label}"
    resolved = inventory.inventory.get(date_key)
    if resolved is None:
        return None, f"NDVI missing for date {date_label}"
    return Path(resolved), None


@lru_cache(maxsize=32)
def read_ndvi_metadata(ndvi_path: str | Path) -> tuple[Any, Any, list[list[float]] | None]:
    """Return NDVI CRS, bounds in native CRS, and bounds in EPSG:4326 (lon/lat)."""
    import rasterio
    from rasterio.crs import CRS

    with rasterio.open(ndvi_path) as dataset:
        bounds = dataset.bounds
        crs = dataset.crs
    if crs is None:
        return None, bounds, None
    wgs_bounds = transform_bounds_always_xy(crs, CRS.from_epsg(4326), bounds)
    bounds_list = [[wgs_bounds[0], wgs_bounds[1]], [wgs_bounds[2], wgs_bounds[3]]]
    return crs, bounds, bounds_list
