"""Compute NDVI for a Sentinel-2 scene directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from infrawatch.analytics.ndvi import compute_ndvi
from infrawatch.analytics.raster_io import read_raster_band, write_ndvi_geotiff
from infrawatch.analytics.resample import resample_to_match
from infrawatch.analytics.scl import scl_valid_mask


def find_band(scene_dir: Path, band: str) -> Path:
    matches = sorted(scene_dir.glob(f"*{band}*.jp2"))
    if not matches:
        raise FileNotFoundError(f"No {band} JP2 file found in {scene_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {band} JP2 files found in {scene_dir}: {matches}")
    return matches[0]


def find_scl(scene_dir: Path) -> Path | None:
    scl_candidates = ["SCL", "scl", "SCL_20m", "SCL20", "classification"]
    matches: list[Path] = []
    for path in scene_dir.glob("*.jp2"):
        name = path.name
        if any(token in name for token in scl_candidates):
            matches.append(path)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple SCL JP2 files found in {scene_dir}: {matches}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NDVI for a Sentinel-2 scene folder.")
    parser.add_argument("--scene-dir", required=True, type=Path, help="Path to scene folder")
    parser.add_argument("--out", type=Path, help="Output GeoTIFF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    red_path = find_band(scene_dir, "B04")
    nir_path = find_band(scene_dir, "B08")

    red, red_profile = read_raster_band(red_path)
    nir, nir_profile = read_raster_band(nir_path)

    if red.shape != nir.shape:
        raise ValueError(
            f"Band shapes do not match: B04={red.shape}, B08={nir.shape}"
        )
    if red_profile.get("transform") != nir_profile.get("transform"):
        raise ValueError("Band georeferencing does not match between B04 and B08")

    ndvi = compute_ndvi(red, nir)
    scl_path = find_scl(scene_dir)
    if scl_path:
        scl, scl_profile = read_raster_band(scl_path)
        scl_resampled = resample_to_match(scl, scl_profile, red_profile, method="nearest")
        valid_mask = scl_valid_mask(scl_resampled, exclude_water=True)
        ndvi = ndvi.astype(np.float32, copy=True)
        ndvi[~valid_mask] = np.nan
    output_path = args.out or scene_dir / "ndvi.tif"
    write_ndvi_geotiff(output_path, ndvi, red_profile)


if __name__ == "__main__":
    main()
