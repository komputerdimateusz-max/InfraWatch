"""Compute NDVI for a Sentinel-2 scene directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from infrawatch.analytics.ndvi import compute_ndvi
from infrawatch.analytics.raster_io import read_raster_band, write_ndvi_geotiff


def find_band(scene_dir: Path, band: str) -> Path:
    matches = sorted(scene_dir.glob(f"*{band}*.jp2"))
    if not matches:
        raise FileNotFoundError(f"No {band} JP2 file found in {scene_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {band} JP2 files found in {scene_dir}: {matches}")
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
    output_path = args.out or scene_dir / "ndvi.tif"
    write_ndvi_geotiff(output_path, ndvi, red_profile)


if __name__ == "__main__":
    main()
