"""Score traction risk from NDVI + corridor lines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from infrawatch.scoring.traction_risk import score_traction_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score traction risk for corridor lines.")
    parser.add_argument("--ndvi", required=True, type=Path, help="Path to ndvi.tif")
    parser.add_argument("--lines", required=True, type=Path, help="Path to GeoJSON LineString features")
    parser.add_argument("--buffer-m", type=float, default=20.0, help="Buffer distance in meters")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    return parser.parse_args()


def _write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results = score_traction_segments(args.ndvi, args.lines, buffer_m=args.buffer_m)

    out_path = args.out
    if out_path.suffix.lower() == ".csv":
        _write_csv(out_path, results)
        json_path = out_path.with_suffix(".json")
        _write_json(json_path, results)
    else:
        _write_json(out_path, results)

    if args.csv:
        _write_csv(args.csv, results)


if __name__ == "__main__":
    main()
