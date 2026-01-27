# Sentinel-2 Ingestion (MVP)

## Purpose
This module performs deterministic discovery and caching of Sentinel-2 L2A imagery from the Copernicus Data Space Ecosystem. It is the first step before vegetation analytics (e.g., NDVI). The ingestion output is auditable and reproducible: each download includes a metadata JSON with the query and source endpoint used.

## Data source
- **Endpoint:** `https://catalogue.dataspace.copernicus.eu/stac/search`
- **Collection:** `SENTINEL-2`
- **Product type filter:** `S2MSI2A` (Sentinel-2 L2A)

## How it works
1. Build a STAC search query for the requested bbox and date range.
2. Optionally apply a **max cloud cover** filter (`eo:cloud_cover <= value`).
3. Sort scenes by acquisition datetime ascending (ties broken by product id).
4. Cache results under `./satellite_data/raw/s2/{YYYYMMDD}/{PRODUCT_ID}/` with a `metadata.json`.

### Determinism & caching
- The scene list is **sorted by acquisition datetime** ascending.
- Folder names use the **UTC acquisition date** and a sanitized product id.
- If `metadata.json` already exists for a scene, the download is skipped.

## How to run
```bash
python scripts/ingest_s2.py \
  --bbox 14.0 52.0 15.5 53.1 \
  --date-from 2025-06-01 \
  --date-to 2025-06-30 \
  --max-scenes 3 \
  --dry-run
```

## Time-series NDVI downloader
For multi-date NDVI-ready downloads (B04/B08/SCL + ndvi.tif) use the time-series script:
```bash
python scripts/download_s2_timeseries.py \
  --bbox 14.0 52.0 15.5 53.1 \
  --date-from 2025-06-01 \
  --date-to 2025-07-31 \
  --max-days 5 \
  --max-cloud-cover 30 \
  --dry-run
```

### Time-series output layout
```
./satellite_data/raw/s2/{YYYYMMDD}/{PRODUCT_ID}/
  \- <B04_10m.jp2>
  \- <B08_10m.jp2>
  \- <SCL_20m.jp2>
  \- ndvi.tif
  \- metadata.json
```

### Optional cloud filter
```bash
python scripts/ingest_s2.py \
  --bbox 14.0 52.0 15.5 53.1 \
  --date-from 2025-06-01 \
  --date-to 2025-06-30 \
  --max-scenes 3 \
  --max-cloud-cover 20
```

## What is downloaded
- A single downloadable asset per scene (typically a zipped product)
- `metadata.json` capturing:
  - product id / title
  - acquisition datetime
  - cloud cover (if available)
  - bbox / footprint
  - source endpoint and query parameters

## Where data is stored
```
./satellite_data/raw/s2/{YYYYMMDD}/{PRODUCT_ID}/
  \- <downloaded-asset>
  \- metadata.json
```

## Credentials (optional)
Copernicus downloads require credentials. Set these env vars if you want to download:
- `COPERNICUS_USERNAME`
- `COPERNICUS_PASSWORD`

If credentials are **not** set, the CLI still works in `--dry-run` mode and explains how to configure auth.

## Limitations (MVP)
- `ingest_s2.py` downloads a single asset per scene (as provided by the STAC item).
- NDVI computation is available only in the time-series downloader.
- Only Sentinel-2 L2A (S2MSI2A) is queried.

## Next steps
- Add integrity checks (checksum validation) for downloaded products.
- Add retries/backoff for transient network issues.
- Add scene filtering by tile or orbit metadata.
