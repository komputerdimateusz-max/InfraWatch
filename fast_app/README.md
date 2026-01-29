# InfraWatch Fast App

A lightweight FastAPI + Leaflet visualization layer for InfraWatch. It reuses the shared scoring and NDVI discovery logic from `src/infrawatch/core` and can run independently from Streamlit.

## Run

```bash
pip install -e .
uvicorn fast_app.main:app --reload --port 8000
```

Then open:

```
http://localhost:8000
```

## Notes

- NDVI base directory defaults to `C:\InfraWatch\satellite_data\raw\s2`.
- Override the base directory with:

```bash
set INFRAWATCH_NDVI_BASE_DIR=C:\path\to\satellite_data\raw\s2
```

(or `export INFRAWATCH_NDVI_BASE_DIR=/path/to/s2` on macOS/Linux).
- If NDVI files are missing, the app returns `NO_DATA` rows and warnings instead of crashing.
