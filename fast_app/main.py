"""FastAPI app for lightweight InfraWatch visualization."""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from infrawatch.core.geo import validate_line_feature_collection
from infrawatch.core.models import ScoreRequest
from infrawatch.core.ndvi_io import (
    DEFAULT_NDVI_BASE_DIR,
    format_date_label,
    read_ndvi_metadata,
    resolve_ndvi_path_for_date,
    scan_ndvi_inventory,
)
from infrawatch.core.score import build_trend_rows, score_feature_collection

LOGGER = logging.getLogger("fast_app")
logging.basicConfig(level=logging.INFO)

APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))

app = FastAPI(title="InfraWatch Fast App", version="0.1.0")

app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")


def _get_base_dir() -> Path:
    env_value = os.getenv("INFRAWATCH_NDVI_BASE_DIR")
    if env_value:
        return Path(env_value)
    return DEFAULT_NDVI_BASE_DIR


def _prepare_dates(requested_dates: list[str], inventory_dates: list[str]) -> list[str]:
    if requested_dates:
        return requested_dates
    if inventory_dates:
        latest = format_date_label(inventory_dates[0])
        return [latest]
    return []


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/ndvi/dates")
async def list_ndvi_dates() -> dict[str, Any]:
    base_dir = _get_base_dir()
    inventory = scan_ndvi_inventory(base_dir)
    dates = [format_date_label(date_key) for date_key in inventory.dates_sorted]
    latest_bounds = None
    if inventory.dates_sorted:
        latest_key = inventory.dates_sorted[0]
        latest_path = inventory.inventory.get(latest_key)
        if latest_path:
            _, _, bounds = read_ndvi_metadata(latest_path)
            latest_bounds = bounds
    return {
        "dates": dates,
        "base_dir": str(base_dir),
        "warnings": inventory.warnings,
        "latest_bounds": latest_bounds,
    }


@app.post("/api/score")
async def score_segments(request: ScoreRequest) -> JSONResponse:
    warnings: list[str] = []
    try:
        feature_collection, geo_warnings = validate_line_feature_collection(
            request.feature_collection
        )
        warnings.extend(geo_warnings)

        if not feature_collection.get("features"):
            warnings.append("No valid line features provided.")
            return JSONResponse(
                status_code=200,
                content={
                    "segments": [],
                    "trend": [],
                    "meta": {
                        "ndvi_path": None,
                        "ndvi_crs": None,
                        "line_count": 0,
                        "dates_scored": [],
                    },
                    "warnings": warnings,
                },
            )

        base_dir = _get_base_dir()
        inventory = scan_ndvi_inventory(base_dir)
        dates_to_score = _prepare_dates(request.dates, inventory.dates_sorted)
        ndvi_paths: dict[str, Path | None] = {}

        if not dates_to_score:
            warnings.append("No NDVI dates available; returning NO_DATA rows.")
        for date_label in dates_to_score:
            resolved, warning = resolve_ndvi_path_for_date(date_label, inventory)
            if warning:
                warnings.append(warning)
            ndvi_paths[date_label] = resolved

        if not ndvi_paths:
            ndvi_paths = {"NO_DATE": None}

        primary_date = next(iter(ndvi_paths.keys()))
        segments, meta = score_feature_collection(
            feature_collection,
            ndvi_paths[primary_date],
            buffer_m=request.buffer_m,
            ndvi_threshold=request.ndvi_threshold,
        )
        trend_rows = build_trend_rows(
            feature_collection,
            ndvi_paths,
            buffer_m=request.buffer_m,
            ndvi_threshold=request.ndvi_threshold,
        )

        response = {
            "segments": segments,
            "trend": trend_rows,
            "meta": {
                "ndvi_path": meta.ndvi_path,
                "ndvi_crs": meta.ndvi_crs,
                "line_count": meta.line_count,
                "dates_scored": list(ndvi_paths.keys()),
            },
            "warnings": warnings,
        }
        return JSONResponse(status_code=200, content=response)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Score request failed")
        return JSONResponse(
            status_code=500,
            content={
                "segments": [],
                "trend": [],
                "meta": {
                    "ndvi_path": None,
                    "ndvi_crs": None,
                    "line_count": 0,
                    "dates_scored": [],
                },
                "warnings": warnings + ["Score request failed."],
                "error": str(exc),
                "trace": "\n".join(traceback.format_exc().splitlines()[-6:]),
            },
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    _ = request
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
