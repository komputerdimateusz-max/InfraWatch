"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from infrawatch.core.score import NDVI_THRESHOLD_DEFAULT


class ScoreRequest(BaseModel):
    feature_collection: dict[str, Any] = Field(..., description="GeoJSON FeatureCollection")
    dates: list[str] = Field(default_factory=list)
    buffer_m: float = Field(default=20.0, ge=0.0)
    ndvi_threshold: float = Field(default=NDVI_THRESHOLD_DEFAULT, ge=0.0, le=1.0)


class SegmentScore(BaseModel):
    segment_id: int
    mean_ndvi: float | None
    p90_ndvi: float | None
    pct_above_0_6: float | None
    risk_score: float | None
    risk_category: str | None
    data_status: str


class TrendRow(BaseModel):
    date: str
    segment_id: int
    mean_ndvi: float | None
    risk_score: float | None
    risk_category: str | None
    data_status: str


class ScoreMeta(BaseModel):
    ndvi_path: str | None
    ndvi_crs: str | None
    line_count: int
    dates_scored: list[str]


class ScoreResponse(BaseModel):
    segments: list[SegmentScore]
    trend: list[TrendRow]
    meta: ScoreMeta
    warnings: list[str]
