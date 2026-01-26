from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping

from pydantic import BaseModel, Field, field_validator, model_validator

from .paths import validate_bbox


class IngestionRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    date_from: date = Field(alias="dateFrom")
    date_to: date = Field(alias="dateTo")
    max_scenes: int = Field(alias="maxScenes", ge=1)
    max_cloud_cover: float | None = Field(default=None, alias="maxCloudCover", ge=0, le=100)
    dry_run: bool = Field(default=False, alias="dryRun")

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return validate_bbox(value)

    @model_validator(mode="after")
    def _validate_dates(self) -> "IngestionRequest":
        if self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to.")
        return self


class SceneSummary(BaseModel):
    product_id: str
    title: str
    acquisition_datetime: datetime
    cloud_cover: float | None
    bbox: tuple[float, float, float, float]
    footprint: Mapping[str, Any] | None
    download_url: str


class SceneMetadata(BaseModel):
    product_id: str
    title: str
    acquisition_datetime: datetime
    cloud_cover: float | None
    bbox: tuple[float, float, float, float]
    footprint: Mapping[str, Any] | None
    source_endpoint: str
    source_query: Mapping[str, Any]
