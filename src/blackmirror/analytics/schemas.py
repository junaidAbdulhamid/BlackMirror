"""Small, strongly typed metadata contracts for derived neural analytics.

Large time-series remain in NPZ artifacts; JSON contains descriptions, statistics,
and paths.  This preserves the Phase 1 binary-array contract.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Hemisphere(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class CorticalRegion(BaseModel):
    model_config = ConfigDict(frozen=True)

    region_id: int = Field(ge=0)
    atlas_label: int = Field(ge=0)
    name: str
    hemisphere: Hemisphere
    atlas: str
    atlas_version: str
    vertex_count: int = Field(ge=1)
    network_name: str | None = None


class AtlasMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    surface_space: str
    source: str
    license: str
    region_count: int = Field(ge=1)
    regions: tuple[CorticalRegion, ...]


class AtlasValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    prediction_vertices: int = Field(ge=0)
    mesh_vertices: int = Field(ge=0)
    atlas_vertices: int = Field(ge=0)
    mapped_vertices: int = Field(ge=0)
    medial_wall_vertices: int = Field(ge=0)
    unknown_vertices: int = Field(ge=0)
    invalid_vertices: int = Field(ge=0)
    mapped_fraction: float = Field(ge=0, le=1)
    hemisphere_counts: dict[str, int]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class ROIResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    region: CorticalRegion
    mean_response: float
    peak_response: float
    peak_timestep: int = Field(ge=0)
    peak_time_seconds: float
    std_response: float = Field(ge=0)
    min_response: float
    max_response: float


class NeuralEventType(StrEnum):
    RESPONSE_PEAK = "response_peak"
    LARGE_TRANSITION = "large_transition"
    REGIONAL_PEAK = "regional_peak"


class NeuralEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    timestep: int = Field(ge=0)
    timestamp_seconds: float
    event_type: NeuralEventType
    score: float = Field(ge=0)
    affected_region_ids: tuple[int, ...] = ()
    global_response: float
    change_magnitude: float = Field(ge=0)


class ArrayArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    keys: dict[str, str]


class AnalyticsMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    analytics_version: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    aggregation: str
    parameters: dict[str, float | int | str | bool | None]
    interpretation_notice: str = (
        "Derived mathematical summaries of predicted cortical BOLD responses; they do not "
        "establish attention, emotion, memory, intent, or any psychological state."
    )


class NeuralAnalyticsResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    run_id: str
    atlas: AtlasMetadata
    validation: AtlasValidationReport
    roi_responses: tuple[ROIResponse, ...]
    events: tuple[NeuralEvent, ...]
    arrays: dict[str, ArrayArtifact]
    metadata: AnalyticsMetadata
