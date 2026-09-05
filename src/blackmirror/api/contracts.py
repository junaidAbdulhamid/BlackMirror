"""Visualization data contracts.

The narrow, JSON-safe view of a Phase 1 run that a renderer needs. These are
deliberately *derived* from `PredictionResult` rather than replacing it: the
Phase 1 contract stays authoritative, and this layer decides only what a browser
should be told.

Rule: no numeric arrays here. Arrays travel as raw binary over dedicated
endpoints; this module carries the metadata needed to interpret them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HemisphereRange(BaseModel):
    """Where one hemisphere lives on the prediction's vertex axis."""

    model_config = ConfigDict(frozen=True)

    name: str
    start: int = Field(ge=0, description="Inclusive first column.")
    end: int = Field(ge=0, description="Exclusive last column.")

    @property
    def count(self) -> int:
        return self.end - self.start


class StimulusSummary(BaseModel):
    """What was shown, and whether the browser can play it back."""

    model_config = ConfigDict(frozen=True)

    filename: str
    media_type: Literal["video", "audio", "text"]
    duration_seconds: float | None
    sha256: str
    playable: bool = Field(
        description="Whether the API can stream the source media. False when the "
        "original file is no longer at its recorded path."
    )


class ModelSummary(BaseModel):
    """Provenance the UI must surface so predictions are never misread."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    name: str
    backend: str
    model_id: str
    license: str | None
    device: str
    dtype: str
    subject_conditioning: str | None
    is_synthetic: bool = Field(
        description="True for the mock backend. The UI MUST label such runs as "
        "synthetic test data, never as a brain prediction."
    )
    feature_extractors: dict[str, str] = Field(default_factory=dict)


class TemporalSummary(BaseModel):
    """Everything needed to map playback time onto a prediction row.

    `stimulus_time_seconds` (in temporal.npz) is the playback clock. Row index is
    NOT `time / tr_seconds`: TRIBE drops event-free segments, so the timeline can
    be sparse and non-uniform.
    """

    model_config = ConfigDict(frozen=True)

    n_time_points: int
    tr_seconds: float
    timeline_is_contiguous: bool
    segments_were_filtered: bool
    n_segments_total: int | None
    n_segments_kept: int | None
    first_segment_start_seconds: float | None
    last_segment_end_seconds: float | None
    covered_seconds: float | None
    hemodynamic_offset_seconds: float
    hemodynamic_offset_verified: bool
    output_is_stimulus_aligned: bool
    notes: tuple[str, ...] = ()


class CorticalSummary(BaseModel):
    """How prediction columns map onto the cortical surface."""

    model_config = ConfigDict(frozen=True)

    surface_space: str
    vertex_count: int
    vertices_per_hemisphere: int
    hemispheres: tuple[HemisphereRange, ...]
    medial_wall_available: bool
    mapping_notes: str | None


class ResponseScale(BaseModel):
    """Statistics a client uses to build a colour scale.

    Supplied by the API so every client normalises identically, and so a Phase 5
    A/B comparison can share one scale across two runs.
    """

    model_config = ConfigDict(frozen=True)

    min: float
    max: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    abs_max: float = Field(
        description="max(|p01|, |p99|). The symmetric half-range for a "
        "zero-centred diverging scale."
    )
    negative_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Share of values below zero. Above ~0.15 a diverging scale "
        "centred on zero is the scientifically appropriate choice.",
    )
    diverging_recommended: bool


class ArrayDescriptor(BaseModel):
    """Describes one binary endpoint so the client can type its ArrayBuffer."""

    model_config = ConfigDict(frozen=True)

    url: str
    dtype: Literal["float32", "int32", "uint8"]
    shape: tuple[int, ...]
    byte_length: int
    description: str


class RunSummary(BaseModel):
    """The single document a visualization client loads first."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    run_id: str
    created_at: str
    status: str
    schema_version: str
    stimulus: StimulusSummary
    model: ModelSummary
    temporal: TemporalSummary
    cortical: CorticalSummary
    scale: ResponseScale
    arrays: dict[str, ArrayDescriptor]
    interpretation_notice: str


class RunListItem(BaseModel):
    """One row in the run index."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    run_id: str
    created_at: str
    stimulus_filename: str
    media_type: str
    backend: str
    is_synthetic: bool
    shape: tuple[int, ...]
    status: str
