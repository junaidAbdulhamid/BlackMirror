"""Typed, versioned metadata for controlled neural comparisons."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComparisonModel(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)


class ComparabilitySeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ComparabilityIssue(ComparisonModel):
    field: str
    severity: ComparabilitySeverity
    message: str


class ComparabilityReport(ComparisonModel):
    comparable: bool
    reference_run_id: str
    candidate_run_id: str
    checks: dict[str, bool]
    issues: tuple[ComparabilityIssue, ...] = ()


class TimelineAlignment(ComparisonModel):
    method: str = "exact_observed_intersection"
    tolerance_seconds: float = Field(ge=0)
    reference_samples: int = Field(ge=1)
    candidate_samples: int = Field(ge=1)
    aligned_samples: int = Field(ge=1)
    reference_coverage_fraction: float = Field(ge=0, le=1)
    candidate_coverage_fraction: float = Field(ge=0, le=1)
    first_time_seconds: float
    last_time_seconds: float
    interpolation_applied: bool = False
    interpolated_candidate_samples: int = Field(default=0, ge=0)
    extrapolated_samples: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_alignment(self) -> TimelineAlignment:
        if self.aligned_samples > self.reference_samples:
            raise ValueError("aligned sample count exceeds reference timeline")
        if self.extrapolated_samples:
            raise ValueError("timeline extrapolation is forbidden")
        if self.last_time_seconds < self.first_time_seconds:
            raise ValueError("alignment time bounds are reversed")
        return self


class RegionDifferenceSummary(ComparisonModel):
    region_id: int = Field(ge=0)
    name: str
    hemisphere: str
    mean_signed_delta: float
    mean_absolute_delta: float = Field(ge=0)
    rms_delta: float = Field(ge=0)
    peak_absolute_delta: float = Field(ge=0)
    peak_signed_delta: float
    peak_time_seconds: float


class NetworkDifferenceSummary(ComparisonModel):
    network_id: int = Field(ge=0)
    name: str
    mean_signed_delta: float
    mean_absolute_delta: float = Field(ge=0)
    rms_delta: float = Field(ge=0)
    peak_absolute_delta: float = Field(ge=0)
    peak_signed_delta: float
    peak_time_seconds: float


class ContentFeatureDifferenceSummary(ComparisonModel):
    name: str
    jointly_available_fraction: float = Field(ge=0, le=1)
    mean_signed_delta: float | None = None
    mean_absolute_delta: float | None = Field(default=None, ge=0)


class ContentComparisonIssue(ComparisonModel):
    code: str
    message: str
    reference_run_id: str
    candidate_run_id: str
    source: str = "phase4_content_analysis"


class DivergenceEvent(ComparisonModel):
    rank: int = Field(ge=1)
    aligned_index: int = Field(ge=0)
    timestamp_seconds: float
    score: float = Field(ge=0)
    cortical_l2_difference: float = Field(ge=0)
    cortical_cosine_similarity: float | None
    top_region_ids: tuple[int, ...] = ()


class DivergenceWindow(ComparisonModel):
    rank: int = Field(ge=1)
    start_time_seconds: float
    end_time_seconds: float
    observed_samples: int = Field(ge=1)
    mean_l2_difference: float = Field(ge=0)
    peak_l2_difference: float = Field(ge=0)
    peak_time_seconds: float

    @model_validator(mode="after")
    def validate_bounds(self) -> DivergenceWindow:
        if not self.start_time_seconds <= self.peak_time_seconds <= self.end_time_seconds:
            raise ValueError("divergence window peak is outside its bounds")
        return self


class ComparisonMetadata(ComparisonModel):
    comparison_version: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    reference_run_id: str
    candidate_run_ids: tuple[str, ...]
    stimulus_sha256: dict[str, str]
    model_fingerprint: str
    analytics_version: str
    atlas_name: str
    atlas_version: str
    functional_atlas_name: str
    functional_atlas_version: str
    functional_atlas_mapping_sha256: str
    functional_atlas_source_url: str
    parameters: dict[str, float | int | str | bool]
    interpretation_notice: str = (
        "Differences between model-predicted cortical responses under controlled pipeline "
        "conditions. They are not causal effects or evidence of psychological states."
    )


class PairwiseComparison(ComparisonModel):
    candidate_run_id: str
    comparability: ComparabilityReport
    alignment: TimelineAlignment
    regions: tuple[RegionDifferenceSummary, ...]
    networks: tuple[NetworkDifferenceSummary, ...] = ()
    content_status: str = "unavailable"
    content_issue: ContentComparisonIssue | None = None
    content_analysis_version: str | None = None
    content_models: dict[str, str] = Field(default_factory=dict)
    content_configuration: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    content_features: tuple[ContentFeatureDifferenceSummary, ...] = ()
    events: tuple[DivergenceEvent, ...]
    windows: tuple[DivergenceWindow, ...]
    arrays_path: str
    array_keys: dict[str, str]


class NeuralComparisonResult(ComparisonModel):
    schema_version: str = "1.1"
    comparison_id: str
    metadata: ComparisonMetadata
    pairs: tuple[PairwiseComparison, ...]

    @model_validator(mode="after")
    def validate_run_membership(self) -> NeuralComparisonResult:
        candidate_ids = tuple(pair.candidate_run_id for pair in self.pairs)
        if not candidate_ids or candidate_ids != self.metadata.candidate_run_ids:
            raise ValueError("pair ordering must exactly match metadata candidate run ids")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate run ids must be unique")
        expected_hashes = {self.metadata.reference_run_id, *candidate_ids}
        if set(self.metadata.stimulus_sha256) != expected_hashes:
            raise ValueError("stimulus hashes must exactly match compared run ids")
        for pair in self.pairs:
            if pair.comparability.reference_run_id != self.metadata.reference_run_id:
                raise ValueError("pair comparability reference does not match metadata")
            if pair.comparability.candidate_run_id != pair.candidate_run_id:
                raise ValueError("pair comparability candidate does not match pair metadata")
        return self
