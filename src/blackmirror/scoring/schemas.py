"""The Phase 6 objective contract.

WHAT THIS IS
    A `NeuralObjective` is a machine-readable statement of what "better" means,
    written down before any variant is scored. Phase 5 answers "are A and B
    different?". This answers "given goal G, what did A and B score?".

WHY THE SEPARATION MATTERS
    Three quantities are kept distinct on every evaluation and never collapsed:

        raw_value        the scientifically meaningful number, in model units
        normalized_value the raw value placed on a comparable 0-1 scale
        score            the normalized value after direction is applied

    Collapsing them is how scoring engines start lying. A raw ROI response of
    0.401 vs 0.402 is a difference of 0.001; min-max normalisation across only
    those two variants reports 0.0 vs 1.0, which looks like a total victory.
    Both numbers are "correct" and only one is honest, so both are persisted
    and the raw value is always available beside the score.

WHAT A SCORE IS NOT
    A score is the value of an explicitly chosen mathematical function of a
    predicted response. It is not persuasiveness, memorability, attention, or
    conversion. Ranking says "variant B scored higher on the objective you
    defined", never "variant B is better".
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Bumped when a change would alter a score for identical inputs.
SCORING_VERSION = "1.1"


class ScoringModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


# ---------------------------------------------------------------------------
# What the objective measures against
# ---------------------------------------------------------------------------


class TargetType(StrEnum):
    """Where on the cortex an objective is measured.

    Every member here is a *spatial* selection with an explicit vertex basis.
    Psychological targets are deliberately absent: there is no verified mapping
    from this model's output to a construct like "attention", so offering one
    would be fabrication with a type annotation on it.
    """

    WHOLE_CORTEX = "whole_cortex"
    ROI = "roi"
    NETWORK = "network"
    HEMISPHERE = "hemisphere"
    CUSTOM_VERTEX_SET = "custom_vertex_set"


class ObjectiveTarget(ScoringModel):
    """The spatial selection. Exactly one identifier applies per target type."""

    type: TargetType
    #: Destrieux region_id for ROI targets.
    region_id: int | None = Field(default=None, ge=0)
    #: Yeo network index (0-6) for NETWORK targets.
    network_id: int | None = Field(default=None, ge=0, le=6)
    hemisphere: Literal["left", "right"] | None = None
    #: Explicit vertex indices for CUSTOM_VERTEX_SET. Persisted in full so a
    #: score can be reproduced without re-deriving the selection.
    vertex_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _identifier_matches_type(self) -> ObjectiveTarget:
        required = {
            TargetType.ROI: self.region_id is not None,
            TargetType.NETWORK: self.network_id is not None,
            TargetType.HEMISPHERE: self.hemisphere is not None,
            TargetType.CUSTOM_VERTEX_SET: bool(self.vertex_indices),
            TargetType.WHOLE_CORTEX: True,
        }
        if not required[self.type]:
            raise ValueError(f"target type {self.type.value} is missing its identifier")
        if self.type is TargetType.CUSTOM_VERTEX_SET and len(set(self.vertex_indices)) != len(
            self.vertex_indices
        ):
            raise ValueError("custom vertex set contains duplicates")
        supplied = {
            "region_id": self.region_id is not None,
            "network_id": self.network_id is not None,
            "hemisphere": self.hemisphere is not None,
            "vertex_indices": bool(self.vertex_indices),
        }
        expected = {
            TargetType.ROI: "region_id",
            TargetType.NETWORK: "network_id",
            TargetType.HEMISPHERE: "hemisphere",
            TargetType.CUSTOM_VERTEX_SET: "vertex_indices",
            TargetType.WHOLE_CORTEX: None,
        }[self.type]
        irrelevant = [name for name, present in supplied.items() if present and name != expected]
        if irrelevant:
            raise ValueError(
                f"target type {self.type.value} has irrelevant identifiers: {irrelevant}"
            )
        return self


# ---------------------------------------------------------------------------
# When the objective is measured
# ---------------------------------------------------------------------------


class TemporalScopeType(StrEnum):
    """Which stimulus interval the metric is computed over.

    CONTENT_EVENT is the one that makes cross-variant comparison meaningful:
    two variants whose CTA sits at different timestamps are each measured over
    *their own* CTA, not over a shared absolute window that would compare a CTA
    against whatever the other variant happened to be showing.
    """

    FULL_STIMULUS = "full_stimulus"
    ABSOLUTE_TIME_WINDOW = "absolute_time_window"
    #: Fractions of the stimulus duration, for variants of differing length.
    NORMALIZED_TIME_WINDOW = "normalized_time_window"
    CONTENT_EVENT = "content_event"
    CONTENT_EVENT_RELATIVE_WINDOW = "content_event_relative_window"


class TemporalScope(ScoringModel):
    type: TemporalScopeType
    start_seconds: float | None = None
    end_seconds: float | None = None
    #: Fractions in [0, 1] for NORMALIZED_TIME_WINDOW.
    start_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    end_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    #: A ContentEventType value. Not an enum import, to keep this module free of
    #: a Phase 4 dependency; the resolver validates it against the real set.
    event_type: str | None = None
    #: Which occurrence to use when a variant has several. "first" and "last"
    #: are deterministic; "longest" breaks ties by earliest start.
    event_selection: Literal["first", "last", "longest"] = "first"
    #: Consider only occurrences that actually contain a prediction sample.
    #:
    #: Phase 4 detects events at sub-second resolution while predictions are
    #: sampled at the TR (1.0 s on current runs), so an event shorter than one
    #: TR can contain no sample at all. Measured on a real run: the first
    #: speech_segment spans [0.031, 0.500) and holds zero samples, which made
    #: the whole objective unmeasurable for that variant even though a 1.9 s
    #: speech segment followed immediately.
    #:
    #: This is OFF by default because skipping occurrences changes which event
    #: "first" refers to, and that must be a stated choice rather than a
    #: silent one. When enabled, the number skipped is recorded on the window.
    skip_events_without_samples: bool = False
    #: Offsets around the event for CONTENT_EVENT_RELATIVE_WINDOW, e.g. -2.0 to
    #: +3.0 seconds around a product reveal.
    offset_start_seconds: float | None = None
    offset_end_seconds: float | None = None

    @model_validator(mode="after")
    def _fields_match_type(self) -> TemporalScope:
        if self.type is TemporalScopeType.ABSOLUTE_TIME_WINDOW:
            if self.start_seconds is None or self.end_seconds is None:
                raise ValueError("absolute window needs start_seconds and end_seconds")
            if not self.end_seconds > self.start_seconds:
                raise ValueError("absolute window must have end_seconds > start_seconds")
        if self.type is TemporalScopeType.NORMALIZED_TIME_WINDOW:
            if self.start_fraction is None or self.end_fraction is None:
                raise ValueError("normalized window needs start_fraction and end_fraction")
            if not self.end_fraction > self.start_fraction:
                raise ValueError("normalized window must have end_fraction > start_fraction")
        if self.type in (
            TemporalScopeType.CONTENT_EVENT,
            TemporalScopeType.CONTENT_EVENT_RELATIVE_WINDOW,
        ) and not self.event_type:
            raise ValueError("content-event scope needs event_type")
        if self.type is TemporalScopeType.CONTENT_EVENT_RELATIVE_WINDOW:
            if self.offset_start_seconds is None or self.offset_end_seconds is None:
                raise ValueError("relative window needs both offsets")
            if not self.offset_end_seconds > self.offset_start_seconds:
                raise ValueError("relative window offsets must increase")
        return self


# ---------------------------------------------------------------------------
# Direction and normalization
# ---------------------------------------------------------------------------


class ObjectiveDirection(StrEnum):
    """How a raw value maps to preference.

    TARGET exists because "more response is better" is an assumption, not a
    finding. An objective can legitimately want a specific level, and scoring
    then measures distance from it rather than magnitude.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class NormalizationStrategy(StrEnum):
    NONE = "none"
    MIN_MAX_WITHIN_EXPERIMENT = "min_max_within_experiment"
    Z_SCORE_WITHIN_EXPERIMENT = "z_score_within_experiment"
    REFERENCE_BASELINE = "reference_baseline"
    ROBUST_PERCENTILE = "robust_percentile"
    TARGET_DISTANCE = "target_distance"


class ObjectiveNormalization(ScoringModel):
    strategy: NormalizationStrategy = NormalizationStrategy.NONE
    #: Required by TARGET direction and TARGET_DISTANCE normalization.
    target_value: float | None = None
    #: Scale over which distance from target decays to zero score.
    target_tolerance: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _target_needs_a_value(self) -> ObjectiveNormalization:
        if self.strategy is NormalizationStrategy.TARGET_DISTANCE and (
            self.target_value is None or self.target_tolerance is None
        ):
            raise ValueError("target-distance normalization needs value and tolerance")
        return self


class ObjectiveProvenance(ScoringModel):
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    created_by: str | None = None
    note: str | None = None
    scoring_version: str = SCORING_VERSION


class NeuralObjective(ScoringModel):
    """One explicitly defined, measurable goal."""

    objective_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1)
    description: str | None = None
    #: A registered metric name; validated against the registry, not an enum,
    #: so new validated metrics can be added without a schema change.
    metric: str
    target: ObjectiveTarget
    temporal_scope: TemporalScope
    direction: ObjectiveDirection
    normalization: ObjectiveNormalization = ObjectiveNormalization()
    weight: float = Field(default=1.0, ge=0.0)
    parameters: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    provenance: ObjectiveProvenance = ObjectiveProvenance()

    @model_validator(mode="after")
    def _target_direction_needs_a_value(self) -> NeuralObjective:
        if self.direction is ObjectiveDirection.TARGET:
            if self.normalization.target_value is None:
                raise ValueError("TARGET direction requires normalization.target_value")
            if self.normalization.target_tolerance is None:
                raise ValueError(
                    "TARGET direction requires normalization.target_tolerance, which sets "
                    "the scale over which distance from the target stops mattering"
                )
        return self


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class ResolvedWindow(ScoringModel):
    """The interval an objective actually measured, per variant.

    Persisted rather than recomputed because a content-event scope resolves to a
    different interval in every variant, and a score is unreadable without
    knowing which seconds produced it.
    """

    start_seconds: float
    end_seconds: float
    #: Indices into the variant's own analytics time axis.
    sample_indices: tuple[int, ...]
    sample_count: int = Field(ge=0)
    source: str = Field(description="How the window was derived, e.g. 'content_event:cta[first]'.")

    @model_validator(mode="after")
    def _consistent(self) -> ResolvedWindow:
        if self.sample_count != len(self.sample_indices):
            raise ValueError("sample_count must match sample_indices")
        if self.end_seconds < self.start_seconds:
            raise ValueError("window end precedes its start")
        return self


class TargetResolution(ScoringModel):
    """Which vertices or series the target selected, and how many."""

    type: TargetType
    label: str
    vertex_count: int = Field(ge=0)
    region_id: int | None = None
    network_id: int | None = None
    hemisphere: str | None = None
    atlas_name: str | None = None
    atlas_version: str | None = None
    mapping_sha256: str | None = None


class TemporalContribution(ScoringModel):
    """Which parts of the window produced the raw value.

    Entries sum to the raw metric value by construction. They do not decompose
    the normalized, direction-adjusted, or composite score.
    Absent when the metric is not a sum over samples.
    """

    times: tuple[float, ...]
    contributions: tuple[float, ...]
    #: Index into `times` of the largest single contribution.
    peak_index: int = Field(ge=0)
    peak_time_seconds: float
    peak_fraction: float | None = Field(
        default=None,
        description="Largest contribution as a share of total magnitude.",
    )

    @model_validator(mode="after")
    def _aligned(self) -> TemporalContribution:
        if len(self.times) != len(self.contributions):
            raise ValueError("temporal contribution times and values must align")
        if not self.times:
            raise ValueError("temporal contribution cannot be empty")
        return self


class RegionalContribution(ScoringModel):
    """Which regions produced an aggregate target's value.

    Only meaningful for targets that span many vertices -- whole cortex, a
    hemisphere, a custom vertex set. An ROI target is already one region, so
    decomposing it would restate the question.

    Shares are of the aggregate's *magnitude*, so a region contributing
    strongly in the negative direction is visible rather than cancelling
    silently against a positive one.
    """

    region_ids: tuple[int, ...]
    region_names: tuple[str, ...]
    #: Mean response of each region over the window, in model units.
    values: tuple[float, ...]
    #: Each region's share of total magnitude, summing to 1.
    shares: tuple[float, ...]
    #: Vertices behind each region within the target, so a large share from a
    #: two-vertex region is not mistaken for a large effect.
    vertex_counts: tuple[int, ...]
    note: str = (
        "Contribution to the aggregate metric, not evidence that a region is "
        "functionally responsible for anything."
    )

    @model_validator(mode="after")
    def _aligned(self) -> RegionalContribution:
        lengths = {
            len(self.region_ids), len(self.region_names),
            len(self.values), len(self.shares), len(self.vertex_counts),
        }
        if len(lengths) != 1:
            raise ValueError("regional contribution fields must be the same length")
        return self


class ObjectiveEvaluation(ScoringModel):
    """One objective, evaluated against one variant.

    `raw_value` is the number the metric produced in model units.
    `normalized_value` places it on a comparable scale; it is None until an
    experiment-level normalizer has seen every variant, because strategies like
    min-max are undefined for a single variant by construction.
    `score` is the normalized value after direction is applied, and is the only
    quantity that may be ranked.
    """

    objective_id: str
    variant_id: str
    raw_value: float | None
    normalized_value: float | None = None
    score: float | None = None
    window: ResolvedWindow
    target: TargetResolution
    statistics: dict[str, float] = Field(
        default_factory=dict,
        description="Supporting numbers: sample count, std, min, max, finite count.",
    )
    temporal_contribution: TemporalContribution | None = None
    regional_contribution: RegionalContribution | None = None
    valid: bool = True
    warnings: tuple[str, ...] = ()


class ObjectiveContribution(ScoringModel):
    """How much one objective moved a composite score."""

    objective_id: str
    weight: float = Field(ge=0)
    score: float
    contribution: float = Field(description="weight x score, i.e. the additive share.")
    contribution_fraction: float | None = Field(
        default=None, description="Share of the composite total, when the total is nonzero."
    )


class VariantScore(ScoringModel):
    variant_id: str
    total_score: float | None
    objective_scores: tuple[ObjectiveEvaluation, ...]
    contributions: tuple[ObjectiveContribution, ...] = ()
    baseline_delta: float | None = None
    baseline_relative_delta: float | None = Field(
        default=None,
        description=(
            "Fractional change against the baseline total. None when the baseline "
            "total is at or near zero, where a percentage is not defined."
        ),
    )
    rank: int | None = Field(default=None, ge=1)
    valid: bool = True
    warnings: tuple[str, ...] = ()


class ParetoAnalysis(ScoringModel):
    """Which variants are not beaten on every objective at once.

    A variant is dominated when another is at least as good on every objective
    and strictly better on at least one. The non-dominated set is the honest
    answer to "which variants are worth considering" when objectives conflict,
    because it needs no weights and therefore embeds no opinion about tradeoffs.
    """

    objective_ids: tuple[str, ...]
    non_dominated: tuple[str, ...]
    dominated: tuple[str, ...]
    #: variant -> the variants that dominate it.
    dominated_by: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class WeightSensitivityPoint(ScoringModel):
    weight: float
    winner_variant_id: str
    margin: float


class SensitivityAnalysis(ScoringModel):
    """How the ranking responds to the weights it was given.

    A ranking that flips within the range of weights a user might plausibly
    have chosen is a weight-dependent ranking, not a finding about the variants.
    """

    swept_objective_id: str
    points: tuple[WeightSensitivityPoint, ...]
    #: Weights at which the top-ranked variant changes.
    flip_weights: tuple[float, ...] = ()
    stable: bool = True
    note: str | None = None


class ScoringMetadata(ScoringModel):
    scoring_version: str = SCORING_VERSION
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    analytics_versions: dict[str, str] = Field(default_factory=dict)
    atlas_name: str | None = None
    atlas_version: str | None = None
    network_mapping_sha256: str | None = None
    objective_set_hash: str
    objective_definition_hash: str
    input_artifact_sha256: dict[str, str] = Field(default_factory=dict)
    #: Every registered metric's formula, so a stored score can be re-read later.
    metric_descriptions: dict[str, dict[str, str]] = Field(default_factory=dict)
    interpretation_notice: str = (
        "Scores are values of explicitly chosen mathematical functions of predicted "
        "cortical responses. A higher score means the variant scored higher on the "
        "objective that was defined, not that it is more persuasive, memorable, "
        "attention-grabbing, or effective."
    )


class ExperimentScoreResult(ScoringModel):
    schema_version: str = "1.1"
    experiment_id: str
    objectives: tuple[NeuralObjective, ...]
    baseline_variant_id: str | None = None
    variant_scores: tuple[VariantScore, ...]
    ranking: tuple[str, ...] = ()
    ranking_margin: float | None = Field(
        default=None, description="Top score minus runner-up score, when both exist."
    )
    pareto: ParetoAnalysis | None = None
    sensitivity: tuple[SensitivityAnalysis, ...] = ()
    normalization_detail: dict[str, dict[str, float]] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "The share of composite-score magnitude each objective actually accounted "
            "for, averaged over variants. A stated weight only becomes the real weight "
            "when the objectives are on comparable scales; otherwise the objective with "
            "the larger units dominates regardless of what the weights say. Compare "
            "these against the stated weights before believing a composite."
        ),
    )
    metadata: ScoringMetadata
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class IntervalContext(ScoringModel):
    """What the content was doing during an interval, from Phase 4.

    Every field is copied from a content analysis. Nothing here is inferred, and
    the presence of a description does not mean it explains anything.
    """

    start_seconds: float
    end_seconds: float
    visual_description: str | None = None
    speech_text: str | None = None
    audio_description: str | None = None
    on_screen_text: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    event_types: tuple[str, ...] = ()


class PairedEffectSummary(ScoringModel):
    """A descriptive paired effect, or a typed explanation for withholding it."""

    status: Literal["available", "unavailable"]
    method: str = "paired_standardized_mean_difference"
    value: float | None = None
    sample_count: int = Field(ge=0)
    reason: str | None = None
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _availability_is_consistent(self) -> PairedEffectSummary:
        if self.status == "available" and self.value is None:
            raise ValueError("an available paired effect needs a value")
        if self.status == "unavailable" and not self.reason:
            raise ValueError("an unavailable paired effect needs a reason")
        return self


class ObjectiveComparison(ScoringModel):
    """One objective, compared between two variants."""

    objective_id: str
    objective_name: str
    metric: str
    direction: ObjectiveDirection
    target_label: str
    leader_variant_id: str | None
    raw_values: dict[str, float | None]
    scores: dict[str, float | None]
    absolute_difference: float | None = None
    relative_difference: float | None = Field(
        default=None,
        description=(
            "Fractional difference against the lower value. None when that value is "
            "at or near zero, where a percentage is not defined."
        ),
    )
    windows: dict[str, ResolvedWindow] = Field(default_factory=dict)
    #: Where each variant's score was concentrated, when the metric decomposes.
    peak_intervals: dict[str, float] = Field(default_factory=dict)
    paired_effect: PairedEffectSummary


class ScoreExplanation(ScoringModel):
    """A structured, deterministic account of why one variant ranked above another.

    Built entirely from measured quantities. The `statements` are generated from
    those numbers by fixed templates, not by a language model, so nothing in an
    explanation can be present that was not computed.

    The co-occurrence boundary is enforced in the wording: content differences
    are reported as having occurred *during the same interval* as the neural
    difference, never as having caused it.
    """

    experiment_id: str
    objective_set_hash: str
    leader_variant_id: str | None
    runner_up_variant_id: str | None
    total_difference: float | None = None
    ranking_is_close: bool = False
    objectives: tuple[ObjectiveComparison, ...] = ()
    #: Content context at each variant's highest-contribution interval.
    context: dict[str, IntervalContext] = Field(default_factory=dict)
    statements: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
