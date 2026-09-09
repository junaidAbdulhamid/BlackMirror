"""Phase 7 contracts: the optimization request, its evidence, and its output.

WHAT PHASE 7 IS
    Phase 6 answers "how well did this variant satisfy objective G?". Phase 7
    answers "where did it fall short, what was on screen there, how did better
    variants differ, and what is worth testing next?".

WHAT PHASE 7 IS NOT
    It does not improve anything. Every output is a *hypothesis* with an
    attached evidence trail, and none of it is validated until Phase 8 builds
    the candidate, re-runs TRIBE and re-scores it. The schemas enforce that
    reading: `expected_direction` is `TEST_FOR_INCREASE`, never
    `WILL_INCREASE`, and confidence is named `evidence_confidence` because it
    measures how well-supported a *test* is, not how likely it is to succeed.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Bumped when a change would alter recommendations for identical inputs.
OPTIMIZATION_VERSION = "1.0"


class OptimizationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ReferenceStrategy(StrEnum):
    """How the variants to learn from are chosen.

    Never implicit. A recommendation's whole meaning depends on what it is
    being compared against, so the choice is recorded on the request and echoed
    on every piece of evidence derived from it.
    """

    BEST_SCORE = "best_score"
    BASELINE = "baseline"
    PARETO_FRONT = "pareto_front"
    MANUAL = "manual"


class OptimizationStrategy(StrEnum):
    """How far from the source variant a recommendation may travel.

    MINIMAL_EDIT prefers the smallest testable change, which is the most
    scientifically useful because it isolates one hypothesis. EXPLORATORY
    permits larger changes and is labelled as harder to attribute.
    """

    MINIMAL_EDIT = "minimal_edit"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


class EditCost(StrEnum):
    """Rough implementation effort, used in ranking rather than in evidence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(StrEnum):
    """Experimental/creative modification risk, not safety risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OptimizationConstraints(OptimizationModel):
    """What a recommendation is not allowed to touch.

    Constraints are checked by the validator and a violating recommendation is
    rejected rather than softened, because a constraint the user set is not a
    preference the optimizer may trade away.
    """

    #: Modalities that must not be modified at all.
    frozen_modalities: tuple[Literal["visual", "audio", "speech", "text"], ...] = ()
    #: Intervals that must be preserved exactly, e.g. a legal disclaimer.
    preserve_intervals: tuple[tuple[float, float], ...] = ()
    #: Maximum permitted change to total duration, in seconds.
    max_duration_change_seconds: float | None = Field(default=None, ge=0)
    #: Event types whose wording must not change.
    frozen_event_types: tuple[str, ...] = ()
    #: Free-text rules recorded for the human reviewer; not machine-enforced.
    notes: tuple[str, ...] = ()
    forbid_new_voiceover: bool = False

    @model_validator(mode="after")
    def _intervals_are_ordered(self) -> OptimizationConstraints:
        for start, end in self.preserve_intervals:
            if end <= start:
                raise ValueError("preserve interval must have end > start")
        return self


class OptimizationRequest(OptimizationModel):
    experiment_id: str = Field(min_length=1, max_length=128)
    source_variant_id: str = Field(min_length=1)
    #: Which stored Phase 6 score to optimise against.
    objective_set_hash: str = Field(min_length=8)
    baseline_variant_id: str | None = None
    #: Explicit references for ReferenceStrategy.MANUAL.
    comparison_variant_ids: tuple[str, ...] = ()
    reference_strategy: ReferenceStrategy = ReferenceStrategy.BEST_SCORE
    strategy: OptimizationStrategy = OptimizationStrategy.MINIMAL_EDIT
    constraints: OptimizationConstraints = OptimizationConstraints()
    max_recommendations: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _manual_needs_references(self) -> OptimizationRequest:
        if (
            self.reference_strategy is ReferenceStrategy.MANUAL
            and not self.comparison_variant_ids
        ):
            raise ValueError("manual reference selection needs comparison_variant_ids")
        if self.source_variant_id in self.comparison_variant_ids:
            raise ValueError("the source variant cannot also be a reference")
        return self


# ---------------------------------------------------------------------------
# Gap and intervals
# ---------------------------------------------------------------------------


class ObjectiveGap(OptimizationModel):
    """How far the source sits from the reference, in the objective's own terms.

    The formula depends on direction, which is why this is computed rather than
    assumed: maximizing wants reference minus current, minimizing wants the
    reverse, and a TARGET objective wants the change in distance from the
    target regardless of side.
    """

    objective_id: str
    direction: str
    source_value: float | None
    reference_value: float | None
    reference_variant_id: str | None
    #: Positive means the source has room to move toward the objective.
    gap: float | None
    #: Gap as a fraction of the reference magnitude, when that is defined.
    relative_gap: float | None = None
    formula: str
    note: str | None = None


class IntervalKind(StrEnum):
    WEAK = "weak"
    STRONG = "strong"


class OptimizationInterval(OptimizationModel):
    """One stretch of the source variant, judged relative to the objective.

    "Weak" is never an absolute statement about response magnitude. It means
    the interval contributed comparatively little to the objective *as defined*,
    which is why the direction and the contribution share are both recorded.
    """

    interval_id: str
    kind: IntervalKind
    objective_id: str
    start_seconds: float
    end_seconds: float
    sample_count: int = Field(ge=1)
    #: Share of the objective's total contribution magnitude in this interval.
    contribution_share: float
    #: Summed per-sample contribution over the interval, in raw units.
    contribution: float
    #: Contribution share of the same wall-clock interval in the reference.
    reference_contribution_share: float | None = None
    reason: str
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceKind(StrEnum):
    """What sort of measurement an evidence item records.

    Every kind but the last records something *observed* in existing variants.
    `SEARCH_SPACE_SAMPLE` is the exception and is kept visibly separate for that
    reason: a Phase 9 candidate's parameter was chosen by a strategy from a
    declared range, not measured anywhere. Filing that under a measured kind
    would make a sampled guess indistinguishable from a grounded finding, which
    is the one confusion this whole evidence model exists to prevent.
    """

    CONTENT_FEATURE_DELTA = "content_feature_delta"
    OBJECTIVE_GAP = "objective_gap"
    CONTRIBUTION_SHARE = "contribution_share"
    CONTENT_EVENT_PRESENCE = "content_event_presence"
    FEATURE_CONSISTENCY = "feature_consistency"
    REGIONAL_CONTRIBUTION = "regional_contribution"
    #: A parameter value a search strategy selected. Nothing was measured.
    SEARCH_SPACE_SAMPLE = "search_space_sample"


class Evidence(OptimizationModel):
    """One measured fact, addressable by id.

    Every recommendation must cite evidence ids. An item records what was
    measured, where, and in which variants -- never an interpretation. The
    `association` field states the epistemic status explicitly so that a reader
    cannot mistake a temporal comparison for a causal finding.
    """

    evidence_id: str
    kind: EvidenceKind
    description: str
    feature: str | None = None
    source_value: float | None = None
    reference_value: float | None = None
    delta: float | None = None
    interval_start_seconds: float | None = None
    interval_end_seconds: float | None = None
    source_variant_id: str | None = None
    reference_variant_ids: tuple[str, ...] = ()
    #: How many of the reference variants show this difference in the same
    #: direction. **None when fewer than two references were available**: a
    #: single reference agreeing with itself is not consistency, and reporting
    #: 1.0 there would make an anecdote indistinguishable from a pattern.
    consistency: float | None = Field(default=None, ge=0, le=1)
    #: How many references the difference was measured against. Always present,
    #: so a reader can see what `consistency` was (or was not) computed from.
    reference_count: int = Field(default=0, ge=0)
    sample_count: int | None = Field(default=None, ge=0)
    #: The epistemic status of this item. `not_measured` marks a value that was
    #: *chosen* rather than observed, which is what a search sample is; for
    #: those items `source_value` holds the setting applied and the
    #: measurement-shaped fields stay empty because no measurement exists.
    association: Literal[
        "temporal_comparison", "measured_value", "not_measured"
    ] = "temporal_comparison"
    note: str | None = None


# ---------------------------------------------------------------------------
# Interventions and recommendations
# ---------------------------------------------------------------------------


class InterventionType(StrEnum):
    """A deliberately small taxonomy.

    Hundreds of microtypes would make Phase 8 unable to group results. These
    eleven map onto the modalities Phase 4 actually measures, so every type has
    a measurable feature behind it.
    """

    TIMING_EDIT = "timing_edit"
    TEXT_EDIT = "text_edit"
    SPEECH_EDIT = "speech_edit"
    VISUAL_EDIT = "visual_edit"
    AUDIO_EDIT = "audio_edit"
    SHOT_EDIT = "shot_edit"
    PACE_EDIT = "pace_edit"
    CONTENT_ORDER_EDIT = "content_order_edit"
    CTA_EDIT = "cta_edit"
    PRODUCT_VISIBILITY_EDIT = "product_visibility_edit"
    SCENE_STRUCTURE_EDIT = "scene_structure_edit"


#: Which modality each intervention type touches, for constraint checking.
INTERVENTION_MODALITY: dict[InterventionType, str] = {
    InterventionType.TIMING_EDIT: "visual",
    InterventionType.TEXT_EDIT: "text",
    InterventionType.SPEECH_EDIT: "speech",
    InterventionType.VISUAL_EDIT: "visual",
    InterventionType.AUDIO_EDIT: "audio",
    InterventionType.SHOT_EDIT: "visual",
    InterventionType.PACE_EDIT: "visual",
    InterventionType.CONTENT_ORDER_EDIT: "visual",
    InterventionType.CTA_EDIT: "text",
    InterventionType.PRODUCT_VISIBILITY_EDIT: "visual",
    InterventionType.SCENE_STRUCTURE_EDIT: "visual",
}


class ExpectedDirection(StrEnum):
    """The only permitted claim about outcome.

    There is no WILL_INCREASE member, and that omission is the point: Phase 7
    cannot know the outcome, and a schema that could express certainty would
    eventually be used to express it.
    """

    TEST_FOR_INCREASE = "test_for_increase"
    TEST_FOR_DECREASE = "test_for_decrease"
    TEST_FOR_TARGET_PROXIMITY = "test_for_target_proximity"


class EditInstruction(OptimizationModel):
    """A machine-readable edit, for Phase 8 to execute or a human to apply."""

    operation: Literal[
        "move_event", "insert_speech", "adjust_feature", "reorder_events",
        "extend_event", "replace_text", "adjust_pace",
    ]
    event_type: str | None = None
    event_id: str | None = None
    from_time: float | None = None
    to_time: float | None = None
    feature: str | None = None
    target_value: float | None = None
    original: str | None = None
    replacement: str | None = None
    note: str | None = None


class ContentIntervention(OptimizationModel):
    intervention_id: str
    type: InterventionType
    target_interval_start: float
    target_interval_end: float
    target_content_event_id: str | None = None
    description: str
    parameters: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    rationale: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    objective_id: str
    expected_direction: ExpectedDirection
    edit_instructions: tuple[EditInstruction, ...] = ()
    edit_cost: EditCost = EditCost.MEDIUM
    modality: str = "visual"


class EvidenceConfidence(OptimizationModel):
    """How well-supported a *test* is. Explicitly not a success probability.

    Nothing here estimates whether the objective will improve; that is
    unknowable before re-simulation. Each component measures how much measured
    support exists for considering the intervention worth running.
    """

    value: float = Field(ge=0, le=1)
    reference_consistency: float = Field(ge=0, le=1)
    feature_difference_strength: float = Field(ge=0, le=1)
    objective_gap_strength: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=0)
    formula: str
    caveat: str = (
        "Evidence confidence measures the strength of the measured support for "
        "running this test. It is not the probability that the objective will "
        "improve, which is unknown until the candidate is re-simulated."
    )


class OptimizationRecommendation(OptimizationModel):
    recommendation_id: str
    title: str
    objective_id: str
    source_variant_id: str
    interventions: tuple[ContentIntervention, ...] = Field(min_length=1)
    target_interval_start: float
    target_interval_end: float
    rationale: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    reference_variant_ids: tuple[str, ...] = ()
    evidence_confidence: EvidenceConfidence
    expected_direction: ExpectedDirection
    risk: RiskLevel = RiskLevel.LOW
    edit_cost: EditCost = EditCost.MEDIUM
    #: Ranking score. Transparent formula, recorded on the result.
    priority: float = 0.0
    #: True when this bundles several interventions, which is harder to attribute.
    is_bundle: bool = False
    hypothesis_id: str | None = None
    warnings: tuple[str, ...] = ()


class OptimizationHypothesis(OptimizationModel):
    """A recommendation restated as something Phase 8 can pass or fail."""

    hypothesis_id: str
    statement: str
    objective_id: str
    source_variant_id: str
    recommendation_id: str
    changed_features: tuple[str, ...] = ()
    interval_start_seconds: float
    interval_end_seconds: float
    expected_direction: ExpectedDirection
    evidence_ids: tuple[str, ...] = ()
    #: Set by Phase 8 after re-simulation. Never set here.
    outcome: Literal["untested", "pass", "fail", "inconclusive"] = "untested"


class ApprovalState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"


class RecommendationReview(OptimizationModel):
    """A human decision on one recommendation.

    Recommendations never modify content on their own. This record is what
    turns one into a candidate, and it keeps the original alongside any
    user modification so the two are never confused.
    """

    recommendation_id: str
    state: ApprovalState
    reviewer: str | None = None
    reason: str | None = None
    #: The user's replacement interventions, when state is MODIFIED.
    modified_interventions: tuple[ContentIntervention, ...] = ()
    reviewed_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class ProposedVariantSpec(OptimizationModel):
    """The Phase 8 input. A description of a variant that does not exist yet."""

    proposed_variant_id: str
    parent_variant_id: str
    experiment_id: str
    hypothesis_ids: tuple[str, ...] = Field(min_length=1)
    interventions: tuple[ContentIntervention, ...] = Field(min_length=1)
    constraints: OptimizationConstraints
    target_objective_id: str
    expected_direction: ExpectedDirection
    edit_instructions: tuple[EditInstruction, ...] = ()
    #: Intervals the source performs well in, which the edit should leave alone.
    preserve_intervals: tuple[tuple[float, float], ...] = ()
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    optimization_version: str = OPTIMIZATION_VERSION
    note: str = (
        "A specification for a candidate to build and re-simulate. No claim is made "
        "that it improves the objective; that is what Phase 8 measures."
    )


class OptimizationTrace(OptimizationModel):
    """The chain from objective to recommendation, for auditing one output."""

    recommendation_id: str
    objective_id: str
    source_variant_id: str
    interval_id: str | None = None
    reference_variant_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    feature_differences: tuple[str, ...] = ()
    hypothesis_id: str | None = None


class OptimizationMetadata(OptimizationModel):
    optimization_version: str = OPTIMIZATION_VERSION
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    scoring_version: str | None = None
    comparison_version: str | None = None
    content_analysis_versions: dict[str, str] = Field(default_factory=dict)
    objective_set_hash: str | None = None
    #: How many candidate features were examined, so a reader can judge how
    #: much selection went into the reported differences.
    features_considered: int = 0
    generator: str = "deterministic"
    llm_model: str | None = None
    prompt_version: str | None = None
    interpretation_notice: str = (
        "These are candidate interventions to test, derived from differences measured "
        "between variants. No recommendation is a validated improvement, and none "
        "establishes a causal relationship. A recommendation becomes scientifically "
        "meaningful only after the candidate is built, re-run through TRIBE and rescored."
    )


class OptimizationResult(OptimizationModel):
    schema_version: str = "1.0"
    request: OptimizationRequest
    gaps: tuple[ObjectiveGap, ...] = ()
    weak_intervals: tuple[OptimizationInterval, ...] = ()
    strong_intervals: tuple[OptimizationInterval, ...] = ()
    reference_variant_ids: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    recommendations: tuple[OptimizationRecommendation, ...] = ()
    hypotheses: tuple[OptimizationHypothesis, ...] = ()
    traces: tuple[OptimizationTrace, ...] = ()
    rejected: tuple[dict[str, str], ...] = ()
    metadata: OptimizationMetadata = OptimizationMetadata()
    warnings: tuple[str, ...] = ()
