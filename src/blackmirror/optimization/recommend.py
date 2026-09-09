"""Turning measured evidence into candidate interventions.

DETERMINISTIC, AND FIRST
    No language model is involved. Each rule maps a specific measured feature
    difference onto a specific intervention type, so every recommendation can
    name the numbers that produced it. An LLM may later rewrite these into
    prose, but it must not be what decides *what* to recommend -- it has no way
    to tell a real 0.17 motion difference from a plausible-sounding one.

ATOMIC BY DEFAULT
    One intervention per recommendation, because Phase 8 has to attribute a
    score change to a cause. A bundle of five edits that improves the objective
    tells you almost nothing about which edit mattered. Bundles are offered
    only under the EXPLORATORY strategy and are labelled as harder to attribute.

WHAT A RULE MAY SAY
    That a feature differed, during which interval, in how many references. Not
    that changing it will improve anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from blackmirror.optimization.schemas import (
    ContentIntervention,
    EditCost,
    EditInstruction,
    Evidence,
    EvidenceConfidence,
    ExpectedDirection,
    InterventionType,
    ObjectiveGap,
    OptimizationInterval,
    OptimizationRecommendation,
    OptimizationStrategy,
    RiskLevel,
)

#: Weights for evidence confidence. Documented rather than tuned: each term is
#: a measured quantity, and the split reflects that agreement across references
#: is worth more than the size of any single difference.
CONFIDENCE_WEIGHTS = {
    "reference_consistency": 0.40,
    "feature_difference_strength": 0.25,
    "objective_gap_strength": 0.25,
    "evidence_volume": 0.10,
}


@dataclass(frozen=True)
class Rule:
    """One feature difference mapped onto one intervention type."""

    feature: str
    intervention: InterventionType
    #: Which sign of delta (reference minus source) this rule responds to.
    direction: int
    template: str
    cost: EditCost
    operation: str
    modality: str


#: Feature-to-intervention rules. Every feature named here is a column Phase 4
#: actually measures; nothing is invented.
RULES: tuple[Rule, ...] = (
    Rule(
        "motion", InterventionType.PACE_EDIT, +1,
        "Increase visual movement during {interval}",
        EditCost.MEDIUM, "adjust_feature", "visual",
    ),
    Rule(
        "motion_structural", InterventionType.VISUAL_EDIT, +1,
        "Introduce subject movement during {interval}",
        EditCost.MEDIUM, "adjust_feature", "visual",
    ),
    Rule(
        "shot_change", InterventionType.SHOT_EDIT, +1,
        "Increase cutting rate during {interval}",
        EditCost.MEDIUM, "adjust_pace", "visual",
    ),
    Rule(
        "speech_present", InterventionType.SPEECH_EDIT, +1,
        "Add spoken narration during {interval}",
        EditCost.MEDIUM, "insert_speech", "speech",
    ),
    Rule(
        "speech_activity", InterventionType.SPEECH_EDIT, +1,
        "Extend speech across {interval}",
        EditCost.MEDIUM, "insert_speech", "speech",
    ),
    Rule(
        "audio_energy", InterventionType.AUDIO_EDIT, +1,
        "Raise audio energy during {interval}",
        EditCost.LOW, "adjust_feature", "audio",
    ),
    Rule(
        "text_present", InterventionType.TEXT_EDIT, +1,
        "Add an on-screen text overlay during {interval}",
        EditCost.LOW, "adjust_feature", "text",
    ),
    Rule(
        "cta_present", InterventionType.CTA_EDIT, +1,
        "Bring the call to action into {interval}",
        EditCost.LOW, "move_event", "text",
    ),
    Rule(
        "object_count", InterventionType.PRODUCT_VISIBILITY_EDIT, +1,
        "Increase on-screen subject visibility during {interval}",
        EditCost.MEDIUM, "adjust_feature", "visual",
    ),
    Rule(
        "brightness", InterventionType.VISUAL_EDIT, +1,
        "Raise scene brightness during {interval}",
        EditCost.LOW, "adjust_feature", "visual",
    ),
    Rule(
        "saturation", InterventionType.VISUAL_EDIT, +1,
        "Increase colour saturation during {interval}",
        EditCost.LOW, "adjust_feature", "visual",
    ),
    Rule(
        "luminance_shift", InterventionType.SHOT_EDIT, +1,
        "Add a lighting or shot transition during {interval}",
        EditCost.MEDIUM, "adjust_pace", "visual",
    ),
)

_BY_FEATURE = {rule.feature: rule for rule in RULES}


def evidence_confidence(
    supporting: list[Evidence], gap: ObjectiveGap, interval: OptimizationInterval
) -> EvidenceConfidence:
    """How well-supported this test is. Not a success probability.

    Each term is a measured quantity normalised to [0, 1]:
      consistency   - fraction of references agreeing on the difference
      strength      - size of the difference relative to the values involved
      gap strength  - how much room the objective has, relative to reference
      volume        - how many independent items support it, saturating at 4
    """
    # Consistency is undefined when only one reference was available. It is
    # scored 0 rather than imputed, which caps a single-reference recommendation
    # at 0.60 -- the honest ceiling for evidence that could not be corroborated.
    consistencies = [e.consistency for e in supporting if e.consistency is not None]
    consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0
    corroborated = bool(consistencies)

    strengths = []
    for item in supporting:
        if item.delta is None or item.source_value is None:
            continue
        scale = max(abs(item.source_value), abs(item.reference_value or 0.0), 1e-9)
        strengths.append(min(1.0, abs(item.delta) / scale))
    strength = sum(strengths) / len(strengths) if strengths else 0.0

    gap_strength = 0.0
    if gap.gap is not None and gap.gap > 0:
        gap_strength = min(1.0, abs(gap.relative_gap)) if gap.relative_gap else 0.5

    volume = min(1.0, len(supporting) / 4.0)

    value = (
        CONFIDENCE_WEIGHTS["reference_consistency"] * consistency
        + CONFIDENCE_WEIGHTS["feature_difference_strength"] * strength
        + CONFIDENCE_WEIGHTS["objective_gap_strength"] * gap_strength
        + CONFIDENCE_WEIGHTS["evidence_volume"] * volume
    )
    # A one-sample interval cannot support a confident test whatever the deltas.
    if interval.sample_count < 2:
        value *= 0.5

    return EvidenceConfidence(
        value=round(min(1.0, max(0.0, value)), 4),
        reference_consistency=round(consistency, 4),
        feature_difference_strength=round(strength, 4),
        objective_gap_strength=round(gap_strength, 4),
        evidence_count=len(supporting),
        formula=(
            "0.40*reference_consistency + 0.25*feature_difference_strength + "
            "0.25*objective_gap_strength + 0.10*min(1, evidence_count/4); "
            "halved when the interval holds fewer than 2 samples"
            + (
                ""
                if corroborated
                else "; consistency scored 0 because only one reference variant was "
                "available, so this value cannot exceed 0.60"
            )
        ),
    )


def generate(
    interval: OptimizationInterval,
    gap: ObjectiveGap,
    evidence: list[Evidence],
    *,
    strategy: OptimizationStrategy,
    source_variant_id: str,
    reference_variant_ids: tuple[str, ...],
    prefix: str,
) -> list[OptimizationRecommendation]:
    """One atomic recommendation per qualifying feature difference."""
    contribution = [e for e in evidence if e.kind.value == "contribution_share"]
    gap_items = [e for e in evidence if e.kind.value == "objective_gap"]
    feature_items = [
        e for e in evidence if e.kind.value == "content_feature_delta" and e.feature
    ]
    event_items = [e for e in evidence if e.kind.value == "content_event_presence"]

    interval_label = f"{interval.start_seconds:.2f}-{interval.end_seconds:.2f}s"
    recommendations: list[OptimizationRecommendation] = []

    for position, item in enumerate(feature_items):
        rule = _BY_FEATURE.get(item.feature or "")
        if rule is None or item.delta is None:
            continue
        # Only recommend moving a feature the way the references actually differ.
        if (item.delta > 0) != (rule.direction > 0):
            continue

        supporting = [item, *contribution, *gap_items]
        confidence = evidence_confidence(supporting, gap, interval)
        target = (item.source_value or 0.0) + item.delta
        intervention = ContentIntervention(
            intervention_id=f"{prefix}-I{position:02d}",
            type=rule.intervention,
            target_interval_start=interval.start_seconds,
            target_interval_end=interval.end_seconds,
            description=rule.template.format(interval=interval_label),
            parameters={
                "feature": rule.feature,
                "source_value": item.source_value,
                "reference_value": item.reference_value,
                "delta": item.delta,
            },
            rationale=(
                f"{len(item.reference_variant_ids)} higher-scoring reference variant(s) "
                f"measured {rule.feature} at {item.reference_value:.4f} during this interval "
                f"against {item.source_value:.4f} in the source. This is a temporal "
                f"comparison, not a demonstrated cause."
            ),
            evidence_ids=tuple(e.evidence_id for e in supporting),
            objective_id=interval.objective_id,
            expected_direction=_direction(gap),
            edit_instructions=(
                EditInstruction(
                    operation=rule.operation,  # type: ignore[arg-type]
                    feature=rule.feature,
                    from_time=interval.start_seconds,
                    to_time=interval.end_seconds,
                    target_value=round(target, 6),
                    note=(
                        "Target taken from the reference mean over the same interval; it is "
                        "a comparison point, not a predicted optimum."
                    ),
                ),
            ),
            edit_cost=rule.cost,
            modality=rule.modality,
        )
        recommendations.append(
            OptimizationRecommendation(
                recommendation_id=f"{prefix}-R{position:02d}",
                title=rule.template.format(interval=interval_label),
                objective_id=interval.objective_id,
                source_variant_id=source_variant_id,
                interventions=(intervention,),
                target_interval_start=interval.start_seconds,
                target_interval_end=interval.end_seconds,
                rationale=intervention.rationale,
                evidence_ids=intervention.evidence_ids,
                reference_variant_ids=reference_variant_ids,
                evidence_confidence=confidence,
                expected_direction=intervention.expected_direction,
                risk=RiskLevel.LOW,
                edit_cost=rule.cost,
            )
        )

    for position, item in enumerate(event_items):
        kind = (item.feature or "").removeprefix("event:")
        supporting = [item, *contribution, *gap_items]
        confidence = evidence_confidence(supporting, gap, interval)
        intervention = ContentIntervention(
            intervention_id=f"{prefix}-IE{position:02d}",
            type=(
                InterventionType.CTA_EDIT if kind == "cta" else InterventionType.SPEECH_EDIT
                if kind == "speech_segment"
                else InterventionType.SCENE_STRUCTURE_EDIT
            ),
            target_interval_start=interval.start_seconds,
            target_interval_end=interval.end_seconds,
            description=f"Introduce a {kind.replace('_', ' ')} into {interval_label}",
            parameters={"event_type": kind},
            rationale=(
                f"Reference variant(s) {', '.join(item.reference_variant_ids)} contain a "
                f"{kind} overlapping this interval and the source does not. Presence is a "
                f"measured difference, not a demonstrated cause."
            ),
            evidence_ids=tuple(e.evidence_id for e in supporting),
            objective_id=interval.objective_id,
            expected_direction=_direction(gap),
            edit_instructions=(
                EditInstruction(
                    operation="move_event",
                    event_type=kind,
                    to_time=interval.start_seconds,
                    note="Timing taken from where references place this event type.",
                ),
            ),
            edit_cost=EditCost.MEDIUM,
            modality="speech" if kind == "speech_segment" else "text",
        )
        recommendations.append(
            OptimizationRecommendation(
                recommendation_id=f"{prefix}-RE{position:02d}",
                title=intervention.description,
                objective_id=interval.objective_id,
                source_variant_id=source_variant_id,
                interventions=(intervention,),
                target_interval_start=interval.start_seconds,
                target_interval_end=interval.end_seconds,
                rationale=intervention.rationale,
                evidence_ids=intervention.evidence_ids,
                reference_variant_ids=reference_variant_ids,
                evidence_confidence=confidence,
                expected_direction=intervention.expected_direction,
                risk=RiskLevel.MEDIUM,
                edit_cost=EditCost.MEDIUM,
            )
        )

    if strategy is OptimizationStrategy.EXPLORATORY and len(recommendations) >= 2:
        recommendations.append(_bundle(recommendations[:3], prefix, reference_variant_ids))
    return recommendations


def _bundle(
    parts: list[OptimizationRecommendation],
    prefix: str,
    reference_variant_ids: tuple[str, ...],
) -> OptimizationRecommendation:
    """Several atomic interventions combined, labelled as hard to attribute."""
    interventions = tuple(item for part in parts for item in part.interventions)
    evidence_ids = tuple(dict.fromkeys(i for part in parts for i in part.evidence_ids))
    confidence = min(parts, key=lambda p: p.evidence_confidence.value).evidence_confidence
    return OptimizationRecommendation(
        recommendation_id=f"{prefix}-BUNDLE",
        title=f"Combined test: {', '.join(part.title for part in parts)}",
        objective_id=parts[0].objective_id,
        source_variant_id=parts[0].source_variant_id,
        interventions=interventions,
        target_interval_start=parts[0].target_interval_start,
        target_interval_end=parts[0].target_interval_end,
        rationale=(
            "Combines several individually-evidenced changes. A score change from this "
            "bundle cannot be attributed to any one of them, so run the atomic tests first "
            "if attribution matters."
        ),
        evidence_ids=evidence_ids,
        reference_variant_ids=reference_variant_ids,
        evidence_confidence=confidence,
        expected_direction=parts[0].expected_direction,
        risk=RiskLevel.HIGH,
        edit_cost=EditCost.HIGH,
        is_bundle=True,
        warnings=("bundled changes cannot be attributed individually by Phase 8",),
    )


def _direction(gap: ObjectiveGap) -> ExpectedDirection:
    if gap.direction == "target":
        return ExpectedDirection.TEST_FOR_TARGET_PROXIMITY
    if gap.direction == "minimize":
        return ExpectedDirection.TEST_FOR_DECREASE
    return ExpectedDirection.TEST_FOR_INCREASE
