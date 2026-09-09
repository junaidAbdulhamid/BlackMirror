"""Rejecting recommendations that should not reach a human.

FOUR GATES
    1. Evidence grounding -- every cited id must exist. An ungrounded
       recommendation is indistinguishable from a guess.
    2. Constraints -- a frozen modality or preserved interval is a rule the
       user set, not a preference to trade away.
    3. Causal language -- Phase 7 produces hypotheses. Text asserting an
       outcome is rejected, because once "will increase conversions" appears in
       a UI, nobody reads the caveat underneath it.
    4. Actionability -- an intervention with no interval, no instructions or a
       zero-length target cannot be built by Phase 8.

WHY REJECT RATHER THAN REWRITE
    Rewriting a causal claim into a hedged one hides that the generator
    produced it. Rejections are recorded with a reason so the failure is
    visible and countable.
"""

from __future__ import annotations

import re

from blackmirror.optimization.schemas import (
    INTERVENTION_MODALITY,
    Evidence,
    OptimizationConstraints,
    OptimizationRecommendation,
)

#: Phrases asserting an outcome, a cause, or a psychological effect. Matched
#: case-insensitively on word boundaries against every free-text field.
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bwill\s+(increase|improve|boost|raise|cause|create|drive|lift)\b", "asserts an outcome"),
    (r"\bguarantee(s|d)?\b", "asserts certainty"),
    (r"\bproven\b", "asserts proof"),
    # Verb forms only. The bare noun appears in disclaimers -- "not a
    # demonstrated cause" -- and flagging those would reject the very text that
    # keeps a recommendation honest.
    (r"\bcaus(es|ed|ing)\b", "asserts causation"),
    (r"\bwill\s+cause\b", "asserts causation"),
    (r"\bbecause of (this|the) (edit|change|intervention)\b", "asserts causation"),
    (r"\bconversions?\b", "claims a business outcome"),
    (r"\bpurchase intent\b", "claims a psychological construct"),
    (r"\bpersuasi(on|ve)\b", "claims persuasion"),
    (r"\bmemorabilit(y|ies)\b|\bimproves? memory\b", "claims a memory effect"),
    (r"\bengagement\b", "claims an unmeasured construct"),
    (r"\battention\b", "claims an unmeasured construct"),
    (r"\bemotional impact\b|\bcreates? emotion\b", "claims an emotional effect"),
    (r"\boptimal\b|\bbest possible\b", "asserts optimality that was not established"),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in FORBIDDEN_PATTERNS
)


def scan_language(text: str) -> list[str]:
    """Reasons this text makes a claim Phase 7 cannot support."""
    return [reason for pattern, reason in _COMPILED if pattern.search(text)]


def validate(
    recommendation: OptimizationRecommendation,
    evidence: dict[str, Evidence],
    constraints: OptimizationConstraints,
) -> list[str]:
    """Blocking problems with one recommendation. Empty means acceptable."""
    problems: list[str] = []

    # 1. Evidence grounding.
    missing = [item for item in recommendation.evidence_ids if item not in evidence]
    if missing:
        problems.append(f"cites evidence that does not exist: {missing}")
    if not recommendation.evidence_ids:
        problems.append("cites no evidence at all")
    for intervention in recommendation.interventions:
        unknown = [item for item in intervention.evidence_ids if item not in evidence]
        if unknown:
            problems.append(
                f"intervention {intervention.intervention_id} cites unknown evidence {unknown}"
            )

    # 2. Constraints.
    for intervention in recommendation.interventions:
        modality = INTERVENTION_MODALITY.get(intervention.type, intervention.modality)
        if modality in constraints.frozen_modalities:
            problems.append(
                f"intervention {intervention.intervention_id} modifies the {modality} "
                f"modality, which the request froze"
            )
        if constraints.forbid_new_voiceover and intervention.type.value == "speech_edit":
            problems.append(
                f"intervention {intervention.intervention_id} adds speech, which the "
                f"request forbade"
            )
        for start, end in constraints.preserve_intervals:
            if (
                intervention.target_interval_start < end
                and intervention.target_interval_end > start
            ):
                problems.append(
                    f"intervention {intervention.intervention_id} targets "
                    f"{intervention.target_interval_start:.2f}-"
                    f"{intervention.target_interval_end:.2f}s, which overlaps the preserved "
                    f"interval {start:.2f}-{end:.2f}s"
                )
        if intervention.type.value in {"text_edit", "cta_edit"} and (
            "cta" in constraints.frozen_event_types
            or "text_overlay" in constraints.frozen_event_types
        ):
            problems.append(
                f"intervention {intervention.intervention_id} edits text whose event type "
                f"the request froze"
            )

    # 3. Causal language, across every free-text field a human will read.
    texts = [recommendation.title, recommendation.rationale]
    texts += [i.description for i in recommendation.interventions]
    texts += [i.rationale for i in recommendation.interventions]
    for text in texts:
        for reason in scan_language(text):
            problems.append(f"unsupported claim ({reason}) in: {text[:80]!r}")

    # 4. Actionability.
    if recommendation.target_interval_end <= recommendation.target_interval_start:
        problems.append("target interval is empty")
    for intervention in recommendation.interventions:
        if not intervention.edit_instructions:
            problems.append(
                f"intervention {intervention.intervention_id} has no machine-readable edit "
                f"instruction, so Phase 8 could not build it"
            )
    return problems


def deduplicate(
    recommendations: list[OptimizationRecommendation],
) -> tuple[list[OptimizationRecommendation], list[dict[str, str]]]:
    """Collapse recommendations that would produce the same edit.

    Structural rather than textual: two recommendations are the same when they
    touch the same intervention type over the same interval, whatever their
    titles say. That catches "move CTA earlier" and "show CTA sooner" without
    needing embeddings.
    """
    seen: dict[tuple[str, float, float], OptimizationRecommendation] = {}
    dropped: list[dict[str, str]] = []
    for recommendation in recommendations:
        key = (
            "+".join(sorted(i.type.value for i in recommendation.interventions)),
            round(recommendation.target_interval_start, 2),
            round(recommendation.target_interval_end, 2),
        )
        existing = seen.get(key)
        if existing is None:
            seen[key] = recommendation
            continue
        # Keep whichever has the stronger measured support.
        keep, drop = (
            (recommendation, existing)
            if recommendation.evidence_confidence.value > existing.evidence_confidence.value
            else (existing, recommendation)
        )
        seen[key] = keep
        dropped.append(
            {
                "recommendation_id": drop.recommendation_id,
                "reason": (
                    f"duplicate of {keep.recommendation_id}: same intervention type over the "
                    f"same interval"
                ),
            }
        )
    return list(seen.values()), dropped


#: Relative implementation effort, used only in ranking.
_COST_WEIGHT = {"low": 1.0, "medium": 1.6, "high": 2.6}


def rank(
    recommendations: list[OptimizationRecommendation],
    *,
    max_recommendations: int,
    diversify: bool = True,
) -> list[OptimizationRecommendation]:
    """Order by a transparent priority, then spread across intervention types.

        priority = evidence_confidence * information_value / cost_weight

    `information_value` is 1.0 for an atomic intervention and 0.6 for a bundle,
    because a bundle answers a vaguer question: a score change cannot be
    attributed to any single part of it. This is a ranking heuristic and
    explicitly not an estimate of success.
    """
    scored: list[OptimizationRecommendation] = []
    for recommendation in recommendations:
        information_value = 0.6 if recommendation.is_bundle else 1.0
        cost = _COST_WEIGHT.get(recommendation.edit_cost.value, 1.6)
        priority = recommendation.evidence_confidence.value * information_value / cost
        scored.append(recommendation.model_copy(update={"priority": round(priority, 6)}))

    scored.sort(key=lambda item: (-item.priority, item.recommendation_id))
    if not diversify:
        return scored[:max_recommendations]

    # One per intervention type first, so a single strong feature cannot fill
    # the whole list with variations on itself.
    chosen: list[OptimizationRecommendation] = []
    used: set[str] = set()
    for recommendation in scored:
        signature = "+".join(sorted(i.type.value for i in recommendation.interventions))
        if signature not in used:
            chosen.append(recommendation)
            used.add(signature)
        if len(chosen) == max_recommendations:
            return chosen
    for recommendation in scored:
        if recommendation not in chosen:
            chosen.append(recommendation)
        if len(chosen) == max_recommendations:
            break
    return chosen
