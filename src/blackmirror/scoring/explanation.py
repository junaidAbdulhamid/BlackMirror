"""Deterministic score explanations.

WHAT THIS DOES
    Turns a scored experiment into a structured account of why one variant
    ranked above another: which objectives separated them, by how much, over
    which intervals, and what the content was doing during those intervals.

WHY IT IS NOT AN LLM
    Every statement is generated from a measured number by a fixed template. A
    language model can summarise this output afterwards, but it must never be
    the thing that decides *why* a variant won, because it would have no way to
    distinguish a real 0.11 difference from a plausible-sounding one.

THE LINE THAT MATTERS
    Content differences are reported as having occurred *during the same
    interval* as a neural difference. They are never reported as having caused
    it. The wording here is deliberate and the caveats are attached to every
    explanation rather than left to the caller.
"""

from __future__ import annotations

from blackmirror.scoring.schemas import (
    ExperimentScoreResult,
    IntervalContext,
    ObjectiveComparison,
    PairedEffectSummary,
    ScoreExplanation,
    VariantScore,
)
from blackmirror.scoring.windows import ContentEventInterval

#: Below this the ranking is reported as effectively a tie.
CLOSE_MARGIN = 0.01

#: A relative difference is undefined against a value this close to zero.
MIN_DENOMINATOR = 1e-9


def explain(
    result: ExperimentScoreResult,
    *,
    content_events: dict[str, list[ContentEventInterval]] | None = None,
    content_descriptions: dict[str, list[dict[str, object]]] | None = None,
) -> ScoreExplanation:
    """Build a structured explanation of the top two variants' separation."""
    ranked = [score for score in result.variant_scores if score.total_score is not None]
    leader = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None

    total_difference = None
    if leader is not None and runner_up is not None:
        total_difference = float(leader.total_score - runner_up.total_score)  # type: ignore[operator]
    close = total_difference is not None and abs(total_difference) < CLOSE_MARGIN

    comparisons = _compare_objectives(result, leader, runner_up)
    context = _context(result, leader, runner_up, content_descriptions)
    statements = _statements(result, leader, runner_up, total_difference, close, comparisons)

    return ScoreExplanation(
        experiment_id=result.experiment_id,
        objective_set_hash=result.metadata.objective_set_hash,
        leader_variant_id=leader.variant_id if leader else None,
        runner_up_variant_id=runner_up.variant_id if runner_up else None,
        total_difference=total_difference,
        ranking_is_close=close,
        objectives=comparisons,
        context=context,
        statements=statements,
        caveats=_caveats(result, close),
    )


def _compare_objectives(
    result: ExperimentScoreResult,
    leader: VariantScore | None,
    runner_up: VariantScore | None,
) -> tuple[ObjectiveComparison, ...]:
    if leader is None:
        return ()
    participants = [leader] + ([runner_up] if runner_up else [])
    comparisons: list[ObjectiveComparison] = []

    for objective in result.objectives:
        raw: dict[str, float | None] = {}
        scores: dict[str, float | None] = {}
        windows = {}
        peaks: dict[str, float] = {}
        for variant in participants:
            evaluation = next(
                (e for e in variant.objective_scores if e.objective_id == objective.objective_id),
                None,
            )
            if evaluation is None:
                continue
            raw[variant.variant_id] = evaluation.raw_value
            scores[variant.variant_id] = evaluation.score
            windows[variant.variant_id] = evaluation.window
            if evaluation.temporal_contribution is not None:
                peaks[variant.variant_id] = evaluation.temporal_contribution.peak_time_seconds

        usable = {k: v for k, v in scores.items() if v is not None}
        leader_id = max(usable, key=lambda k: usable[k]) if usable else None

        difference = relative = None
        raw_usable = {k: v for k, v in raw.items() if v is not None}
        if len(raw_usable) == 2:
            high, low = sorted(raw_usable.values(), reverse=True)
            difference = float(high - low)
            # A percentage against a near-zero denominator is not a number a
            # reader should be shown, so it is withheld rather than inflated.
            relative = float(difference / abs(low)) if abs(low) > MIN_DENOMINATOR else None

        comparisons.append(
            ObjectiveComparison(
                objective_id=objective.objective_id,
                objective_name=objective.name,
                metric=objective.metric,
                direction=objective.direction,
                target_label=_target_label(leader, objective.objective_id),
                leader_variant_id=leader_id,
                raw_values=raw,
                scores=scores,
                absolute_difference=difference,
                relative_difference=relative,
                windows=windows,
                peak_intervals=peaks,
                paired_effect=PairedEffectSummary(
                    status="unavailable",
                    sample_count=min(
                        (window.sample_count for window in windows.values()), default=0
                    ),
                    reason=(
                        "Persisted score results do not retain paired target-series samples; "
                        "a standardized paired effect cannot be reconstructed without "
                        "fabricating observations."
                    ),
                    assumptions=(
                        "A descriptive paired effect would require aligned paired samples.",
                        "Temporal samples are autocorrelated and are not independent subjects.",
                        "No inferential p-value or confidence interval is implied.",
                    ),
                ),
            )
        )
    return tuple(comparisons)


def _target_label(variant: VariantScore, objective_id: str) -> str:
    evaluation = next(
        (e for e in variant.objective_scores if e.objective_id == objective_id), None
    )
    return evaluation.target.label if evaluation else "unresolved"


def _context(
    result: ExperimentScoreResult,
    leader: VariantScore | None,
    runner_up: VariantScore | None,
    descriptions: dict[str, list[dict[str, object]]] | None,
) -> dict[str, IntervalContext]:
    """Content at each variant's highest-contribution interval, if available."""
    if descriptions is None:
        return {}
    context: dict[str, IntervalContext] = {}
    for variant in [v for v in (leader, runner_up) if v is not None]:
        peak_time = None
        for evaluation in variant.objective_scores:
            if evaluation.temporal_contribution is not None:
                peak_time = evaluation.temporal_contribution.peak_time_seconds
                break
        if peak_time is None:
            continue
        events = descriptions.get(variant.variant_id) or []
        overlapping = [
            event
            for event in events
            if _number(event, "start_time") <= peak_time < _number(event, "end_time")
        ]
        if not overlapping:
            continue
        first = overlapping[0]
        context[variant.variant_id] = IntervalContext(
            start_seconds=_number(first, "start_time"),
            end_seconds=_number(first, "end_time"),
            visual_description=_as_text(first.get("visual_description")),
            speech_text=_as_text(first.get("speech_text")),
            audio_description=_as_text(first.get("audio_description")),
            on_screen_text=_strings(first.get("on_screen_text")),
            objects=_strings(first.get("objects")),
            event_types=tuple(sorted({str(e.get("event_type")) for e in overlapping})),
        )
    return context


def _number(event: dict[str, object], key: str) -> float:
    value = event.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _strings(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list | tuple) else ()


def _as_text(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


def _statements(
    result: ExperimentScoreResult,
    leader: VariantScore | None,
    runner_up: VariantScore | None,
    total_difference: float | None,
    close: bool,
    comparisons: tuple[ObjectiveComparison, ...],
) -> tuple[str, ...]:
    """Fixed templates over measured numbers. No inference, no causal language."""
    if leader is None:
        return ("No variant produced a complete score for this objective set.",)

    lines: list[str] = []
    if runner_up is None:
        lines.append(
            f"{leader.variant_id} is the only variant with a complete score for this "
            f"objective set, so no ranking comparison is available."
        )
        return tuple(lines)

    lines.append(
        f"{leader.variant_id} ranked highest for the selected objective set, scoring "
        f"{leader.total_score:.4f} against {runner_up.total_score:.4f} for "
        f"{runner_up.variant_id}."
    )
    if close:
        lines.append(
            f"The separation is {total_difference:.4f}, which is small enough that the "
            f"ordering should not be treated as a meaningful difference between the variants."
        )

    for comparison in comparisons:
        if comparison.leader_variant_id is None or comparison.absolute_difference is None:
            continue
        values = ", ".join(
            f"{vid} {value:.4f}"
            for vid, value in comparison.raw_values.items()
            if value is not None
        )
        relative = (
            f" ({comparison.relative_difference:+.1%} relative)"
            if comparison.relative_difference is not None
            else ""
        )
        lines.append(
            f"On '{comparison.objective_name}' ({comparison.metric} over "
            f"{comparison.target_label}, direction {comparison.direction.value}), the raw "
            f"values were {values}; a difference of {comparison.absolute_difference:.4f}"
            f"{relative}. {comparison.leader_variant_id} scored higher."
        )
        peak = comparison.peak_intervals.get(comparison.leader_variant_id)
        if peak is not None:
            lines.append(
                f"  Its largest single contribution to that objective fell at "
                f"{peak:.2f}s within the measured window."
            )

    # Conflicts must stay visible rather than being resolved by the composite.
    winners = {c.leader_variant_id for c in comparisons if c.leader_variant_id}
    if len(winners) > 1:
        lines.append(
            "The objectives disagree: different variants lead on different objectives, so "
            "the composite ordering depends on the weights that were chosen."
        )
    return tuple(lines)


def _caveats(result: ExperimentScoreResult, close: bool) -> tuple[str, ...]:
    caveats = [
        "Content described alongside an interval occurred during the same interval as the "
        "measured neural difference. It is not shown to have caused it, and no causal claim "
        "is available from this comparison.",
        "Scores are values of explicitly chosen mathematical functions of predicted "
        "average-subject cortical responses. A higher score means the variant scored higher "
        "on the objective that was defined, not that it is more persuasive, memorable or "
        "effective.",
        "These predictions are deterministic model output. There is no sampling "
        "distribution, so no p-value, confidence interval or statistical significance is "
        "available for a score difference.",
    ]
    if close:
        caveats.append(
            "The top two variants are separated by less than the reporting threshold; treat "
            "them as indistinguishable under this objective set."
        )
    if any("does not reflect the stated weighting" in w for w in result.warnings):
        caveats.append(
            "At least one objective accounted for a share of the composite far from its "
            "stated weight, so the composite ordering does not reflect the weighting that "
            "was requested. Read the per-objective values."
        )
    return tuple(caveats)
