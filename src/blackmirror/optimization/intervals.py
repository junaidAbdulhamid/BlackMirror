"""Objective gap analysis and weak/strong interval detection.

WHY "WEAK" MUST BE OBJECTIVE-RELATIVE
    It is tempting to call an interval weak because the response is low. That
    would be wrong: a MINIMIZE objective wants a low value, and a TARGET
    objective wants a specific one. Weakness here means "contributed
    comparatively little to the objective as defined", computed from Phase 6's
    per-sample contribution decomposition, whose entries sum exactly to the raw
    value. That property is what makes a contribution *share* meaningful.

WHY STRONG INTERVALS MATTER AS MUCH
    An optimizer that only knows where a variant is weak will happily propose
    edits that destroy the parts already working. Strong intervals are detected
    for the explicit purpose of marking them PRESERVE, and they flow through to
    the proposed variant spec.

WHAT IT REFUSES TO DO
    Four of the seven Phase 6 metrics (stability, temporal change, divergence,
    pattern similarity) are not sums over samples, so they expose no temporal
    decomposition. For those, interval detection returns nothing and says why,
    rather than inventing a split.
"""

from __future__ import annotations

import numpy as np

from blackmirror.optimization.schemas import (
    IntervalKind,
    ObjectiveGap,
    OptimizationInterval,
)
from blackmirror.scoring.schemas import (
    ExperimentScoreResult,
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveEvaluation,
    VariantScore,
)

#: Contribution share below which an interval is a weak candidate. With a 1 s
#: TR a 30 s stimulus has ~30 samples, so an even spread is ~3.3% each; a
#: quarter of even is a meaningful shortfall rather than sampling noise.
WEAK_SHARE_RATIO = 0.25

#: Share above which an interval is treated as carrying the objective.
STRONG_SHARE_RATIO = 2.0

#: Adjacent qualifying samples closer than this are merged into one interval.
MERGE_GAP_SECONDS = 1.5


def analyse_gap(
    objective: NeuralObjective,
    source: VariantScore,
    reference: VariantScore | None,
) -> ObjectiveGap:
    """How far the source sits from the reference, in the objective's own terms.

    The formula is direction-dependent and is recorded on the result, because
    a single "gap" number means three different things across MAXIMIZE,
    MINIMIZE and TARGET, and a reader cannot tell which without being told.
    """
    source_value = _raw(source, objective.objective_id)
    reference_value = _raw(reference, objective.objective_id) if reference else None
    reference_id = reference.variant_id if reference else None

    if source_value is None:
        return ObjectiveGap(
            objective_id=objective.objective_id,
            direction=objective.direction.value,
            source_value=None,
            reference_value=reference_value,
            reference_variant_id=reference_id,
            gap=None,
            formula="unavailable",
            note="the source variant has no measured value for this objective",
        )

    if objective.direction is ObjectiveDirection.TARGET:
        target = objective.normalization.target_value
        if target is None:
            return ObjectiveGap(
                objective_id=objective.objective_id,
                direction=objective.direction.value,
                source_value=source_value,
                reference_value=reference_value,
                reference_variant_id=reference_id,
                gap=None,
                formula="unavailable",
                note="a TARGET objective without a target value has no defined gap",
            )
        gap = abs(source_value - target)
        return ObjectiveGap(
            objective_id=objective.objective_id,
            direction=objective.direction.value,
            source_value=source_value,
            reference_value=target,
            reference_variant_id=None,
            gap=gap,
            relative_gap=(gap / abs(target)) if abs(target) > 1e-9 else None,
            formula="abs(source - target)",
            note="the reference for a TARGET objective is the target value, not a variant",
        )

    if reference_value is None:
        return ObjectiveGap(
            objective_id=objective.objective_id,
            direction=objective.direction.value,
            source_value=source_value,
            reference_value=None,
            reference_variant_id=reference_id,
            gap=None,
            formula="unavailable",
            note="no reference variant has a measured value for this objective",
        )

    if objective.direction is ObjectiveDirection.MAXIMIZE:
        gap = reference_value - source_value
        formula = "reference - source"
    else:
        gap = source_value - reference_value
        formula = "source - reference"

    denominator = abs(reference_value)
    return ObjectiveGap(
        objective_id=objective.objective_id,
        direction=objective.direction.value,
        source_value=source_value,
        reference_value=reference_value,
        reference_variant_id=reference_id,
        gap=gap,
        relative_gap=(gap / denominator) if denominator > 1e-9 else None,
        formula=formula,
        note=(
            None
            if gap > 0
            else "the source already matches or exceeds the reference on this objective"
        ),
    )


def detect_intervals(
    objective: NeuralObjective,
    source: VariantScore,
    reference: VariantScore | None = None,
) -> tuple[list[OptimizationInterval], list[str]]:
    """Weak and strong intervals for one objective. Returns (intervals, warnings)."""
    warnings: list[str] = []
    evaluation = _evaluation(source, objective.objective_id)
    if evaluation is None or not evaluation.valid:
        return [], [
            f"objective {objective.objective_id} has no valid evaluation for "
            f"{source.variant_id}, so no intervals can be located"
        ]

    contribution = evaluation.temporal_contribution
    if contribution is None:
        return [], [
            f"metric {objective.metric} exposes no temporal decomposition -- it is not a "
            f"sum over samples -- so intervals cannot be attributed for objective "
            f"{objective.objective_id}. Use a mean, peak or integrated objective to locate "
            f"where a score came from."
        ]

    times = np.asarray(contribution.times, dtype=np.float64)
    values = np.asarray(contribution.contributions, dtype=np.float64)
    magnitude = float(np.abs(values).sum())
    if magnitude <= 0:
        return [], [
            f"every sample contributed zero to objective {objective.objective_id}; there is "
            f"nothing to attribute"
        ]

    shares = np.abs(values) / magnitude
    even = 1.0 / len(shares)
    weak_mask = shares < even * WEAK_SHARE_RATIO
    strong_mask = shares > even * STRONG_SHARE_RATIO

    reference_shares = _reference_shares(reference, objective.objective_id)
    if len(shares) < 3:
        warnings.append(
            f"objective {objective.objective_id} resolved to only {len(shares)} sample(s); "
            f"interval detection over so few points is descriptive at best"
        )

    intervals: list[OptimizationInterval] = []
    for kind, mask, reason in (
        (
            IntervalKind.WEAK,
            weak_mask,
            f"contributed under {WEAK_SHARE_RATIO:.0%} of an even share of this objective",
        ),
        (
            IntervalKind.STRONG,
            strong_mask,
            f"contributed over {STRONG_SHARE_RATIO:.0f}x an even share of this objective",
        ),
    ):
        for index, group in enumerate(_groups(times, mask)):
            selected = np.asarray(group, dtype=np.int64)
            start = float(times[selected[0]])
            end = float(times[selected[-1]]) + _step(times)
            intervals.append(
                OptimizationInterval(
                    interval_id=f"{kind.value}-{objective.objective_id}-{index}",
                    kind=kind,
                    objective_id=objective.objective_id,
                    start_seconds=round(start, 3),
                    end_seconds=round(end, 3),
                    sample_count=int(selected.size),
                    contribution_share=round(float(shares[selected].sum()), 6),
                    contribution=round(float(values[selected].sum()), 6),
                    reference_contribution_share=_reference_share(
                        reference_shares, start, end
                    ),
                    reason=reason,
                    warnings=(
                        (
                            f"this interval spans {selected.size} prediction sample(s); "
                            f"below 2 it cannot support content evidence and no "
                            f"recommendation will be generated from it",
                        )
                        if selected.size < 2
                        else ()
                    ),
                )
            )
    return intervals, warnings


def _groups(times: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    """Contiguous runs of qualifying samples, merged across small gaps."""
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        if float(times[index]) - float(times[groups[-1][-1]]) <= MERGE_GAP_SECONDS:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return groups


def _step(times: np.ndarray) -> float:
    """The sampling period, so an interval covers its last sample's duration."""
    if times.size < 2:
        return 1.0
    return float(np.median(np.diff(times)))


def _reference_shares(
    reference: VariantScore | None, objective_id: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if reference is None:
        return None
    evaluation = _evaluation(reference, objective_id)
    if evaluation is None or evaluation.temporal_contribution is None:
        return None
    contribution = evaluation.temporal_contribution
    values = np.abs(np.asarray(contribution.contributions, dtype=np.float64))
    total = float(values.sum())
    if total <= 0:
        return None
    return np.asarray(contribution.times, dtype=np.float64), values / total


def _reference_share(
    shares: tuple[np.ndarray, np.ndarray] | None, start: float, end: float
) -> float | None:
    """The reference's share of the same wall-clock interval.

    Wall-clock rather than event-relative: this answers "what was the reference
    doing at the same moment", which is the comparison a reader expects when an
    interval is quoted in seconds.
    """
    if shares is None:
        return None
    times, values = shares
    window = (times >= start) & (times < end)
    if not window.any():
        return None
    return round(float(values[window].sum()), 6)


def _evaluation(variant: VariantScore, objective_id: str) -> ObjectiveEvaluation | None:
    return next(
        (e for e in variant.objective_scores if e.objective_id == objective_id), None
    )


def _raw(variant: VariantScore | None, objective_id: str) -> float | None:
    if variant is None:
        return None
    evaluation = _evaluation(variant, objective_id)
    return evaluation.raw_value if evaluation else None


def select_references(
    result: ExperimentScoreResult,
    source_variant_id: str,
    strategy: str,
    *,
    manual_ids: tuple[str, ...] = (),
    baseline_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Choose which variants to learn from. Returns (ids, warnings).

    Never implicit: a recommendation's meaning depends entirely on what it was
    compared against, so the strategy is an explicit request field and the
    chosen ids are recorded on the result.
    """
    warnings: list[str] = []
    others = [
        score.variant_id
        for score in result.variant_scores
        if score.variant_id != source_variant_id and score.total_score is not None
    ]
    if not others:
        return [], ["no other variant has a complete score, so there is nothing to learn from"]

    if strategy == "manual":
        missing = [item for item in manual_ids if item not in others]
        if missing:
            warnings.append(f"requested reference(s) not scored and ignored: {missing}")
        return [item for item in manual_ids if item in others], warnings

    if strategy == "baseline":
        if baseline_id is None or baseline_id not in others:
            return [], ["baseline reference requested but no scored baseline is available"]
        return [baseline_id], warnings

    if strategy == "pareto_front":
        if result.pareto is None:
            return [], ["Pareto reference requested but the score has no Pareto analysis"]
        front = [item for item in result.pareto.non_dominated if item in others]
        if not front:
            warnings.append("no other variant is on the Pareto front")
        return front, warnings

    # BEST_SCORE: every variant that outranks the source, best first.
    source = next(
        (s for s in result.variant_scores if s.variant_id == source_variant_id), None
    )
    if source is None or source.total_score is None:
        return [], ["the source variant has no complete score"]
    better = [
        score
        for score in result.variant_scores
        if score.variant_id in others
        and score.total_score is not None
        and score.total_score > source.total_score
    ]
    if not better:
        warnings.append(
            "no variant scores higher than the source on this objective set, so there is no "
            "better-performing reference to learn from"
        )
    better.sort(key=lambda item: -(item.total_score or 0.0))
    return [item.variant_id for item in better], warnings
