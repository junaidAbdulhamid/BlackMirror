"""Placing raw metric values on a comparable scale, without inflating them.

THE CENTRAL HAZARD
    Min-max normalisation across an experiment maps the worst variant to 0 and
    the best to 1 *by construction*, whatever the actual spread. With A = 0.401
    and B = 0.402 it reports 0.0 and 1.0: a 0.25% difference rendered as total
    dominance. The normaliser cannot tell that this happened, because the
    arithmetic is identical whether the spread is 0.001 or 10.

    Three things are done about it. The raw value is always preserved and
    displayed beside the score. A relative-spread warning is attached whenever
    the range is negligible compared with the values themselves. And min-max is
    NOT the default.

WHY NONE IS THE DEFAULT
    A raw metric already lives on a meaningful scale in model units. Normalising
    is a presentation decision, and a scoring engine should not make that
    decision silently on the user's behalf. `NONE` passes the raw value through
    as the score (after direction), so the default behaviour is the honest one.

DIRECTION IS APPLIED LAST
    Normalisation answers "where does this sit among the variants?".
    Direction answers "is higher preferable?". Keeping them separate means a
    MINIMIZE objective is not implemented by negating a metric somewhere deep
    in the stack, where the sign would be easy to lose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackmirror.scoring.schemas import (
    NeuralObjective,
    NormalizationStrategy,
    ObjectiveDirection,
)

#: Below this ratio of (range / typical magnitude), min-max is reporting noise
#: as if it were signal.
NEGLIGIBLE_SPREAD_RATIO = 0.01


@dataclass(frozen=True)
class NormalizationOutcome:
    normalized: dict[str, float | None]
    scores: dict[str, float | None]
    warnings: tuple[str, ...]
    detail: dict[str, float]


def normalize_and_score(
    objective: NeuralObjective,
    raw_by_variant: dict[str, float | None],
    *,
    baseline_variant_id: str | None = None,
) -> NormalizationOutcome:
    """Normalise every variant's raw value, then apply direction to get scores."""
    usable = {k: v for k, v in raw_by_variant.items() if v is not None and np.isfinite(v)}
    warnings: list[str] = []
    detail: dict[str, float] = {}

    if not usable:
        return NormalizationOutcome(
            normalized=dict.fromkeys(raw_by_variant),
            scores=dict.fromkeys(raw_by_variant),
            warnings=("no variant produced a finite value for this objective",),
            detail={},
        )

    strategy = objective.normalization.strategy
    values = np.array(list(usable.values()), dtype=np.float64)

    if strategy is NormalizationStrategy.NONE:
        normalized = dict(usable)

    elif strategy is NormalizationStrategy.MIN_MAX_WITHIN_EXPERIMENT:
        low, high = float(values.min()), float(values.max())
        span = high - low
        detail |= {"min": low, "max": high, "span": span}
        if span <= 0:
            warnings.append(
                "every variant produced the same value; min-max is undefined, so all "
                "normalized values are 0.5"
            )
            normalized = dict.fromkeys(usable, 0.5)
        else:
            magnitude = float(np.abs(values).max())
            if magnitude > 0 and span / magnitude < NEGLIGIBLE_SPREAD_RATIO:
                warnings.append(
                    f"min-max stretched a spread of {span:.6g} across values of magnitude "
                    f"~{magnitude:.6g} ({span / magnitude:.2%}) to the full 0-1 range. The "
                    f"normalized scores will look decisive; the raw values are nearly "
                    f"identical. Read the raw values."
                )
            normalized = {k: (v - low) / span for k, v in usable.items()}

    elif strategy is NormalizationStrategy.Z_SCORE_WITHIN_EXPERIMENT:
        mean = float(values.mean())
        # Population sd: these are all the variants there are, not a sample.
        spread = float(values.std(ddof=0))
        detail |= {"mean": mean, "std": spread}
        if spread <= 0:
            warnings.append("zero variance across variants; all z-scores are 0")
            normalized = dict.fromkeys(usable, 0.0)
        else:
            if len(usable) < 3:
                warnings.append(
                    f"a z-score over {len(usable)} variant(s) is not a meaningful "
                    f"standardisation; with two variants it always yields -1 and +1"
                )
            normalized = {k: (v - mean) / spread for k, v in usable.items()}

    elif strategy is NormalizationStrategy.REFERENCE_BASELINE:
        if baseline_variant_id is None or baseline_variant_id not in usable:
            return NormalizationOutcome(
                normalized=dict.fromkeys(raw_by_variant),
                scores=dict.fromkeys(raw_by_variant),
                warnings=(
                    "reference-baseline normalization needs a declared baseline variant "
                    "with a finite value for this objective",
                ),
                detail={},
            )
        base = usable[baseline_variant_id]
        detail["baseline"] = base
        if abs(base) < 1e-12:
            warnings.append(
                "the baseline value is at or near zero, so a ratio is undefined; the "
                "difference from baseline is reported instead of a relative change"
            )
            normalized = {k: v - base for k, v in usable.items()}
        else:
            normalized = {k: (v - base) / abs(base) for k, v in usable.items()}

    elif strategy is NormalizationStrategy.ROBUST_PERCENTILE:
        low, high = (float(x) for x in np.percentile(values, [10, 90]))
        detail |= {"p10": low, "p90": high}
        span = high - low
        if span <= 0:
            warnings.append("the 10th and 90th percentiles coincide; falling back to 0.5")
            normalized = dict.fromkeys(usable, 0.5)
        else:
            # Clipped, so an outlier cannot push another variant off the scale.
            normalized = {k: float(np.clip((v - low) / span, 0.0, 1.0)) for k, v in usable.items()}

    else:  # TARGET_DISTANCE
        target = objective.normalization.target_value
        tolerance = objective.normalization.target_tolerance
        assert target is not None and tolerance is not None
        detail |= {"target": target, "tolerance": tolerance}
        # 1 at the target, decaying to 0 at one tolerance away, then clamped.
        normalized = {
            k: float(max(0.0, 1.0 - abs(v - target) / tolerance)) for k, v in usable.items()
        }

    scores = _apply_direction(objective, normalized, warnings)
    return NormalizationOutcome(
        normalized={k: normalized.get(k) for k in raw_by_variant},
        scores={k: scores.get(k) for k in raw_by_variant},
        warnings=tuple(warnings),
        detail=detail,
    )


def _apply_direction(
    objective: NeuralObjective,
    normalized: dict[str, float],
    warnings: list[str],
) -> dict[str, float]:
    """Turn a normalized value into a score where higher is always preferable."""
    direction = objective.direction
    strategy = objective.normalization.strategy

    if direction is ObjectiveDirection.MAXIMIZE:
        return dict(normalized)

    if direction is ObjectiveDirection.MINIMIZE:
        # Negation, not reciprocal: it preserves spacing between variants, which
        # a reciprocal would distort, and it cannot divide by zero.
        return {k: -v for k, v in normalized.items()}

    # TARGET: closeness to the requested value, independent of which side.
    if strategy is NormalizationStrategy.TARGET_DISTANCE:
        # Already a closeness in [0, 1]; higher is nearer the target.
        return dict(normalized)
    target = objective.normalization.target_value
    tolerance = objective.normalization.target_tolerance
    assert target is not None and tolerance is not None
    warnings.append(
        f"direction TARGET with {strategy.value} normalization: closeness is computed on "
        f"the normalized values, so the target {target:g} is interpreted on that scale"
    )
    return {k: float(max(0.0, 1.0 - abs(v - target) / tolerance)) for k, v in normalized.items()}
