"""Direction-aware Phase 8 outcome measurement."""

from __future__ import annotations

from blackmirror.resimulation.schemas import ObjectiveDelta, Outcome
from blackmirror.scoring.schemas import ExperimentScoreResult, ObjectiveDirection


def evaluate_objectives(
    score: ExperimentScoreResult,
    *,
    parent_run_id: str,
    candidate_run_id: str,
    tolerance: float,
) -> tuple[ObjectiveDelta, ...]:
    variants = {item.variant_id: item for item in score.variant_scores}
    if parent_run_id not in variants or candidate_run_id not in variants:
        raise ValueError("score does not contain both parent and candidate")
    objectives = {item.objective_id: item for item in score.objectives}
    parent = {item.objective_id: item for item in variants[parent_run_id].objective_scores}
    candidate = {item.objective_id: item for item in variants[candidate_run_id].objective_scores}
    output = []
    for objective_id, objective in objectives.items():
        left = parent[objective_id]
        right = candidate[objective_id]
        raw_delta = _delta(left.raw_value, right.raw_value)
        score_delta = _delta(left.score, right.score)
        if not left.valid or not right.valid or raw_delta is None:
            outcome = Outcome.INCONCLUSIVE
        else:
            preference = raw_delta
            if objective.direction is ObjectiveDirection.MINIMIZE:
                preference = -raw_delta
            elif objective.direction is ObjectiveDirection.TARGET:
                target = objective.normalization.target_value
                assert target is not None
                preference = abs(left.raw_value - target) - abs(right.raw_value - target)  # type: ignore[operator]
            outcome = (
                Outcome.IMPROVED
                if preference > tolerance
                else Outcome.WORSENED
                if preference < -tolerance
                else Outcome.UNCHANGED
            )
        output.append(
            ObjectiveDelta(
                objective_id=objective_id,
                direction=objective.direction.value,
                parent_raw_value=left.raw_value,
                candidate_raw_value=right.raw_value,
                raw_delta=raw_delta,
                parent_score=left.score,
                candidate_score=right.score,
                score_delta=score_delta,
                outcome=outcome,
            )
        )
    return tuple(output)


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else right - left
