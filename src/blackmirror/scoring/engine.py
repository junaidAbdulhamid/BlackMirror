"""Goal-conditioned scoring across N variants.

THE ORDER OF OPERATIONS, AND WHY IT IS THIS ORDER
    1. Evaluate every (variant, objective) pair -> raw values only.
    2. Normalise per objective ACROSS variants -> normalized values.
    3. Apply direction -> per-objective scores.
    4. Weight and sum -> composite score per variant.
    5. Rank, compute contributions, Pareto, sensitivity.

    Step 2 is why scoring is an experiment-level operation and not a
    variant-level one: min-max, z-score and baseline strategies are all defined
    relative to the other variants, so a single variant has no score in
    isolation. Evaluation still works alone and yields a raw value, which is the
    quantity that genuinely does not depend on the comparison set.

WEIGHTS
    Weights are normalised to sum to 1 and the applied weights are persisted.
    A weight is a statement about what the user values; it is never inferred
    from the data, because inferring it would mean the engine choosing its own
    objective.

WHAT RANKING MEANS
    Ranking orders variants by the composite score of the objectives supplied.
    It is not a quality ordering. The margin between first and second is
    reported so a near-tie is visible rather than presented as a decision.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from blackmirror.scoring.evaluator import ObjectiveEvaluator, ScoringVariant
from blackmirror.scoring.metrics import MetricRegistry, default_registry
from blackmirror.scoring.normalization import normalize_and_score
from blackmirror.scoring.schemas import (
    SCORING_VERSION,
    ExperimentScoreResult,
    NeuralObjective,
    NormalizationStrategy,
    ObjectiveContribution,
    ObjectiveEvaluation,
    ParetoAnalysis,
    ScoringMetadata,
    SensitivityAnalysis,
    VariantScore,
    WeightSensitivityPoint,
)

#: Below this, a first-vs-second gap is reported as effectively a tie.
CLOSE_MARGIN = 0.01


def objective_set_hash(
    objectives: tuple[NeuralObjective, ...], *, cache_context: object | None = None
) -> str:
    """Stable identity for a set of objectives, for caching and reproducibility."""
    payload = [
        objective.model_dump(mode="json", exclude={"provenance"}) for objective in objectives
    ]
    encoded = json.dumps(
        {"objectives": payload, "cache_context": cache_context},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalized_weights(
    objectives: tuple[NeuralObjective, ...],
) -> tuple[dict[str, float], list[str]]:
    """Weights scaled to sum to 1, with what was done reported."""
    warnings: list[str] = []
    raw = {objective.objective_id: float(objective.weight) for objective in objectives}
    total = sum(raw.values())
    if total <= 0:
        equal = 1.0 / len(raw) if raw else 0.0
        warnings.append(
            "all objective weights were zero; falling back to equal weights so the "
            "composite is defined"
        )
        return dict.fromkeys(raw, equal), warnings
    if abs(total - 1.0) > 1e-9:
        warnings.append(
            f"objective weights summed to {total:g} and were rescaled to 1; the applied "
            f"weights are recorded on each contribution"
        )
    return {key: value / total for key, value in raw.items()}, warnings


class GoalConditionedScoringEngine:
    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or default_registry()
        self.evaluator = ObjectiveEvaluator(self.registry)

    def score_experiment(
        self,
        experiment_id: str,
        variants: tuple[ScoringVariant, ...],
        objectives: tuple[NeuralObjective, ...],
        *,
        baseline_variant_id: str | None = None,
        sensitivity_steps: int = 21,
        cache_key: str | None = None,
        input_artifact_sha256: dict[str, str] | None = None,
    ) -> ExperimentScoreResult:
        if not variants:
            raise ValueError("scoring needs at least one variant")
        if not objectives:
            raise ValueError("scoring needs at least one objective")
        ids = [variant.variant_id for variant in variants]
        if len(set(ids)) != len(ids):
            raise ValueError("variant ids must be unique")
        if baseline_variant_id is not None and baseline_variant_id not in ids:
            raise ValueError(f"baseline variant {baseline_variant_id!r} is not among the variants")
        if len({objective.objective_id for objective in objectives}) != len(objectives):
            raise ValueError("objective ids must be unique")
        first = variants[0]
        for variant in variants[1:]:
            if (variant.atlas_name, variant.atlas_version) != (
                first.atlas_name,
                first.atlas_version,
            ):
                raise ValueError("variants use incompatible anatomical atlases")
            if variant.responses.shape[1] != first.responses.shape[1]:
                raise ValueError("variants use incompatible cortical vertex counts")
            if variant.model_fingerprint != first.model_fingerprint:
                raise ValueError("variants use incompatible model fingerprints")
            if variant.analytics_version != first.analytics_version:
                raise ValueError("variants use incompatible analytics versions")
            if variant.network_mapping_sha256 != first.network_mapping_sha256:
                raise ValueError("variants use incompatible functional-network mappings")

        warnings: list[str] = []
        weights, weight_warnings = normalized_weights(objectives)
        warnings.extend(weight_warnings)
        baseline = next(
            (v for v in variants if v.variant_id == baseline_variant_id), None
        )

        # 1. Raw evaluation.
        evaluations: dict[str, dict[str, ObjectiveEvaluation]] = {}
        for objective in objectives:
            evaluations[objective.objective_id] = {
                variant.variant_id: self.evaluator.evaluate(
                    variant, objective, baseline=baseline
                )
                for variant in variants
            }

        # 2-3. Normalise across variants, then apply direction.
        normalization_detail: dict[str, dict[str, float]] = {}
        scores: dict[str, dict[str, float | None]] = {}
        normalized: dict[str, dict[str, float | None]] = {}
        for objective in objectives:
            per_variant = evaluations[objective.objective_id]
            outcome = normalize_and_score(
                objective,
                {vid: ev.raw_value for vid, ev in per_variant.items()},
                baseline_variant_id=baseline_variant_id,
            )
            scores[objective.objective_id] = outcome.scores
            normalized[objective.objective_id] = outcome.normalized
            normalization_detail[objective.objective_id] = outcome.detail
            warnings.extend(
                f"[{objective.objective_id}] {message}" for message in outcome.warnings
            )

        # 4. Weighted composite.
        variant_scores = [
            self._variant_score(variant, objectives, evaluations, normalized, scores, weights)
            for variant in variants
        ]
        variant_scores = self._attach_baseline(variant_scores, baseline_variant_id, warnings)
        effective = effective_weights(variant_scores)
        warnings.extend(_scale_warnings(objectives, weights, effective))
        ordered = self._rank(variant_scores)

        margin = None
        scored = [item for item in ordered if item.total_score is not None]
        if len(scored) >= 2:
            margin = float(scored[0].total_score - scored[1].total_score)  # type: ignore[operator]
            if abs(margin) < CLOSE_MARGIN:
                warnings.append(
                    f"the top two variants are separated by {margin:.6g} on the composite "
                    f"score; this is close enough that the ordering should not be treated "
                    f"as a meaningful difference"
                )

        pareto = analyze_pareto(objectives, scores) if len(objectives) > 1 else None
        sensitivity = (
            tuple(
                sweep_weight(objectives, scores, target_id=objective.objective_id,
                             steps=sensitivity_steps)
                for objective in objectives
            )
            if len(objectives) > 1
            else ()
        )

        return ExperimentScoreResult(
            experiment_id=experiment_id,
            objectives=objectives,
            baseline_variant_id=baseline_variant_id,
            variant_scores=tuple(ordered),
            ranking=tuple(item.variant_id for item in ordered if item.total_score is not None),
            ranking_margin=margin,
            pareto=pareto,
            sensitivity=sensitivity,
            normalization_detail=normalization_detail,
            effective_weights=effective,
            metadata=ScoringMetadata(
                scoring_version=SCORING_VERSION,
                analytics_versions={
                    variant.variant_id: variant.analytics_version or "unknown"
                    for variant in variants
                },
                atlas_name=variants[0].atlas_name,
                atlas_version=variants[0].atlas_version,
                network_mapping_sha256=variants[0].network_mapping_sha256,
                objective_definition_hash=objective_set_hash(objectives),
                objective_set_hash=cache_key or objective_set_hash(objectives),
                input_artifact_sha256=input_artifact_sha256 or {},
                metric_descriptions={
                    objective.metric: self.registry.get(objective.metric).describe()
                    for objective in objectives
                },
            ),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _variant_score(
        variant: ScoringVariant,
        objectives: tuple[NeuralObjective, ...],
        evaluations: dict[str, dict[str, ObjectiveEvaluation]],
        normalized: dict[str, dict[str, float | None]],
        scores: dict[str, dict[str, float | None]],
        weights: dict[str, float],
    ) -> VariantScore:
        per_objective: list[ObjectiveEvaluation] = []
        contributions: list[ObjectiveContribution] = []
        total = 0.0
        complete = True
        warnings: list[str] = []

        for objective in objectives:
            oid = objective.objective_id
            evaluation = evaluations[oid][variant.variant_id]
            score = scores[oid][variant.variant_id]
            per_objective.append(
                evaluation.model_copy(
                    update={
                        "normalized_value": normalized[oid][variant.variant_id],
                        "score": score,
                    }
                )
            )
            if score is None:
                complete = False
                warnings.append(
                    f"objective {oid} produced no score for this variant, so no composite "
                    f"score is reported; a partial sum would rank it against variants that "
                    f"were measured on more objectives"
                )
                continue
            contribution = weights[oid] * score
            total += contribution
            contributions.append(
                ObjectiveContribution(
                    objective_id=oid, weight=weights[oid], score=score, contribution=contribution
                )
            )

        if complete and contributions:
            denominator = sum(abs(item.contribution) for item in contributions)
            contributions = [
                item.model_copy(
                    update={
                        "contribution_fraction": (
                            abs(item.contribution) / denominator if denominator > 0 else None
                        )
                    }
                )
                for item in contributions
            ]

        return VariantScore(
            variant_id=variant.variant_id,
            total_score=total if complete else None,
            objective_scores=tuple(per_objective),
            contributions=tuple(contributions),
            valid=complete,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _attach_baseline(
        variant_scores: list[VariantScore],
        baseline_variant_id: str | None,
        warnings: list[str],
    ) -> list[VariantScore]:
        if baseline_variant_id is None:
            return variant_scores
        base = next(
            (item for item in variant_scores if item.variant_id == baseline_variant_id), None
        )
        if base is None or base.total_score is None:
            warnings.append(
                "the declared baseline has no composite score, so baseline deltas are omitted"
            )
            return variant_scores
        base_total = base.total_score
        updated = []
        for item in variant_scores:
            if item.total_score is None:
                updated.append(item)
                continue
            delta = item.total_score - base_total
            # A percentage against a near-zero baseline is not a number anyone
            # should read, so it is withheld rather than printed as huge.
            relative = delta / abs(base_total) if abs(base_total) > 1e-9 else None
            updated.append(
                item.model_copy(
                    update={"baseline_delta": delta, "baseline_relative_delta": relative}
                )
            )
        return updated

    @staticmethod
    def _rank(variant_scores: list[VariantScore]) -> list[VariantScore]:
        scored = sorted(
            (item for item in variant_scores if item.total_score is not None),
            key=lambda item: (-item.total_score, item.variant_id),  # type: ignore[operator]
        )
        unscored = sorted(
            (item for item in variant_scores if item.total_score is None),
            key=lambda item: item.variant_id,
        )
        ranked = [
            item.model_copy(update={"rank": position})
            for position, item in enumerate(scored, start=1)
        ]
        return ranked + list(unscored)


#: How far an objective's realised share may drift from its stated weight
#: before the composite is reported as not honouring its own weighting.
WEIGHT_DRIFT_TOLERANCE = 0.15


def effective_weights(variant_scores: list[VariantScore]) -> dict[str, float]:
    """The share of composite magnitude each objective actually accounted for.

    A stated weight is only the real weight when every objective's score lives on
    a comparable scale. Under `NONE` normalisation the raw metrics keep their own
    units, so an objective measured in unit-seconds (INTEGRATED_RESPONSE) swamps
    one measured in units (MEAN_RESPONSE) whatever the weights say. Measured: a
    declared 50/50 split behaved as 5/95.
    """
    totals: dict[str, list[float]] = {}
    for score in variant_scores:
        if score.total_score is None or not score.contributions:
            continue
        magnitude = sum(abs(item.contribution) for item in score.contributions)
        if magnitude <= 0:
            continue
        for item in score.contributions:
            totals.setdefault(item.objective_id, []).append(abs(item.contribution) / magnitude)
    return {
        objective_id: float(np.mean(shares)) for objective_id, shares in sorted(totals.items())
    }


def _scale_warnings(
    objectives: tuple[NeuralObjective, ...],
    stated: dict[str, float],
    effective: dict[str, float],
) -> list[str]:
    """Report a composite whose weights are not the weights actually applied."""
    warnings: list[str] = []
    if len(objectives) < 2 or not effective:
        return warnings

    unnormalised = [
        objective
        for objective in objectives
        if objective.normalization.strategy is NormalizationStrategy.NONE
    ]
    metrics = {objective.metric for objective in objectives}
    if len(unnormalised) > 1 and len(metrics) > 1:
        warnings.append(
            "this composite sums objectives that use different metrics without "
            "normalising them, so the objectives are being added in different units. "
            "The stated weights are not the weights that will apply. Use a normalization "
            "strategy, or compare the objectives separately."
        )

    for objective_id, share in effective.items():
        drift = share - stated.get(objective_id, 0.0)
        if abs(drift) > WEIGHT_DRIFT_TOLERANCE:
            warnings.append(
                f"objective {objective_id} was given weight "
                f"{stated.get(objective_id, 0.0):.2f} but accounted for {share:.0%} of the "
                f"composite score's magnitude. The composite does not reflect the stated "
                f"weighting; read the per-objective scores."
            )
    return warnings


def analyze_pareto(
    objectives: tuple[NeuralObjective, ...],
    scores: dict[str, dict[str, float | None]],
) -> ParetoAnalysis:
    """Non-dominated variants, using per-objective scores where higher is better."""
    objective_ids = tuple(objective.objective_id for objective in objectives)
    variant_ids = sorted({vid for table in scores.values() for vid in table})
    usable = [
        vid
        for vid in variant_ids
        if all(scores[oid].get(vid) is not None for oid in objective_ids)
    ]
    warnings: list[str] = []
    if len(usable) < len(variant_ids):
        warnings.append(
            f"{len(variant_ids) - len(usable)} variant(s) were excluded from the Pareto "
            f"analysis because they lack a score on at least one objective"
        )

    dominated_by: dict[str, tuple[str, ...]] = {}
    for candidate in usable:
        dominators = []
        for other in usable:
            if other == candidate:
                continue
            at_least_as_good = all(
                scores[oid][other] >= scores[oid][candidate]  # type: ignore[operator]
                for oid in objective_ids
            )
            strictly_better = any(
                scores[oid][other] > scores[oid][candidate]  # type: ignore[operator]
                for oid in objective_ids
            )
            if at_least_as_good and strictly_better:
                dominators.append(other)
        if dominators:
            dominated_by[candidate] = tuple(dominators)

    dominated = tuple(sorted(dominated_by))
    return ParetoAnalysis(
        objective_ids=objective_ids,
        non_dominated=tuple(v for v in usable if v not in dominated_by),
        dominated=dominated,
        dominated_by=dominated_by,
        warnings=tuple(warnings),
    )


def sweep_weight(
    objectives: tuple[NeuralObjective, ...],
    scores: dict[str, dict[str, float | None]],
    *,
    target_id: str,
    steps: int = 21,
) -> SensitivityAnalysis:
    """Vary one objective's weight from 0 to 1 and watch the winner.

    The remaining weight is split among the other objectives in their existing
    proportions, so the sweep changes one dimension rather than silently
    reshaping the whole weighting.
    """
    others = [o for o in objectives if o.objective_id != target_id]
    other_total = sum(o.weight for o in others)
    variant_ids = sorted(
        vid for vid in {v for table in scores.values() for v in table}
        if all(scores[o.objective_id].get(vid) is not None for o in objectives)
    )
    if len(variant_ids) < 2:
        return SensitivityAnalysis(
            swept_objective_id=target_id,
            points=(),
            stable=True,
            note="fewer than two fully scored variants; a weight sweep cannot change a winner",
        )

    points: list[WeightSensitivityPoint] = []
    for weight in np.linspace(0.0, 1.0, steps):
        applied = {target_id: float(weight)}
        for other in others:
            share = (other.weight / other_total) if other_total > 0 else 1.0 / max(1, len(others))
            applied[other.objective_id] = float((1.0 - weight) * share)
        totals: dict[str, float] = {}
        for vid in variant_ids:
            running = 0.0
            for oid, share in applied.items():
                value = scores[oid][vid]
                assert value is not None  # variant_ids filtered to fully scored
                running += share * value
            totals[vid] = running
        ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 0.0
        points.append(
            WeightSensitivityPoint(
                weight=round(float(weight), 6),
                winner_variant_id=ordered[0][0],
                margin=float(margin),
            )
        )

    flips = tuple(
        second.weight
        for first, second in itertools.pairwise(points)
        if first.winner_variant_id != second.winner_variant_id
    )
    return SensitivityAnalysis(
        swept_objective_id=target_id,
        points=tuple(points),
        flip_weights=flips,
        stable=not flips,
        note=(
            None
            if not flips
            else (
                f"the top-ranked variant changes at weight(s) {[round(f, 3) for f in flips]}; "
                f"this ranking depends on the weighting, not only on the variants"
            )
        ),
    )
