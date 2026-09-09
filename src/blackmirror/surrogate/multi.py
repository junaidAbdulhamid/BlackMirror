"""Steps 41-43: surrogates over several objectives at once.

WHY ONE MODEL PER OBJECTIVE
    A multi-output regressor shares structure across targets, which helps when
    the objectives are correlated and hurts when they are not. At these sample
    sizes there is no way to establish which case holds, so the safer choice is
    independent models: each objective gets its own surrogate, its own
    validation and its own trust verdict. One objective being unlearnable then
    degrades that objective rather than quietly contaminating the others.

WHY SCALARIZATION RATHER THAN EXPECTED HYPERVOLUME
    Step 42 warns against reaching for EHVI without justification, and the
    warning is right. Hypervolume-based acquisition needs a reference point, a
    reliable frontier estimate, and enough observations for the frontier to mean
    something. With a handful of evaluations it would be fitting a shape to
    noise and reporting it with confidence.

    Random scalarization is the honest alternative. Each round draws a weight
    vector, optimises the weighted sum under those weights, and moves on. Over
    rounds the weights sweep the trade-off space, so the observations spread
    along the frontier instead of piling up wherever a fixed weighting happened
    to point. The frontier is then reported from the *real* observations by
    Phase 9's Pareto machinery, never from predictions.

WHAT IS NEVER DONE HERE
    A predicted frontier is not reported as a frontier. Predictions steer which
    candidate is measured; the Pareto set shown to a person is computed from
    measured values only.
"""

from __future__ import annotations

from random import Random

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from blackmirror.scoring.schemas import NeuralObjective
from blackmirror.surrogate.encoder import GenomeFeatureEncoder
from blackmirror.surrogate.schemas import (
    SurrogateDataset,
    SurrogateModelType,
    SurrogatePrediction,
    SurrogateTrainingRecord,
)
from blackmirror.surrogate.trainer import SurrogateTrainer, TrainedSurrogate, TrustPolicy
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class ScalarizationWeights(BaseModel):
    """One draw of the trade-off, recorded so a round can be explained."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    weights: dict[str, float] = Field(min_length=1)
    round_number: int = Field(default=0, ge=0)

    def describe(self) -> str:
        parts = ", ".join(
            f"{name} {weight:.2f}" for name, weight in sorted(self.weights.items())
        )
        return f"weights this round: {parts}"


def sample_weights(
    objective_ids: list[str], rng: Random, *, round_number: int = 0
) -> ScalarizationWeights:
    """Step 43. A weight vector on the simplex, drawn uniformly.

    Uniform on the simplex rather than uniform per-coordinate-then-normalised,
    which concentrates near the centre and would under-sample the extremes where
    the interesting trade-offs sit. Exponential draws normalised give the
    correct uniform distribution.
    """
    draws = [rng.expovariate(1.0) for _ in objective_ids]
    total = sum(draws) or 1.0
    return ScalarizationWeights(
        weights={name: value / total for name, value in zip(objective_ids, draws, strict=True)},
        round_number=round_number,
    )


class MultiObjectiveSurrogate:
    """One trained surrogate per objective, sharing an encoder."""

    def __init__(
        self,
        encoder: GenomeFeatureEncoder,
        objectives: tuple[NeuralObjective, ...],
        *,
        model_type: SurrogateModelType = SurrogateModelType.GAUSSIAN_PROCESS,
        policy: TrustPolicy | None = None,
        seed: int = 0,
    ) -> None:
        if not objectives:
            raise ValueError("a multi-objective surrogate needs at least one objective")
        self.encoder = encoder
        self.objectives = objectives
        self.trainers: dict[str, SurrogateTrainer] = {
            objective.objective_id: SurrogateTrainer(
                encoder, model_type=model_type, policy=policy, seed=seed
            )
            for objective in objectives
        }
        self.trained: dict[str, TrainedSurrogate] = {}

    @property
    def objective_ids(self) -> list[str]:
        return [objective.objective_id for objective in self.objectives]

    # --- fitting ----------------------------------------------------------

    def fit(self, dataset: SurrogateDataset) -> dict[str, TrainedSurrogate]:
        """Fit one model per objective, each on its own directional target.

        A record missing an objective is dropped from *that* objective's
        training set only. Dropping it everywhere would discard measurements
        that are perfectly good for the objectives they do carry.
        """
        self.trained = {}
        for objective in self.objectives:
            per_objective = self._dataset_for(dataset, objective)
            if per_objective.size < 2:
                logger.info(
                    "objective %s has %d usable observations; not fitting",
                    objective.objective_id,
                    per_objective.size,
                )
                continue
            self.trained[objective.objective_id] = self.trainers[
                objective.objective_id
            ].fit(per_objective)
        return self.trained

    def _dataset_for(
        self, dataset: SurrogateDataset, objective: NeuralObjective
    ) -> SurrogateDataset:
        from blackmirror.search.fitness import directional_fitness

        records: list[SurrogateTrainingRecord] = []
        for record in dataset.records:
            raw = record.objective_values.get(objective.objective_id)
            if raw is None:
                continue
            folded = directional_fitness(raw, objective)
            if folded is None:
                continue
            records.append(
                record.model_copy(
                    update={
                        "directional_target": folded,
                        "primary_objective_id": objective.objective_id,
                    }
                )
            )
        return dataset.model_copy(update={"records": tuple(records)})

    # --- prediction -------------------------------------------------------

    @property
    def trusted_objectives(self) -> list[str]:
        return [
            objective_id
            for objective_id, trained in self.trained.items()
            if trained.usable
        ]

    @property
    def all_trusted(self) -> bool:
        return len(self.trusted_objectives) == len(self.objectives)

    def predict(
        self, candidates: dict[str, dict[str, float]]
    ) -> dict[str, list[SurrogatePrediction]]:
        """Per-objective estimates for every candidate."""
        return {
            objective_id: self.trainers[objective_id].predict(candidates)
            for objective_id in self.trained
            if self.trainers[objective_id].is_fitted
        }

    def scalarized(
        self,
        candidates: dict[str, dict[str, float]],
        weights: ScalarizationWeights,
    ) -> list[SurrogatePrediction]:
        """Weighted-sum predictions, for a single-objective acquisition.

        Objectives are standardised before weighting. Without that the weights
        would be dominated by whichever objective happens to have the larger
        numeric range, so a "50/50" weighting would not be one, and the sweep
        across rounds would explore a much narrower slice of the trade-off than
        it appears to.

        Uncertainties combine as the square root of the weighted sum of
        variances, which is correct if the objectives' errors are independent.
        They may not be, so this is a working assumption and is documented as
        one rather than presented as a derivation.
        """
        per_objective = self.predict(candidates)
        if not per_objective:
            return []
        ids = list(candidates)
        totals = np.zeros(len(ids), dtype=float)
        variances = np.zeros(len(ids), dtype=float)
        have_uncertainty = True
        used = 0.0

        for objective_id, predictions in per_objective.items():
            weight = weights.weights.get(objective_id, 0.0)
            if weight <= 0:
                continue
            used += weight
            by_id = {item.candidate_id: item for item in predictions}
            means = np.asarray([by_id[cid].predicted_score for cid in ids], dtype=float)
            spread = float(means.std()) or 1.0
            totals += weight * (means - float(means.mean())) / spread
            sigmas = [by_id[cid].uncertainty for cid in ids]
            if any(value is None for value in sigmas):
                have_uncertainty = False
            else:
                scaled = np.asarray(sigmas, dtype=float) / spread
                variances += (weight * scaled) ** 2

        if used == 0:  # pragma: no cover - weights are drawn on the simplex
            return []

        reference = next(iter(per_objective.values()))
        by_id = {item.candidate_id: item for item in reference}
        combined = np.sqrt(variances) if have_uncertainty else None
        return [
            SurrogatePrediction(
                candidate_id=cid,
                genome=candidates[cid],
                predicted_score=float(totals[index]),
                uncertainty=None if combined is None else float(combined[index]),
                ood_score=by_id[cid].ood_score,
                model_type=by_id[cid].model_type,
                model_version=by_id[cid].model_version,
                training_dataset_size=by_id[cid].training_dataset_size,
                training_dataset_hash=by_id[cid].training_dataset_hash,
                reason=weights.describe(),
            )
            for index, cid in enumerate(ids)
        ]

    def diagnostics(self) -> dict[str, object]:
        return {
            objective_id: {
                "trusted": trained.trusted,
                "reason": trained.trust_reason,
                "state": trained.state.value,
                "spearman": trained.prefix_validation.spearman
                or trained.cross_validation.spearman,
                "samples": trained.metadata.training_dataset_size,
            }
            for objective_id, trained in self.trained.items()
        }


def observed_pareto_front(results: object, objectives: tuple[NeuralObjective, ...]) -> object:
    """The frontier from *measured* values, never from predictions.

    A thin pass-through to Phase 9's implementation, kept here so the intent is
    unmistakable at the call site: what a person is shown as the trade-off set
    comes from the real evaluator.
    """
    from blackmirror.search.pareto import pareto_front

    return pareto_front(results, objectives)  # type: ignore[arg-type]


__all__ = [
    "MultiObjectiveSurrogate",
    "ScalarizationWeights",
    "observed_pareto_front",
    "sample_weights",
]
