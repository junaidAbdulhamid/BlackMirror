"""Steps 29-37, 46, 49-52: the surrogate-assisted optimization loop.

THE LOOP
    bootstrap with real evaluations
      -> train surrogate on real observations only
      -> generate a large cheap pool
      -> predict score and uncertainty for all of it
      -> score by acquisition, pick a diverse batch
      -> spend a real evaluation on the winner
      -> record predicted against actual
      -> retrain
      -> repeat

WHAT MAKES THIS DIFFERENT FROM PHASE 9
    Phase 9's strategies propose candidates from the shape of the space and what
    they have seen. This proposes them from a *model of the objective*, which is
    what lets it screen thousands of points for the cost of screening none. The
    expensive evaluator is unchanged and still authoritative; the surrogate only
    decides what to send it.

THE TWO THINGS THIS REFUSES TO DO
    It never trains on its own predictions, and it never lets an untrusted
    surrogate choose. The second is the important one: when validation says the
    model cannot rank, selection falls back to a Phase 9 strategy and the round
    is recorded as a fallback. A search that quietly kept using a broken model
    would spend hours confirming its own errors.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from random import Random

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from blackmirror.scoring.schemas import NeuralObjective
from blackmirror.search.budget import BudgetLedger, SearchBudget, admit
from blackmirror.search.fitness import CandidateFitness, best_of, directional_fitness
from blackmirror.search.genome import CandidateGenome, genome_distance
from blackmirror.search.space import ContentSearchSpace
from blackmirror.surrogate.acquisition import (
    AcquisitionName,
    AcquisitionRegistry,
    AcquisitionScore,
    score_candidates,
    select_diverse_batch,
)
from blackmirror.surrogate.encoder import GenomeFeatureEncoder
from blackmirror.surrogate.models import SurrogateError
from blackmirror.surrogate.pool import CandidatePoolGenerator, PoolConfig
from blackmirror.surrogate.schemas import (
    DatasetIdentity,
    SurrogateDataset,
    SurrogateState,
    SurrogateTrainingRecord,
)
from blackmirror.surrogate.trainer import SurrogateTrainer, TrainedSurrogate, TrustPolicy
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

Evaluate = Callable[[CandidateGenome], CandidateFitness]
Fallback = Callable[[int], list[CandidateGenome]]


class BayesianOptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    #: Step 30. Real evaluations spent covering the space before the model runs.
    bootstrap_evaluations: int = Field(default=8, ge=2, le=1000)
    acquisition: AcquisitionName = AcquisitionName.EXPECTED_IMPROVEMENT
    #: Step 33. Exploration controls, exposed rather than buried.
    kappa: float = Field(default=2.0, ge=0.0)
    xi: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=1, ge=1, le=32)
    minimum_batch_distance: float = Field(default=0.05, ge=0.0, le=1.0)
    pool: PoolConfig = PoolConfig()
    random_seed: int = Field(default=0, ge=0)
    #: Step 49. Stop when the best acquisition stays below this for a while.
    #: Interpreted as "nothing worth an evaluation is left in this pool", never
    #: as convergence to a global optimum.
    acquisition_epsilon: float | None = None
    acquisition_patience: int = Field(default=3, ge=1)


class RoundRecord(BaseModel):
    """Steps 34-35. One round: what was predicted, chosen, and then measured."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    round_number: int = Field(ge=1)
    mode: str
    candidate_id: str
    genome: dict[str, float]
    predicted_score: float | None = None
    predicted_uncertainty: float | None = None
    ood_score: float | None = None
    acquisition: AcquisitionName | None = None
    acquisition_value: float | None = None
    acquisition_rank: int | None = None
    incumbent_before: float | None = None
    actual_score: float | None = None
    #: actual minus predicted. The number that says whether to believe the model.
    prediction_error: float | None = None
    pool_size: int = 0
    surrogate_state: SurrogateState = SurrogateState.UNINITIALIZED
    surrogate_trusted: bool = False
    selection_reason: str = ""
    wall_seconds: float = 0.0
    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class BayesianOptimizationOrchestrator:
    """Step 29. Runs the loop and records every decision behind it."""

    def __init__(
        self,
        space: ContentSearchSpace,
        objective: NeuralObjective,
        identity: DatasetIdentity,
        evaluate: Evaluate,
        *,
        config: BayesianOptimizationConfig | None = None,
        trust: TrustPolicy | None = None,
        budget: SearchBudget | None = None,
        ledger: BudgetLedger | None = None,
        fallback: Fallback | None = None,
        encoder: GenomeFeatureEncoder | None = None,
        trainer: SurrogateTrainer | None = None,
    ) -> None:
        self.space = space
        self.objective = objective
        self.evaluate = evaluate
        self.config = config or BayesianOptimizationConfig()
        self.budget = budget or SearchBudget(max_candidates=self.config.bootstrap_evaluations + 8)
        self.ledger = ledger or BudgetLedger()
        self.encoder = encoder or GenomeFeatureEncoder.from_space(space)
        self.trainer = trainer or SurrogateTrainer(
            self.encoder, policy=trust or TrustPolicy(), seed=self.config.random_seed
        )
        self.dataset = SurrogateDataset(identity=identity)
        self.pool_generator = CandidatePoolGenerator(space, config=self.config.pool)
        self.fallback = fallback
        self.rounds: list[RoundRecord] = []
        self.results: list[CandidateFitness] = []
        self.trained: TrainedSurrogate | None = None
        self._rng = Random(self.config.random_seed)
        self._counter = 0
        self._since_retrain = 0
        self._degraded_reason: str | None = None
        self._low_acquisition_rounds = 0
        self.stopped_reason = ""

    # --- state ------------------------------------------------------------

    @property
    def state(self) -> SurrogateState:
        """Step 53. Where the loop is, from the caller's point of view.

        A model that has not been fitted yet is reported as bootstrapping
        whether or not the quota has just been met, because from outside the
        distinction between "still gathering" and "gathered, about to fit" is
        not a difference in what the surrogate is doing: nothing.
        """
        if self.trained is None:
            return (
                SurrogateState.UNINITIALIZED
                if self.dataset.size == 0
                else SurrogateState.BOOTSTRAPPING
            )
        return self.trained.state

    @property
    def incumbent(self) -> float | None:
        best = best_of(self.results)
        return None if best is None else best.scalar_fitness

    @property
    def evaluated_fingerprints(self) -> set[str]:
        return {
            CandidateGenome(genome_id=item.candidate_id, values=item.genome).fingerprint(
                self.space
            )
            for item in self.dataset.records
        }

    # --- the loop ---------------------------------------------------------

    def run(self, rounds: int) -> list[RoundRecord]:
        """Run up to `rounds` real evaluations, respecting the budget."""
        for _ in range(rounds):
            decision = admit(1, self.ledger, self.budget)
            if decision.admitted == 0:
                self.stopped_reason = f"budget exhausted: {decision.reason}"
                break
            if not self._step():
                break
        return self.rounds

    def _step(self) -> bool:
        """One round. Returns False when the loop should stop."""
        import time

        started = time.perf_counter()
        self._counter += 1

        if self.dataset.size < self.config.bootstrap_evaluations:
            genome = self._bootstrap_candidate()
            mode, score = "bootstrap", None
        else:
            self._ensure_trained()
            if self.trained is not None and self.trained.usable:
                try:
                    selected = self._propose_by_acquisition()
                except SurrogateError as error:
                    # The model produced something that is not a number. That is
                    # a reason to stop trusting this round's selection, not a
                    # reason to abandon a search with budget left, so a Phase 9
                    # strategy chooses instead and the round says why.
                    logger.warning("surrogate could not score the pool: %s", error)
                    self._degraded_reason = str(error)
                    genome = self._fallback_candidate()
                    mode, score = "fallback", None
                else:
                    if selected is None:
                        self.stopped_reason = (
                            "the candidate pool contained nothing worth an evaluation"
                        )
                        return False
                    genome, score = selected
                    mode = "surrogate"
            else:
                genome = self._fallback_candidate()
                mode, score = "fallback", None

        if genome is None:
            self.stopped_reason = "no candidate could be proposed"
            return False

        self.ledger = self.ledger.with_proposed(1)
        fitness = self.evaluate(genome)
        self.ledger = self.ledger.with_evaluation(
            wall_seconds=fitness.compute.wall_seconds,
            cached=fitness.compute.served_from_cache,
            failed=not fitness.is_usable,
        )
        self.results.append(fitness)

        actual = fitness.scalar_fitness if fitness.is_usable else None
        predicted = None if score is None else score.predicted_mean
        self.rounds.append(
            RoundRecord(
                round_number=self._counter,
                mode=mode,
                candidate_id=genome.genome_id,
                genome=dict(genome.values),
                predicted_score=predicted,
                predicted_uncertainty=None if score is None else score.predicted_uncertainty,
                ood_score=None if score is None else score.ood_score,
                acquisition=None if score is None else score.acquisition,
                acquisition_value=None if score is None else score.value,
                acquisition_rank=1 if score is not None else None,
                incumbent_before=None if score is None else score.incumbent,
                actual_score=actual,
                prediction_error=(
                    None if actual is None or predicted is None else actual - predicted
                ),
                pool_size=self.config.pool.size if mode == "surrogate" else 0,
                surrogate_state=self.state,
                surrogate_trusted=bool(self.trained and self.trained.trusted),
                selection_reason=(
                    score.reason
                    if score is not None
                    else (
                        "bootstrap: covering the space before the model is fitted"
                        if mode == "bootstrap"
                        else f"fallback: {self._fallback_reason()}"
                    )
                ),
                wall_seconds=time.perf_counter() - started,
            )
        )

        if fitness.is_usable:
            self._record(genome, fitness)
        self._since_retrain += 1

        if self._acquisition_exhausted(score):
            self.stopped_reason = (
                f"best acquisition stayed below {self.config.acquisition_epsilon:g} for "
                f"{self.config.acquisition_patience} rounds; nothing in the pool looks "
                f"worth an expensive evaluation. This is not convergence to an optimum."
            )
            return False
        return True

    def _fallback_reason(self) -> str:
        if self._degraded_reason is not None:
            reason, self._degraded_reason = self._degraded_reason, None
            return reason
        return self.trained.trust_reason if self.trained else "no model fitted yet"

    # --- candidate sources ------------------------------------------------

    def _bootstrap_candidate(self) -> CandidateGenome | None:
        """Step 30. Cover the space before any model is fitted."""
        seen = self.evaluated_fingerprints
        for _ in range(64):
            values = self.space.sample(self._rng)
            if self.space.is_neutral(values):
                continue
            genome = CandidateGenome(
                genome_id=f"boot{self._counter:04d}",
                values=values,
                proposed_by="bootstrap",
            )
            if genome.fingerprint(self.space) not in seen:
                return genome
        return None

    def _fallback_candidate(self) -> CandidateGenome | None:
        """Step 51. A Phase 9 strategy chooses when the surrogate may not."""
        if self.fallback is not None:
            proposed = self.fallback(1)
            if proposed:
                return proposed[0]
        return self._bootstrap_candidate()

    def _propose_by_acquisition(
        self,
    ) -> tuple[CandidateGenome, AcquisitionScore] | None:
        pool = self.pool_generator.generate(
            self._rng,
            incumbents=self._incumbent_genomes(),
            exclude=self.evaluated_fingerprints,
            generation=self._counter,
        )
        if not pool:
            return None

        by_id = {genome.genome_id: genome for genome in pool}
        predictions = self.trainer.predict(
            {genome.genome_id: dict(genome.values) for genome in pool}
        )
        acquisition = self._usable_acquisition()
        scores = score_candidates(
            predictions,
            acquisition=acquisition,
            incumbent=self.incumbent,
            kappa=self.config.kappa,
            xi=self.config.xi,
        )
        batch = select_diverse_batch(
            scores,
            {genome.genome_id: dict(genome.values) for genome in pool},
            lambda left, right: genome_distance(
                CandidateGenome(genome_id="l", values=left),
                CandidateGenome(genome_id="r", values=right),
                self.space,
            ),
            count=self.config.batch_size,
            minimum_distance=self.config.minimum_batch_distance,
        )
        if not batch:
            return None
        chosen = batch[0]
        return by_id[chosen.candidate_id], chosen

    def _usable_acquisition(self) -> AcquisitionName:
        """Fall back to greedy when the configured acquisition needs sigma.

        Silently computing Expected Improvement without an uncertainty would
        raise; picking greedy and saying so keeps the loop running with an
        honest label on how the candidate was chosen.
        """
        registry = AcquisitionRegistry()
        wanted = self.config.acquisition
        if not registry.requires_uncertainty(wanted):
            return wanted
        if self.trained is not None and self.trained.metadata.supports_uncertainty:
            return wanted
        logger.info(
            "%s needs a predictive uncertainty this surrogate does not supply; "
            "using greedy mean",
            wanted.value,
        )
        return AcquisitionName.GREEDY_MEAN

    def _incumbent_genomes(self) -> list[CandidateGenome]:
        ranked = sorted(
            (item for item in self.results if item.is_usable),
            key=lambda item: item.scalar_fitness or float("-inf"),
            reverse=True,
        )
        return [item.genome for item in ranked[: self.config.pool.incumbent_count]]

    # --- dataset and model ------------------------------------------------

    def _record(self, genome: CandidateGenome, fitness: CandidateFitness) -> None:
        """Step 1. Only real evaluations become training rows."""
        raw = fitness.raw_objectives.get(self.objective.objective_id)
        if raw is None:
            return
        folded = directional_fitness(raw, self.objective)
        if folded is None:
            return
        self.dataset = self.dataset.with_record(
            SurrogateTrainingRecord(
                candidate_id=genome.genome_id,
                genome=dict(genome.values),
                objective_values={
                    key: value
                    for key, value in fitness.raw_objectives.items()
                    if value is not None
                },
                directional_target=folded,
                primary_objective_id=self.objective.objective_id,
                feasible=fitness.is_feasible,
                sequence=self.dataset.size,
                evaluation_wall_seconds=fitness.compute.wall_seconds,
            )
        )

    def _ensure_trained(self) -> None:
        """Steps 36-37. Retrain on the configured cadence."""
        if self.trained is not None and not self.trainer.should_retrain(self._since_retrain):
            return
        if self.dataset.size < 2:
            return
        self.trained = self.trainer.fit(self.dataset)
        self._since_retrain = 0
        logger.info(
            "surrogate v%d on %d observations: %s",
            self.trained.metadata.model_version,
            self.dataset.size,
            self.trained.trust_reason,
        )

    def _acquisition_exhausted(self, score: AcquisitionScore | None) -> bool:
        if self.config.acquisition_epsilon is None or score is None:
            return False
        if score.value < self.config.acquisition_epsilon:
            self._low_acquisition_rounds += 1
        else:
            self._low_acquisition_rounds = 0
        return self._low_acquisition_rounds >= self.config.acquisition_patience

    # --- reporting --------------------------------------------------------

    def diagnostics(self) -> dict[str, object]:
        """Step 35. Predicted against actual, and what it cost to find out."""
        predicted = [
            row for row in self.rounds
            if row.predicted_score is not None and row.actual_score is not None
        ]
        errors = np.asarray([row.prediction_error for row in predicted], dtype=float)
        best = best_of(self.results)
        return {
            "rounds": len(self.rounds),
            "bootstrap_rounds": sum(1 for row in self.rounds if row.mode == "bootstrap"),
            "surrogate_rounds": sum(1 for row in self.rounds if row.mode == "surrogate"),
            "fallback_rounds": sum(1 for row in self.rounds if row.mode == "fallback"),
            "training_samples": self.dataset.size,
            "target_spread": self.dataset.target_spread(),
            "state": self.state.value,
            "trusted": bool(self.trained and self.trained.trusted),
            "trust_reason": self.trained.trust_reason if self.trained else "never fitted",
            "best_observed": None if best is None else best.scalar_fitness,
            "best_candidate_id": None if best is None else best.candidate_id,
            "prediction_mae": float(np.mean(np.abs(errors))) if errors.size else None,
            "prediction_bias": float(np.mean(errors)) if errors.size else None,
            "predictions_compared": int(errors.size),
            "tribe_runs": self.ledger.tribe_runs,
            "wall_seconds": self.ledger.wall_seconds,
            "stopped_reason": self.stopped_reason,
        }


__all__ = [
    "BayesianOptimizationConfig",
    "BayesianOptimizationOrchestrator",
    "RoundRecord",
]
