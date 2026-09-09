"""Step 47: the loop that spends the budget.

THE LOOP
    while not stopping:
        propose -> filter duplicates -> admit against budget -> evaluate
        -> record -> let the strategy observe -> checkpoint

    Every one of those is a separate object, so this module contains no search
    logic, no budget arithmetic and no scoring. What it owns is the order they
    happen in, and the guarantee that budget is checked *before* work is
    scheduled rather than after.

WHY IT CHECKPOINTS AFTER EVERY EVALUATION
    An evaluation is roughly 52 minutes. Losing a generation to a crash is
    losing most of a day. State is written after each result, so a killed
    process resumes at the next candidate rather than the last generation
    boundary.

WHY A FAILED CANDIDATE IS NOT AN ERROR
    Step 54. The evaluator returns failures as data, and the loop records them,
    counts them against the budget they consumed, and continues. A search that
    died because one ffmpeg invocation failed would waste every hour spent
    before it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

from blackmirror.search.budget import BudgetLedger, admit
from blackmirror.search.evaluator import CandidateEvaluator
from blackmirror.search.fitness import CandidateFitness, best_infeasible, tied_with_best
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.guardrails import GuardrailPolicy
from blackmirror.search.pareto import ParetoFront
from blackmirror.search.population import SearchPopulation, build_population
from blackmirror.search.scheduler import CandidateScheduler, HumanOverride
from blackmirror.search.schemas import (
    SearchExperiment,
    SearchStatus,
    StoppingReason,
)
from blackmirror.search.storage import SearchStore
from blackmirror.search.strategy import SearchStrategy, unique_genomes
from blackmirror.search.tracking import (
    SearchEfficiencyMetrics,
    SearchEventType,
    SearchTracker,
    StoppingPolicy,
    budget_warnings,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Called after each evaluation with the experiment as it now stands.
ProgressHook = Callable[["SearchRunState"], None]


class SearchRunState:
    """The mutable half of a search: ledger, tracker, strategy state."""

    def __init__(self, experiment: SearchExperiment, tracker: SearchTracker) -> None:
        self.experiment = experiment
        self.tracker = tracker
        self.ledger = experiment.ledger
        self.generation = experiment.generation
        self.stopping_reason: StoppingReason | None = None
        self.stopping_detail = ""

    @property
    def best(self) -> CandidateFitness | None:
        return self.tracker.best_fitness

    def metrics(self, minimum_improvement: float) -> SearchEfficiencyMetrics:
        return self.tracker.metrics(self.ledger, minimum_improvement=minimum_improvement)


class NeuralSearchOrchestrator:
    """Runs a bounded search and persists enough to resume it."""

    def __init__(
        self,
        experiment: SearchExperiment,
        strategy: SearchStrategy,
        evaluator: CandidateEvaluator,
        *,
        artifact_root: Path | None = None,
        root_fitness: float | None = None,
        guardrails: GuardrailPolicy | None = None,
        scheduler: CandidateScheduler | None = None,
        override: HumanOverride | None = None,
    ) -> None:
        self.experiment = experiment
        self.strategy = strategy
        self.evaluator = evaluator
        self.artifact_root = Path(artifact_root or "artifacts")
        self.directory = self.artifact_root / "search" / experiment.search_id
        self.tracker = SearchTracker(root_fitness=root_fitness)
        self.policy = StoppingPolicy(experiment.config, experiment.budget)
        self.state = SearchRunState(experiment, self.tracker)
        self.guardrails = guardrails or GuardrailPolicy()
        self.override = override or HumanOverride()
        self.scheduler = scheduler or CandidateScheduler(
            max_concurrent=experiment.config.max_concurrent_candidates,
            override=self.override,
        )
        self.populations: list[SearchPopulation] = []
        self.store = SearchStore(self.artifact_root)
        self._initialized = False

    # --- control ----------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the loop to end at the next candidate boundary (Step 72)."""
        self.policy.request_stop()

    # --- running ----------------------------------------------------------

    def run(self, *, on_progress: ProgressHook | None = None) -> SearchRunState:
        """Run until a stopping condition fires. Returns the final state."""
        if not self._initialized:
            self.strategy.initialize(
                self.experiment.space, self.experiment.config.random_seed
            )
            self._initialized = True
            self.tracker.event(
                SearchEventType.SEARCH_STARTED,
                f"{self.experiment.search_id} using {self.strategy.name.value} "
                f"with seed {self.experiment.config.random_seed}",
            )
            for warning in self.experiment.warnings():
                self.tracker.event(SearchEventType.BUDGET_WARNING, warning)
            self._checkpoint()

        config = self.experiment.config
        while True:
            stop = self.policy.check(
                self.state.ledger,
                self.state.best,
                strategy_stop=self.strategy.should_stop(),
            )
            if stop is not None:
                self._finish(*stop)
                break

            proposed = self.strategy.propose(
                config.candidate_batch_size, self.state.generation
            )
            proposed = unique_genomes(proposed, self.experiment.space)
            if not proposed:
                self._finish(
                    StoppingReason.SPACE_EXHAUSTED,
                    "the strategy proposed no further candidates",
                )
                break

            decision = admit(len(proposed), self.state.ledger, self.experiment.budget)
            if decision.admitted == 0:
                self._finish(StoppingReason.MAX_CANDIDATES, decision.reason)
                break
            admitted = proposed[: decision.admitted]
            self.state.ledger = self.state.ledger.with_proposed(len(admitted))
            results = self._evaluate_batch(admitted, on_progress)
            self.strategy.observe(results)

            self.populations.append(
                build_population(
                    self.state.generation,
                    results,
                    self.experiment.space,
                    keep=config.beam_width,
                )
            )
            self.state.ledger = self.state.ledger.with_generation_completed()
            self.policy.note_generation(self.state.best)
            self.tracker.event(
                SearchEventType.GENERATION_COMPLETED,
                f"generation {self.state.generation} finished with "
                f"{len(results)} evaluation(s)",
                generation=self.state.generation,
            )
            for warning in budget_warnings(self.state.ledger, self.experiment.budget):
                self.tracker.event(SearchEventType.BUDGET_WARNING, warning)
            self.state.generation += 1
            self._checkpoint()

        return self.state

    def _evaluate_batch(
        self, genomes: tuple[CandidateGenome, ...], on_progress: ProgressHook | None
    ) -> list[CandidateFitness]:
        """Order the batch, evaluate it, and judge each result's feasibility.

        Concurrency comes from the scheduler and defaults to one worker.
        Inference is memory-bound on this machine, so parallel evaluations
        contend for the same weights and finish later in total; the knob is
        honoured but the default is deliberate.
        """
        elites = {
            item.candidate_id
            for item in self.tracker.results[-self.experiment.config.beam_width :]
            if item.is_eligible
        }
        queued = self.scheduler.prioritize(genomes, elite_ids=elites)
        for item in queued:
            self.tracker.event(
                SearchEventType.CANDIDATE_PROPOSED,
                f"{item.genome.genome_id} queued: {item.describe()}",
                candidate_id=item.genome.genome_id,
                generation=self.state.generation,
            )

        results: list[CandidateFitness] = []

        def run_one(genome: CandidateGenome) -> CandidateFitness:
            fitness = self._judge(self.evaluator.evaluate(genome))
            results.append(fitness)
            self.state.ledger = self.state.ledger.with_evaluation(
                wall_seconds=fitness.compute.wall_seconds,
                cached=fitness.compute.served_from_cache,
                failed=not fitness.is_usable,
            )
            self.tracker.record(
                fitness, generation=self.state.generation, ledger=self.state.ledger
            )
            self._checkpoint()
            if on_progress is not None:
                on_progress(self.state)
            return fitness

        # A stop requested mid-batch takes effect at the next candidate, not
        # after the whole generation, so a user is not made to wait another
        # hour for work they asked to end.
        self.scheduler.run(
            queued,
            run_one,
            should_continue=lambda: not self.policy.user_stopped and not self.override.paused,
        )
        return results

    def _judge(self, fitness: CandidateFitness) -> CandidateFitness:
        """Attach the guardrail verdict, which decides eligibility to win."""
        if not fitness.is_usable:
            return fitness
        feasibility = self.guardrails.check(
            {k: v for k, v in fitness.raw_objectives.items()},
            duration_change=fitness.duration_change_seconds,
        )
        if not feasibility.feasible:
            self.tracker.event(
                SearchEventType.CANDIDATE_ELIMINATED,
                f"{fitness.candidate_id} breached a guardrail: {feasibility.summary}",
                candidate_id=fitness.candidate_id,
                generation=self.state.generation,
            )
        return fitness.model_copy(update={"feasibility": feasibility})

    def _finish(self, reason: StoppingReason, detail: str) -> None:
        self.state.stopping_reason = reason
        self.state.stopping_detail = detail
        self.tracker.event(
            SearchEventType.STOPPING_CONDITION_MET, f"{reason.value}: {detail}"
        )
        self.tracker.event(
            SearchEventType.SEARCH_COMPLETED,
            f"{len(self.tracker.results)} evaluation(s), "
            f"{self.state.ledger.tribe_runs} inference run(s)",
        )
        self.state.experiment = self.experiment.model_copy(
            update={
                "status": (
                    SearchStatus.STOPPED
                    if reason is StoppingReason.USER_STOPPED
                    else SearchStatus.COMPLETED
                ),
                "stopping_reason": reason,
                "ledger": self.state.ledger,
                "generation": self.state.generation,
                "updated_at": dt.datetime.now(dt.UTC),
            }
        )
        self._checkpoint()
        self._persist_artifacts()

    # --- persistence ------------------------------------------------------

    def _checkpoint(self) -> None:
        """Write state atomically, so a crash mid-write cannot corrupt it."""
        try:
            self.store.write_state(self.experiment.search_id, self.snapshot())
        except OSError as exc:  # pragma: no cover - disk failure
            logger.warning("could not checkpoint search state: %s", exc)

    def _persist_artifacts(self) -> None:
        """The full layout: definition, evaluations, populations, report."""
        search_id = self.experiment.search_id
        try:
            self.store.write_definition(self.state.experiment)
            for fitness in self.tracker.results:
                self.store.write_evaluation(search_id, fitness)
            for population in self.populations:
                self.store.write_population(
                    search_id, population.generation, population.model_dump(mode="json")
                )
            self.store.write_named(
                search_id,
                "trajectory.json",
                [point.model_dump(mode="json") for point in self.tracker.trajectory],
            )
            self.store.write_named(
                search_id,
                "events.json",
                [event.model_dump(mode="json") for event in self.tracker.events],
            )
            if self.experiment.is_multi_objective:
                self.store.write_named(
                    search_id, "pareto_front.json", self.pareto().model_dump(mode="json")
                )
            self.store.write_named(search_id, "final_report.json", self.report())
        except (OSError, ValueError) as exc:  # pragma: no cover - disk failure
            logger.warning("could not persist search artifacts: %s", exc)

    def snapshot(self) -> dict[str, object]:
        """Everything needed to resume, and everything a report needs."""
        experiment = self.state.experiment.model_copy(
            update={"ledger": self.state.ledger, "generation": self.state.generation}
        )
        best = self.state.best
        return {
            "experiment": experiment.model_dump(mode="json"),
            "strategy": self.strategy.get_state(),
            "best": (
                None
                if self.tracker.best is None
                else self.tracker.best.model_dump(mode="json")
            ),
            "metrics": self.state.metrics(
                self.experiment.config.minimum_improvement
            ).model_dump(mode="json"),
            "trajectory": [point.model_dump(mode="json") for point in self.tracker.trajectory],
            "events": [event.model_dump(mode="json") for event in self.tracker.events],
            "results": [item.model_dump(mode="json") for item in self.tracker.results],
            "stopping_reason": (
                None if self.state.stopping_reason is None else self.state.stopping_reason.value
            ),
            "stopping_detail": self.state.stopping_detail,
            "best_candidate_id": None if best is None else best.candidate_id,
        }

    def resume_from(self, snapshot: dict[str, object]) -> None:
        """Restore strategy and ledger so the loop continues where it stopped.

        Evaluated candidates are replayed into the tracker rather than
        re-evaluated, which is the whole point: at roughly an hour each,
        re-running seven completed candidates would cost most of a day to
        learn nothing.
        """
        strategy_state = snapshot.get("strategy")
        if isinstance(strategy_state, dict):
            self.strategy.initialize(
                self.experiment.space, self.experiment.config.random_seed
            )
            self.strategy.set_state(strategy_state)
            self._initialized = True

        experiment = snapshot.get("experiment")
        if isinstance(experiment, dict):
            ledger = experiment.get("ledger")
            if isinstance(ledger, dict):
                self.state.ledger = BudgetLedger.model_validate(ledger)
            generation = experiment.get("generation")
            if isinstance(generation, int):
                self.state.generation = generation

        results = snapshot.get("results")
        if isinstance(results, list):
            restored = [CandidateFitness.model_validate(item) for item in results]
            self.tracker.results = restored
            self.strategy.observe(restored)

    # --- reporting --------------------------------------------------------

    def pareto(self) -> ParetoFront:
        """The non-dominated set across every declared objective (Step 59)."""
        from blackmirror.search.pareto import pareto_front

        return pareto_front(self.tracker.results, self.experiment.objectives)

    def report(self) -> dict[str, object]:
        """Step 89. What was found, what it cost, and what it does not show."""
        config = self.experiment.config
        metrics = self.state.metrics(config.minimum_improvement)
        best = self.state.best
        usable = [item for item in self.tracker.results if item.is_usable]
        tied = tied_with_best(usable, minimum_improvement=config.minimum_improvement)
        caveats: list[str] = [self.experiment.interpretation_notice]
        if metrics.improvement_is_resolvable is False:
            caveats.append(
                f"The best candidate's improvement over the root "
                f"({metrics.absolute_improvement:+.4f}) is smaller than the measured "
                f"nuisance floor of {config.minimum_improvement:g}. It is not "
                f"distinguishable from a change of implementation detail."
            )
        if tied:
            caveats.append(
                f"The best candidate is not measurably ahead of "
                f"{len(tied)} other(s): "
                f"{', '.join(item.candidate_id for item in tied)}. The ranking among "
                f"them is not a finding."
            )
        blocked = best_infeasible(self.tracker.results)
        if blocked is not None and (
            best is None
            or (blocked.scalar_fitness or 0) > (best.scalar_fitness or float("-inf"))
        ):
            caveats.append(
                f"The highest score belonged to {blocked.candidate_id}, which was "
                f"disqualified: {blocked.feasibility.summary}. It is not the result."
            )
        caveats.extend(config.warnings())
        return {
            "search_id": self.experiment.search_id,
            "strategy": self.strategy.name.value,
            "seed": config.random_seed,
            "status": self.state.experiment.status.value,
            "stopping_reason": (
                None if self.state.stopping_reason is None else self.state.stopping_reason.value
            ),
            "stopping_detail": self.state.stopping_detail,
            "best_observed": None
            if best is None
            else {
                "candidate_id": best.candidate_id,
                "genome": best.genome.canonical_values(self.experiment.space),
                "fitness": best.scalar_fitness,
                "raw_objectives": best.raw_objectives,
                "candidate_run_id": best.candidate_run_id,
                "variant_path": best.variant_path,
                "lineage": list(best.genome.parent_genome_ids),
            },
            "metrics": metrics.model_dump(mode="json"),
            "pareto_front": (
                self.pareto().model_dump(mode="json")
                if self.experiment.is_multi_objective
                else None
            ),
            "populations": [item.model_dump(mode="json") for item in self.populations],
            "lineage": self.winning_lineage(),
            "guardrails": list(self.guardrails.describe()),
            "caveats": caveats,
        }

    def winning_lineage(self) -> list[dict[str, object]]:
        """Step 90. The exact path from the root to the best observed variant."""
        best = self.state.best
        if best is None:
            return []
        by_id = {item.candidate_id: item for item in self.tracker.results}
        chain: list[CandidateFitness] = []
        cursor: CandidateFitness | None = best
        seen: set[str] = set()
        while cursor is not None and cursor.candidate_id not in seen:
            seen.add(cursor.candidate_id)
            chain.append(cursor)
            parents = cursor.genome.parent_genome_ids
            cursor = by_id.get(parents[0]) if parents else None
        chain.reverse()
        return [
            {
                "candidate_id": item.candidate_id,
                "generation": item.genome.generation,
                "origin": item.genome.origin.value,
                "genome": item.genome.canonical_values(self.experiment.space),
                "fitness": item.scalar_fitness,
            }
            for item in chain
        ]


__all__ = ["NeuralSearchOrchestrator", "ProgressHook", "SearchRunState"]
