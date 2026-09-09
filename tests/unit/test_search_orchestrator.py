"""Step 47 and Steps 80-84: the loop, its limits, and resuming it.

The evaluator is stood in for by a synthetic surface. What is under test is the
order of operations and the guarantees around them: that budget is checked
before work is scheduled, that a failing candidate does not end the run, that a
stop is honoured, and that resuming does not pay again for candidates already
evaluated. At roughly 52 minutes each, every one of those is worth real hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.search.budget import SearchBudget
from blackmirror.search.fitness import CandidateFitness, ComputeCost, FitnessStatus
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.orchestrator import NeuralSearchOrchestrator
from blackmirror.search.schemas import (
    SearchConfig,
    SearchExperiment,
    SearchStatus,
    StoppingReason,
)
from blackmirror.search.space import ContentSearchSpace, continuous
from blackmirror.search.strategy import RandomSearch
from blackmirror.search.tracking import SearchEventType


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),)
    )


def _objective() -> NeuralObjective:
    return NeuralObjective(
        objective_id="o1", name="o1", metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE,
    )


def _experiment(tmp_path: Path, **overrides: object) -> SearchExperiment:
    fields: dict[str, object] = {
        "search_id": "s1",
        "experiment_id": "exp",
        "root_run_id": "root",
        "root_media_path": str(tmp_path / "root.mp4"),
        "root_media_sha256": "a" * 64,
        "objective_set_hash": "b" * 64,
        "objective_definition_hash": "c" * 64,
        "objectives": (_objective(),),
        "space": _space(),
        "config": SearchConfig(random_seed=3, candidate_batch_size=2, patience=100),
        "budget": SearchBudget(max_candidates=6, max_generations=10),
    }
    fields.update(overrides)
    return SearchExperiment(**fields)  # type: ignore[arg-type]


class StubEvaluator:
    """A synthetic surface with a known optimum at gain = 3."""

    def __init__(self, *, fail_ids: set[str] | None = None, wall: float = 1.0) -> None:
        self.fail_ids = fail_ids or set()
        self.wall = wall
        self.calls: list[str] = []

    def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
        self.calls.append(genome.genome_id)
        if genome.genome_id in self.fail_ids:
            return CandidateFitness(
                candidate_id=genome.genome_id, genome=genome,
                status=FitnessStatus.EVALUATION_FAILED, reason="synthetic failure",
                compute=ComputeCost(wall_seconds=self.wall, tribe_runs=1),
            )
        value = -((genome.values["gain"] - 3.0) ** 2)
        return CandidateFitness(
            candidate_id=genome.genome_id, genome=genome,
            status=FitnessStatus.EVALUATED, scalar_fitness=value,
            primary_objective_id="o1", raw_objectives={"o1": value},
            candidate_run_id=f"run-{genome.genome_id}",
            compute=ComputeCost(wall_seconds=self.wall, tribe_runs=1),
        )


def _orchestrator(
    tmp_path: Path, evaluator: object | None = None, **overrides: object
) -> NeuralSearchOrchestrator:
    return NeuralSearchOrchestrator(
        _experiment(tmp_path, **overrides),
        RandomSearch(),
        evaluator or StubEvaluator(),  # type: ignore[arg-type]
        artifact_root=tmp_path,
        root_fitness=-9.0,
    )


class TestBudgetEnforcement:
    def test_never_exceeds_the_candidate_budget(self, tmp_path: Path) -> None:
        """Step 80: budget 5, attempt many, at most 5 evaluations occur."""
        evaluator = StubEvaluator()
        orchestrator = _orchestrator(
            tmp_path, evaluator,
            budget=SearchBudget(max_candidates=5, max_generations=50),
        )

        orchestrator.run()

        assert len(evaluator.calls) == 5

    def test_stops_on_the_generation_limit(self, tmp_path: Path) -> None:
        orchestrator = _orchestrator(
            tmp_path, budget=SearchBudget(max_candidates=100, max_generations=2)
        )

        state = orchestrator.run()

        assert state.stopping_reason is StoppingReason.MAX_GENERATIONS
        assert state.ledger.generations_completed == 2

    def test_a_partial_admission_is_honoured(self, tmp_path: Path) -> None:
        """Batch of 2 against 1 remaining evaluates exactly 1."""
        evaluator = StubEvaluator()
        orchestrator = _orchestrator(
            tmp_path, evaluator,
            budget=SearchBudget(max_candidates=3, max_generations=50),
        )

        orchestrator.run()

        assert len(evaluator.calls) == 3


class TestFailureHandling:
    def test_a_failed_candidate_does_not_end_the_search(self, tmp_path: Path) -> None:
        """Step 54. Otherwise one bad ffmpeg call wastes every prior hour."""
        evaluator = StubEvaluator(fail_ids={"c0001", "c0002"})
        orchestrator = _orchestrator(tmp_path, evaluator)

        state = orchestrator.run()

        assert len(evaluator.calls) == 6
        assert state.ledger.evaluation_failures == 2
        assert state.best is not None

    def test_failures_are_recorded_as_events(self, tmp_path: Path) -> None:
        orchestrator = _orchestrator(tmp_path, StubEvaluator(fail_ids={"c0001"}))

        orchestrator.run()

        types = [event.type for event in orchestrator.tracker.events]
        assert SearchEventType.CANDIDATE_FAILED in types

    def test_a_search_where_everything_fails_still_completes(
        self, tmp_path: Path
    ) -> None:
        evaluator = StubEvaluator(fail_ids={f"c{i:04d}" for i in range(1, 20)})
        orchestrator = _orchestrator(tmp_path, evaluator)

        state = orchestrator.run()

        assert state.best is None
        assert state.stopping_reason is not None
        assert state.metrics(0.0157).failure_rate == pytest.approx(1.0)


class TestStopping:
    def test_target_reached_ends_the_search_early(self, tmp_path: Path) -> None:
        """Step 83."""
        orchestrator = _orchestrator(
            tmp_path,
            config=SearchConfig(
                random_seed=3, candidate_batch_size=4, target_fitness=-25.0, patience=100
            ),
            budget=SearchBudget(max_candidates=100, max_generations=50),
        )

        state = orchestrator.run()

        assert state.stopping_reason is StoppingReason.TARGET_REACHED
        assert state.best is not None and state.best.scalar_fitness >= -25.0

    def test_patience_ends_a_search_that_stops_improving(self, tmp_path: Path) -> None:
        """Step 82."""
        class Flat:
            def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
                return CandidateFitness(
                    candidate_id=genome.genome_id, genome=genome,
                    status=FitnessStatus.EVALUATED, scalar_fitness=0.5,
                    primary_objective_id="o1",
                    compute=ComputeCost(wall_seconds=1.0, tribe_runs=1),
                )

        orchestrator = _orchestrator(
            tmp_path, Flat(),
            config=SearchConfig(random_seed=3, candidate_batch_size=1, patience=3),
            budget=SearchBudget(max_candidates=100, max_generations=50),
        )

        state = orchestrator.run()

        assert state.stopping_reason is StoppingReason.NO_IMPROVEMENT

    def test_a_requested_stop_is_honoured_at_the_next_candidate(
        self, tmp_path: Path
    ) -> None:
        """Step 72: a user must not wait another hour for work they ended."""
        orchestrator = _orchestrator(
            tmp_path, budget=SearchBudget(max_candidates=100, max_generations=50)
        )
        seen: list[int] = []

        def stop_after_two(state: object) -> None:
            seen.append(1)
            if len(seen) == 2:
                orchestrator.request_stop()

        state = orchestrator.run(on_progress=stop_after_two)

        assert state.stopping_reason is StoppingReason.USER_STOPPED
        assert len(seen) == 2
        assert state.experiment.status is SearchStatus.STOPPED

    def test_a_completed_search_records_why_it_stopped(self, tmp_path: Path) -> None:
        state = _orchestrator(tmp_path).run()

        assert state.experiment.status is SearchStatus.COMPLETED
        assert state.experiment.stopping_reason is not None


class TestDuplicatesAndCost:
    def test_the_same_point_is_never_evaluated_twice(self, tmp_path: Path) -> None:
        """Step 81. A duplicate's whole cost is the evaluation it triggers."""
        evaluator = StubEvaluator()
        orchestrator = _orchestrator(tmp_path, evaluator)

        orchestrator.run()
        space = orchestrator.experiment.space
        fingerprints = [
            item.genome.fingerprint(space) for item in orchestrator.tracker.results
        ]

        assert len(fingerprints) == len(set(fingerprints))
        assert len(evaluator.calls) == len(set(evaluator.calls))

    def test_the_ledger_matches_what_was_evaluated(self, tmp_path: Path) -> None:
        evaluator = StubEvaluator(wall=60.0)
        orchestrator = _orchestrator(tmp_path, evaluator)

        state = orchestrator.run()

        assert state.ledger.candidates_evaluated == len(evaluator.calls)
        assert state.ledger.tribe_runs == len(evaluator.calls)
        assert state.ledger.wall_seconds == pytest.approx(60.0 * len(evaluator.calls))


class TestCheckpointAndResume:
    def test_state_is_written_after_every_evaluation(self, tmp_path: Path) -> None:
        """A crash must cost one candidate, not a generation."""
        orchestrator = _orchestrator(tmp_path)
        written: list[int] = []

        def count(state: object) -> None:
            path = tmp_path / "search" / "s1" / "state.json"
            written.append(len(json.loads(path.read_text())["results"]))

        orchestrator.run(on_progress=count)

        assert written == list(range(1, len(written) + 1))

    def test_the_checkpoint_is_valid_json_with_the_trajectory(
        self, tmp_path: Path
    ) -> None:
        orchestrator = _orchestrator(tmp_path)
        orchestrator.run()

        payload = json.loads((tmp_path / "search" / "s1" / "state.json").read_text())

        assert payload["experiment"]["search_id"] == "s1"
        assert len(payload["trajectory"]) == len(orchestrator.tracker.results)
        assert payload["metrics"]["evaluations"] == len(orchestrator.tracker.results)
        assert payload["events"]

    def test_resume_does_not_re_evaluate_completed_candidates(
        self, tmp_path: Path
    ) -> None:
        """Step 84. Re-running seven candidates would cost most of a day."""
        first = _orchestrator(
            tmp_path, StubEvaluator(),
            budget=SearchBudget(max_candidates=4, max_generations=50),
        )
        first.run()
        snapshot = first.snapshot()

        evaluator = StubEvaluator()
        resumed = NeuralSearchOrchestrator(
            _experiment(tmp_path, budget=SearchBudget(max_candidates=8, max_generations=50)),
            RandomSearch(),
            evaluator,  # type: ignore[arg-type]
            artifact_root=tmp_path,
            root_fitness=-9.0,
        )
        resumed.resume_from(snapshot)
        resumed.run()

        already = {item["candidate_id"] for item in snapshot["results"]}  # type: ignore[union-attr]
        assert not already & set(evaluator.calls)
        assert len(evaluator.calls) == 4

    def test_resume_restores_the_ledger(self, tmp_path: Path) -> None:
        first = _orchestrator(
            tmp_path, budget=SearchBudget(max_candidates=4, max_generations=50)
        )
        first.run()

        resumed = _orchestrator(
            tmp_path, budget=SearchBudget(max_candidates=8, max_generations=50)
        )
        resumed.resume_from(first.snapshot())

        assert resumed.state.ledger.candidates_evaluated == 4


class TestReport:
    def test_names_the_best_observed_and_what_it_cost(self, tmp_path: Path) -> None:
        orchestrator = _orchestrator(tmp_path)
        orchestrator.run()

        report = orchestrator.report()

        assert report["best_observed"] is not None
        assert report["metrics"]["evaluations"] == 6
        assert report["strategy"] == "random_search"
        assert report["seed"] == 3

    def test_carries_the_caveat_that_this_is_not_a_global_optimum(
        self, tmp_path: Path
    ) -> None:
        orchestrator = _orchestrator(tmp_path)
        orchestrator.run()

        caveats = " ".join(orchestrator.report()["caveats"])  # type: ignore[arg-type]

        assert "not a global optimum" in caveats

    def test_says_when_the_improvement_is_inside_the_noise_floor(
        self, tmp_path: Path
    ) -> None:
        """The single most misleading thing this system could omit."""
        class Barely:
            def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
                return CandidateFitness(
                    candidate_id=genome.genome_id, genome=genome,
                    status=FitnessStatus.EVALUATED, scalar_fitness=0.305,
                    primary_objective_id="o1",
                    compute=ComputeCost(wall_seconds=1.0, tribe_runs=1),
                )

        orchestrator = NeuralSearchOrchestrator(
            _experiment(tmp_path), RandomSearch(), Barely(),  # type: ignore[arg-type]
            artifact_root=tmp_path, root_fitness=0.300,
        )
        orchestrator.run()

        caveats = " ".join(orchestrator.report()["caveats"])  # type: ignore[arg-type]

        assert "nuisance floor" in caveats
        assert "not" in caveats

    def test_lists_candidates_the_winner_is_not_measurably_ahead_of(
        self, tmp_path: Path
    ) -> None:
        class Close:
            def __init__(self) -> None:
                self.n = 0

            def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
                self.n += 1
                return CandidateFitness(
                    candidate_id=genome.genome_id, genome=genome,
                    status=FitnessStatus.EVALUATED,
                    scalar_fitness=0.30 + 0.001 * self.n,
                    primary_objective_id="o1",
                    compute=ComputeCost(wall_seconds=1.0, tribe_runs=1),
                )

        orchestrator = NeuralSearchOrchestrator(
            _experiment(tmp_path), RandomSearch(), Close(),  # type: ignore[arg-type]
            artifact_root=tmp_path, root_fitness=0.0,
        )
        orchestrator.run()

        report = orchestrator.report()

        assert report["metrics"]["tied_candidate_ids"]
        assert "not measurably ahead" in " ".join(report["caveats"])  # type: ignore[arg-type]
