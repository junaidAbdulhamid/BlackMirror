"""Steps 11-15: fitness, the evaluator over Phase 8, and random search.

Phase 8 is stood in for. Its own suite already proves the state machine, and a
real evaluation costs roughly 52 minutes, so what is tested here is the wiring:
that a genome becomes a valid Phase 8 request, that a result becomes a fitness
with the direction folded in correctly, and that a failure anywhere returns a
result instead of ending the search.
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

import pytest

from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.optimization.schemas import OptimizationConstraints
from blackmirror.resimulation.schemas import (
    FailedAttempt,
    ObjectiveDelta,
    Outcome,
    ResimulationResult,
    RunStatus,
    Stage,
    StopReason,
)
from blackmirror.scoring.engine import objective_set_hash
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.search.decode import GenomeDecoder
from blackmirror.search.evaluator import CandidateEvaluator
from blackmirror.search.fitness import (
    CandidateFitness,
    ComputeCost,
    FitnessStatus,
    directional_fitness,
)
from blackmirror.search.genome import CandidateGenome, GenomeOrigin
from blackmirror.search.schemas import SearchStrategyName
from blackmirror.search.space import ContentSearchSpace, continuous
from blackmirror.search.strategy import (
    RandomSearch,
    SearchStrategyRegistry,
    default_registry,
    unique_genomes,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="candidate evaluation requires ffmpeg and ffprobe on PATH",
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("media") / "root.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-nostdin", "-fflags", "+bitexact",
            "-f", "lavfi", "-i", "testsrc=size=128x96:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-x264-params", "threads=1", "-c:a", "aac", "-b:a", "64k",
            "-map_metadata", "-1", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(continuous("gain", Op.AUDIO_GAIN_DB, -6.0, 6.0, resolution=0.5),)
    )


def _objective(direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE) -> NeuralObjective:
    normalization = (
        ObjectiveNormalization(target_value=0.5, target_tolerance=0.1)
        if direction is ObjectiveDirection.TARGET
        else ObjectiveNormalization()
    )
    return NeuralObjective(
        objective_id="o1",
        name="o1",
        metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction,
        normalization=normalization,
    )


def _decoder(clip: Path, **kwargs: object) -> GenomeDecoder:
    fields: dict[str, object] = {
        "search_id": "s1",
        "experiment_id": "exp",
        "space": _space(),
        "parent_variant_id": "root-run",
        "root_media_path": str(clip),
        "root_duration_seconds": 1.0,
        "objective": _objective(),
    }
    fields.update(kwargs)
    return GenomeDecoder(**fields)  # type: ignore[arg-type]


def _genome(gain: float = 3.0, genome_id: str = "c1") -> CandidateGenome:
    return CandidateGenome(
        genome_id=genome_id,
        values={"gain": gain},
        origin=GenomeOrigin.RANDOM_SAMPLE,
        proposed_by="random_search",
    )


class FakeOrchestrator:
    """Stands in for Phase 8, recording the request it was handed."""

    def __init__(
        self,
        *,
        candidate_value: float | None = 0.42,
        status: RunStatus = RunStatus.COMPLETED,
        raise_exc: Exception | None = None,
    ) -> None:
        self.candidate_value = candidate_value
        self.status = status
        self.raise_exc = raise_exc
        self.requests: list[object] = []

    def start(self, request: object) -> ResimulationResult:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        deltas = (
            ObjectiveDelta(
                objective_id="o1",
                direction="maximize",
                parent_raw_value=0.1,
                candidate_raw_value=self.candidate_value,
                raw_delta=None if self.candidate_value is None else self.candidate_value - 0.1,
                parent_score=0.1,
                candidate_score=self.candidate_value,
                score_delta=None,
                outcome=Outcome.IMPROVED,
            ),
        )
        failures = (
            ()
            if self.status is RunStatus.COMPLETED
            else (
                FailedAttempt(
                    attempt=1, stage=Stage.INFERENCE,
                    error_type="RuntimeError", message="inference blew up",
                ),
            )
        )
        return ResimulationResult(
            request=request,  # type: ignore[arg-type]
            cache_key="a" * 64,
            status=self.status,
            current_stage=Stage.EVALUATED if self.status is RunStatus.COMPLETED else Stage.BOUND,
            candidate_run_id="cand-run",
            objective_deltas=deltas if self.status is RunStatus.COMPLETED else (),
            failures=failures,
            stop_reason=(
                StopReason.COMPLETED
                if self.status is RunStatus.COMPLETED
                else StopReason.FAILED_STAGE
            ),
        )


def _evaluator(clip: Path, tmp_path: Path, orchestrator: object, **kwargs: object):
    decoder = _decoder(clip, **kwargs)
    objectives = (decoder.objective,)
    return CandidateEvaluator(
        tmp_path,
        decoder,
        orchestrator,  # type: ignore[arg-type]
        objectives=objectives,
        objective_set_hash="b" * 64,
        objective_definition_hash=objective_set_hash(objectives),
        candidate_dir=tmp_path / "candidates",
    )


# ---------------------------------------------------------------------------
# Step 12 - fitness
# ---------------------------------------------------------------------------


class TestDirectionalFitness:
    def test_maximize_keeps_the_value(self) -> None:
        assert directional_fitness(0.4, _objective(ObjectiveDirection.MAXIMIZE)) == 0.4

    def test_minimize_is_negated_so_higher_is_still_better(self) -> None:
        """Every strategy takes maxima; one convention keeps them all correct."""
        assert directional_fitness(0.4, _objective(ObjectiveDirection.MINIMIZE)) == -0.4

    def test_target_becomes_negative_distance_from_the_target(self) -> None:
        objective = _objective(ObjectiveDirection.TARGET)  # target 0.5

        assert directional_fitness(0.5, objective) == 0.0
        assert directional_fitness(0.3, objective) == pytest.approx(-0.2)
        assert directional_fitness(0.7, objective) == pytest.approx(-0.2)

    def test_a_missing_value_stays_missing(self) -> None:
        """A stand-in number would be ranked against real ones and could win."""
        assert directional_fitness(None, _objective()) is None


class TestFitnessContract:
    def test_an_evaluated_candidate_must_carry_a_number(self) -> None:
        with pytest.raises(ValueError, match="needs a scalar fitness"):
            CandidateFitness(
                candidate_id="c1", genome=_genome(), status=FitnessStatus.EVALUATED
            )

    def test_a_failure_must_carry_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must carry a reason"):
            CandidateFitness(
                candidate_id="c1", genome=_genome(), status=FitnessStatus.EVALUATION_FAILED
            )

    def test_only_an_evaluated_candidate_is_usable(self) -> None:
        usable = CandidateFitness(
            candidate_id="c1", genome=_genome(), status=FitnessStatus.EVALUATED,
            scalar_fitness=0.4, primary_objective_id="o1",
        )
        unusable = CandidateFitness(
            candidate_id="c2", genome=_genome(genome_id="c2"),
            status=FitnessStatus.NOT_MEASURABLE, reason="no value",
        )

        assert usable.is_usable is True
        assert unusable.is_usable is False

    def test_a_margin_inside_the_noise_floor_does_not_count_as_better(self) -> None:
        """A bare `>` would promote noise into a reported winner."""
        best = CandidateFitness(
            candidate_id="a", genome=_genome(genome_id="a"),
            status=FitnessStatus.EVALUATED, scalar_fitness=0.400,
            primary_objective_id="o1",
        )
        marginal = CandidateFitness(
            candidate_id="b", genome=_genome(genome_id="b"),
            status=FitnessStatus.EVALUATED, scalar_fitness=0.405,
            primary_objective_id="o1",
        )
        clear = CandidateFitness(
            candidate_id="c", genome=_genome(genome_id="c"),
            status=FitnessStatus.EVALUATED, scalar_fitness=0.440,
            primary_objective_id="o1",
        )

        assert marginal.beats(best, minimum_improvement=0.0157) is False
        assert clear.beats(best, minimum_improvement=0.0157) is True

    def test_anything_usable_beats_nothing(self) -> None:
        first = CandidateFitness(
            candidate_id="a", genome=_genome(), status=FitnessStatus.EVALUATED,
            scalar_fitness=-5.0, primary_objective_id="o1",
        )

        assert first.beats(None, minimum_improvement=0.0157) is True


# ---------------------------------------------------------------------------
# Steps 11, 50 - the evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_produces_a_fitness_from_a_completed_run(
        self, clip: Path, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(clip, tmp_path, FakeOrchestrator(candidate_value=0.42))

        fitness = evaluator.evaluate(_genome())

        assert fitness.status is FitnessStatus.EVALUATED
        assert fitness.scalar_fitness == pytest.approx(0.42)
        assert fitness.raw_objectives["o1"] == pytest.approx(0.42)
        assert fitness.candidate_run_id == "cand-run"
        assert fitness.compute.tribe_runs == 1

    def test_builds_a_request_phase_8_validates(self, clip: Path, tmp_path: Path) -> None:
        """Construction is the proof: the model rejects an inconsistent one."""
        orchestrator = FakeOrchestrator()
        evaluator = _evaluator(clip, tmp_path, orchestrator)

        evaluator.evaluate(_genome())

        request = orchestrator.requests[0]
        assert request.parent_run_id == "root-run"  # type: ignore[attr-defined]
        assert request.optimization_request_key == "search-s1"  # type: ignore[attr-defined]
        assert request.binding.adapter_id == "ffmpeg-materialized"  # type: ignore[attr-defined]

    def test_the_binding_marks_the_candidate_as_machine_made(
        self, clip: Path, tmp_path: Path
    ) -> None:
        """Nothing human-supplied ever carries this adapter id."""
        orchestrator = FakeOrchestrator()
        _evaluator(clip, tmp_path, orchestrator).evaluate(_genome())

        binding = orchestrator.requests[0].binding  # type: ignore[attr-defined]
        assert binding.method == "deterministic_ffmpeg_edit"
        assert binding.source_sha256 != binding.variant_sha256

    def test_minimize_objective_flips_the_scalar(
        self, clip: Path, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(
            clip, tmp_path, FakeOrchestrator(candidate_value=0.42),
            objective=_objective(ObjectiveDirection.MINIMIZE),
        )

        fitness = evaluator.evaluate(_genome())

        assert fitness.scalar_fitness == pytest.approx(-0.42)
        assert fitness.raw_objectives["o1"] == pytest.approx(0.42)

    def test_a_neutral_genome_is_rejected_before_any_compute(
        self, clip: Path, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator()
        evaluator = _evaluator(clip, tmp_path, orchestrator)

        fitness = evaluator.evaluate(_genome(gain=0.0))

        assert fitness.status is FitnessStatus.REJECTED
        assert orchestrator.requests == []
        assert fitness.compute.tribe_runs == 0

    def test_a_constraint_violation_is_rejected_before_any_compute(
        self, clip: Path, tmp_path: Path
    ) -> None:
        orchestrator = FakeOrchestrator()
        evaluator = _evaluator(
            clip, tmp_path, orchestrator,
            constraints=OptimizationConstraints(frozen_modalities=("audio",)),
        )

        fitness = evaluator.evaluate(_genome())

        assert fitness.status is FitnessStatus.REJECTED
        assert "frozen" in fitness.reason
        assert orchestrator.requests == []

    def test_a_pipeline_exception_becomes_a_result_not_a_crash(
        self, clip: Path, tmp_path: Path
    ) -> None:
        """Step 54: one bad candidate must not end a search costing hours."""
        evaluator = _evaluator(
            clip, tmp_path, FakeOrchestrator(raise_exc=RuntimeError("tribe died"))
        )

        fitness = evaluator.evaluate(_genome())

        assert fitness.status is FitnessStatus.EVALUATION_FAILED
        assert "tribe died" in fitness.reason
        assert fitness.is_usable is False

    def test_a_failed_run_reports_the_stage_message(
        self, clip: Path, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(clip, tmp_path, FakeOrchestrator(status=RunStatus.FAILED))

        fitness = evaluator.evaluate(_genome())

        assert fitness.status is FitnessStatus.EVALUATION_FAILED
        assert "inference blew up" in fitness.reason

    def test_an_unmeasurable_objective_is_not_a_failure_but_is_not_usable(
        self, clip: Path, tmp_path: Path
    ) -> None:
        evaluator = _evaluator(clip, tmp_path, FakeOrchestrator(candidate_value=None))

        fitness = evaluator.evaluate(_genome())

        assert fitness.status is FitnessStatus.NOT_MEASURABLE
        assert fitness.is_usable is False
        assert "cannot be ranked" in fitness.reason

    def test_re_evaluating_the_same_genome_reuses_the_media(
        self, clip: Path, tmp_path: Path
    ) -> None:
        """The second encode is free; only the inference would cost anything."""
        evaluator = _evaluator(clip, tmp_path, FakeOrchestrator())

        first = evaluator.evaluate(_genome())
        second = evaluator.evaluate(_genome())

        assert first.variant_sha256 == second.variant_sha256
        assert second.compute.served_from_cache is True
        assert second.compute.tribe_runs == 0


# ---------------------------------------------------------------------------
# Steps 14-15, 48-49 - random search and the registry
# ---------------------------------------------------------------------------


class TestRandomSearch:
    def test_proposes_valid_points_inside_the_space(self) -> None:
        space = _space()
        strategy = RandomSearch()
        strategy.initialize(space, seed=1)

        genomes = strategy.propose(10, generation=0)

        assert len(genomes) == 10
        for genome in genomes:
            genome.validate_against(space)

    def test_is_reproducible_from_its_seed(self) -> None:
        """Two runs of the same search must propose the same candidates."""
        def run() -> list[dict[str, float]]:
            strategy = RandomSearch()
            strategy.initialize(_space(), seed=99)
            return [dict(g.values) for g in strategy.propose(8, generation=0)]

        assert run() == run()

    def test_different_seeds_explore_differently(self) -> None:
        def run(seed: int) -> list[dict[str, float]]:
            strategy = RandomSearch()
            strategy.initialize(_space(), seed=seed)
            return [dict(g.values) for g in strategy.propose(8, generation=0)]

        assert run(1) != run(2)

    def test_never_proposes_the_same_point_twice(self) -> None:
        """A duplicate's entire cost is the evaluation it would trigger."""
        space = _space()
        strategy = RandomSearch()
        strategy.initialize(space, seed=5)

        genomes = strategy.propose(20, generation=0)
        fingerprints = {genome.fingerprint(space) for genome in genomes}

        assert len(fingerprints) == len(genomes)

    def test_never_proposes_the_unedited_root(self) -> None:
        """Its score is already known and would cost a full inference again."""
        space = _space()
        strategy = RandomSearch()
        strategy.initialize(space, seed=3)

        for genome in strategy.propose(30, generation=0):
            assert genome.is_neutral(space) is False

    def test_reports_exhaustion_rather_than_looping_forever(self) -> None:
        """A tiny space runs out; the search must notice instead of hanging."""
        from blackmirror.search.space import categorical

        space = ContentSearchSpace(
            parameters=(categorical("speed", Op.SPEED, (0.9, 1.1)),)
        )
        strategy = RandomSearch()
        strategy.initialize(space, seed=0)

        genomes = strategy.propose(10, generation=0)
        stop, reason = strategy.should_stop()

        assert len(genomes) == 2
        assert stop is True
        assert "no further distinct points" in reason

    def test_never_returns_more_than_requested(self) -> None:
        strategy = RandomSearch()
        strategy.initialize(_space(), seed=0)

        assert len(strategy.propose(3, generation=0)) == 3
        assert strategy.propose(0, generation=0) == ()

    def test_state_round_trip_continues_the_same_stream(self) -> None:
        """Resume continues the stream rather than restarting it.

        A restart would re-propose the points already evaluated, spending the
        budget twice to learn nothing new.
        """
        space = _space()

        # Run four, capture the state, then run four more: that second four is
        # what a resumed search must reproduce exactly.
        original = RandomSearch()
        original.initialize(space, seed=7)
        original.propose(4, generation=0)
        checkpoint = original.get_state()
        expected = [dict(g.values) for g in original.propose(4, generation=1)]

        restored = RandomSearch()
        restored.initialize(space, seed=7)
        restored.set_state(checkpoint)

        assert [dict(g.values) for g in restored.propose(4, generation=1)] == expected

    def test_state_is_json_serializable(self) -> None:
        """Checkpoints are JSON; a tuple from getstate would not survive."""
        import json

        strategy = RandomSearch()
        strategy.initialize(_space(), seed=2)
        strategy.propose(3, generation=0)

        json.dumps(strategy.get_state())


class TestRegistry:
    def test_creates_the_strategies_that_exist(self) -> None:
        registry = default_registry()

        strategy = registry.create(SearchStrategyName.RANDOM_SEARCH)

        assert isinstance(strategy, RandomSearch)

    def test_refuses_a_strategy_it_does_not_hold(self) -> None:
        """Advertising an unbuilt algorithm would let it silently do nothing."""
        registry = SearchStrategyRegistry({SearchStrategyName.RANDOM_SEARCH: RandomSearch})

        with pytest.raises(ValueError, match="not implemented yet"):
            registry.create(SearchStrategyName.EVOLUTIONARY_SEARCH)

    def test_lists_every_strategy_it_can_build(self) -> None:
        available = default_registry().available()

        assert set(available) == set(SearchStrategyName)

    def test_every_advertised_strategy_can_actually_be_created(self) -> None:
        """A registry entry that cannot be built is worse than a missing one."""
        registry = default_registry()

        for name in registry.available():
            assert registry.create(name).name is name

    def test_a_strategy_can_be_registered(self) -> None:
        registry = SearchStrategyRegistry({})
        registry.register(SearchStrategyName.RANDOM_SEARCH, RandomSearch)

        assert SearchStrategyName.RANDOM_SEARCH in registry


class TestDeduplication:
    def test_drops_repeated_points_keeping_the_first(self) -> None:
        space = _space()
        genomes = (_genome(3.0, "a"), _genome(3.0, "b"), _genome(4.0, "c"))

        unique = unique_genomes(genomes, space)

        assert [genome.genome_id for genome in unique] == ["a", "c"]


class TestComputeCost:
    def test_records_what_an_evaluation_consumed(self) -> None:
        cost = ComputeCost(wall_seconds=3120.0, tribe_runs=1)

        assert cost.wall_seconds == 3120.0
        assert cost.served_from_cache is False


def test_fitness_timestamp_is_timezone_aware() -> None:
    """Naive timestamps compare wrongly against Phase 8's aware ones."""
    fitness = CandidateFitness(
        candidate_id="c1", genome=_genome(), status=FitnessStatus.REJECTED, reason="x"
    )

    assert fitness.evaluated_at.tzinfo is not None
    assert fitness.evaluated_at.tzinfo.utcoffset(fitness.evaluated_at) == dt.timedelta(0)


class TestBestTracking:
    """Which candidate is best must not depend on the order results arrived."""

    def _fit(self, cid: str, value: float) -> CandidateFitness:
        return CandidateFitness(
            candidate_id=cid, genome=_genome(genome_id=cid),
            status=FitnessStatus.EVALUATED, scalar_fitness=value,
            primary_objective_id="o1",
        )

    def test_best_is_the_argmax_regardless_of_order(self) -> None:
        """Gating promotion on the noise floor made this order-dependent.

        A stronger candidate arriving second would fail to displace a weaker
        first one, so a different seed reported a different winner among
        identical numbers.
        """
        from blackmirror.search.fitness import best_of

        a, b, c = self._fit("a", 0.291), self._fit("b", 0.296), self._fit("c", 0.299)

        assert best_of([a, b, c]).candidate_id == "c"
        assert best_of([c, b, a]).candidate_id == "c"
        assert best_of([b, c, a]).candidate_id == "c"

    def test_best_ignores_unusable_candidates(self) -> None:
        from blackmirror.search.fitness import best_of

        failed_one = CandidateFitness(
            candidate_id="x", genome=_genome(genome_id="x"),
            status=FitnessStatus.EVALUATION_FAILED, reason="died",
        )

        assert best_of([failed_one, self._fit("a", 0.1)]).candidate_id == "a"

    def test_no_usable_candidate_means_no_best(self) -> None:
        from blackmirror.search.fitness import best_of

        assert best_of([]) is None

    def test_ties_within_the_noise_floor_are_reported_beside_the_best(self) -> None:
        """Naming a single winner without this would overstate the finding."""
        from blackmirror.search.fitness import tied_with_best

        results = [self._fit("a", 0.291), self._fit("b", 0.296), self._fit("c", 0.299)]

        tied = tied_with_best(results, minimum_improvement=0.0157)

        assert {item.candidate_id for item in tied} == {"a", "b"}

    def test_a_clear_winner_has_no_ties(self) -> None:
        from blackmirror.search.fitness import tied_with_best

        results = [self._fit("a", 0.10), self._fit("b", 0.50)]

        assert tied_with_best(results, minimum_improvement=0.0157) == ()

    def test_being_ahead_and_being_measurably_ahead_are_different(self) -> None:
        higher = self._fit("b", 0.296)
        lower = self._fit("a", 0.291)

        assert higher.is_better_than(lower) is True
        assert higher.beats(lower, minimum_improvement=0.0157) is False
        assert higher.within_noise_of(lower, minimum_improvement=0.0157) is True
