"""Steps 16-35, 43-47, 77-87: strategies, tracking, stopping, orchestration.

The strategies are exercised against synthetic objectives with known optima, as
Steps 77-79 specify. That is the only way to test a search algorithm honestly:
on the real pipeline the optimum is unknown, so "it found a good point" would be
unfalsifiable. Here the answer is known in advance and the algorithm either
reaches it or does not.

No claim is made that any strategy is universally better. What these prove is
that each one does the thing its name says.
"""

from __future__ import annotations

from random import Random

import pytest

from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.search.budget import BudgetLedger, SearchBudget
from blackmirror.search.fitness import CandidateFitness, FitnessStatus
from blackmirror.search.genome import CandidateGenome, GenomeOrigin
from blackmirror.search.mutation import (
    MutationMagnitude,
    MutationRegistry,
    crossover,
    neighbours,
)
from blackmirror.search.schemas import SearchConfig, StoppingReason
from blackmirror.search.space import ContentSearchSpace, categorical, continuous
from blackmirror.search.strategies import (
    BeamSearch,
    EpsilonGreedy,
    EvolutionarySearch,
    GridSearch,
    HillClimbing,
    LocalSearch,
)
from blackmirror.search.strategy import RandomSearch
from blackmirror.search.tracking import (
    SearchEventType,
    SearchTracker,
    StoppingPolicy,
    budget_warnings,
)

# ---------------------------------------------------------------------------
# Synthetic surfaces with known optima (Steps 77-79)
# ---------------------------------------------------------------------------

#: f(x) = -(x - 3)^2 over gain in [-10, 10]. Optimum at x = 3.
def _parabola_space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),)
    )


def _parabola(values: dict[str, float]) -> float:
    return -((values["gain"] - 3.0) ** 2)


def _fitness(genome: CandidateGenome, value: float) -> CandidateFitness:
    return CandidateFitness(
        candidate_id=genome.genome_id,
        genome=genome,
        status=FitnessStatus.EVALUATED,
        scalar_fitness=value,
        primary_objective_id="o1",
        raw_objectives={"o1": value},
    )


def _drive(strategy, surface, *, rounds: int, batch: int = 4):  # type: ignore[no-untyped-def]
    """Run a strategy against a synthetic surface, returning every result."""
    everything: list[CandidateFitness] = []
    for generation in range(rounds):
        proposed = strategy.propose(batch, generation)
        if not proposed:
            break
        batch_results = [_fitness(g, surface(g.values)) for g in proposed]
        strategy.observe(batch_results)
        everything.extend(batch_results)
    return everything


# ---------------------------------------------------------------------------
# Steps 28-30 - mutation, neighbourhood, crossover
# ---------------------------------------------------------------------------


class TestMutation:
    def test_a_mutation_always_changes_something(self) -> None:
        """A child identical to its parent would waste an evaluation slot."""
        space = _parabola_space()
        registry = MutationRegistry()
        parent = CandidateGenome(genome_id="p", values={"gain": 1.0})
        rng = Random(0)

        for index in range(50):
            child = registry.mutate(parent, space, rng, genome_id=f"c{index}", rate=0.0)
            assert child.values != parent.values

    def test_mutation_stays_inside_the_domain(self) -> None:
        space = _parabola_space()
        registry = MutationRegistry()
        parent = CandidateGenome(genome_id="p", values={"gain": 9.9})
        rng = Random(1)

        for index in range(100):
            child = registry.mutate(
                parent, space, rng, genome_id=f"c{index}",
                magnitude=MutationMagnitude.LARGE,
            )
            child.validate_against(space)

    def test_larger_magnitudes_move_further_on_average(self) -> None:
        space = _parabola_space()
        registry = MutationRegistry()
        parent = CandidateGenome(genome_id="p", values={"gain": 0.0})

        def spread(magnitude: MutationMagnitude) -> float:
            rng = Random(4)
            moves = [
                abs(
                    registry.mutate(
                        parent, space, rng, genome_id=f"c{i}", magnitude=magnitude
                    ).values["gain"]
                )
                for i in range(200)
            ]
            return sum(moves) / len(moves)

        assert spread(MutationMagnitude.SMALL) < spread(MutationMagnitude.MEDIUM)
        assert spread(MutationMagnitude.MEDIUM) < spread(MutationMagnitude.LARGE)

    def test_categorical_mutation_never_returns_the_same_choice(self) -> None:
        space = ContentSearchSpace(
            parameters=(categorical("speed", Op.SPEED, (0.9, 1.0, 1.1)),)
        )
        registry = MutationRegistry()
        parent = CandidateGenome(genome_id="p", values={"speed": 1.0})
        rng = Random(2)

        for index in range(30):
            child = registry.mutate(parent, space, rng, genome_id=f"c{index}")
            assert child.values["speed"] != 1.0

    def test_neighbours_move_one_parameter_at_a_time(self) -> None:
        """Attributable moves: exactly one thing differs from the parent."""
        space = ContentSearchSpace(
            parameters=(
                continuous("gain", Op.AUDIO_GAIN_DB, -6.0, 6.0),
                continuous("brightness", Op.BRIGHTNESS, -0.2, 0.2),
            )
        )
        parent = CandidateGenome(genome_id="p", values={"gain": 0.0, "brightness": 0.0})

        for point in neighbours(parent, space):
            differing = [
                name for name, value in point.items() if value != parent.values[name]
            ]
            assert len(differing) == 1

    def test_neighbours_never_include_the_parent(self) -> None:
        space = _parabola_space()
        parent = CandidateGenome(genome_id="p", values={"gain": 1.0})

        assert all(point != parent.values for point in neighbours(parent, space))

    def test_crossover_takes_every_value_from_a_parent(self) -> None:
        """A child can only sit at a combination its parents already justified."""
        space = ContentSearchSpace(
            parameters=(
                continuous("gain", Op.AUDIO_GAIN_DB, -6.0, 6.0),
                continuous("brightness", Op.BRIGHTNESS, -0.2, 0.2),
            )
        )
        left = CandidateGenome(genome_id="l", values={"gain": -5.0, "brightness": 0.15})
        right = CandidateGenome(genome_id="r", values={"gain": 4.0, "brightness": -0.1})

        child = crossover(left, right, space, Random(3), genome_id="c")

        for name, value in child.values.items():
            assert value in (left.values[name], right.values[name])
        assert child.parent_genome_ids == ("l", "r")
        assert child.origin is GenomeOrigin.CROSSOVER


# ---------------------------------------------------------------------------
# Step 16 - grid
# ---------------------------------------------------------------------------


class TestGridSearch:
    def test_refuses_a_grid_larger_than_its_limit(self) -> None:
        """60 points at ~52 minutes each is over two days of compute."""
        space = ContentSearchSpace(
            parameters=(
                continuous("gain", Op.AUDIO_GAIN_DB, -6.0, 6.0),
                continuous("brightness", Op.BRIGHTNESS, -0.2, 0.2),
            )
        )
        strategy = GridSearch(steps=8, max_points=32)

        with pytest.raises(ValueError, match="above the limit"):
            strategy.initialize(space, seed=0)

    def test_covers_the_grid_and_then_stops(self) -> None:
        strategy = GridSearch(steps=5, max_points=64)
        strategy.initialize(_parabola_space(), seed=0)

        results = _drive(strategy, _parabola, rounds=10, batch=2)
        stop, reason = strategy.should_stop()

        # Five grid points, but 0.0 is the neutral gain and would rebuild the
        # root, so it is skipped rather than costing an inference.
        assert len(results) == 4
        assert 0.0 not in {item.genome.values["gain"] for item in results}
        assert stop is True
        assert "every point in the grid" in reason

    def test_finds_the_optimum_when_the_grid_contains_it(self) -> None:
        """A grid over [-10, 10] in 5 steps hits -10,-5,0,5,10; 5 is nearest 3."""
        strategy = GridSearch(steps=5, max_points=64)
        strategy.initialize(_parabola_space(), seed=0)

        results = _drive(strategy, _parabola, rounds=10, batch=5)
        best = max(results, key=lambda item: item.scalar_fitness or -1e9)

        assert best.genome.values["gain"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Steps 17-19 - local search and hill climbing
# ---------------------------------------------------------------------------


class TestLocalSearchAndHillClimbing:
    def test_hill_climbing_moves_toward_the_optimum(self) -> None:
        """Step 78: from x = -8, climbing must get closer to x = 3.

        The start is fixed rather than random. With a random one the test
        sometimes begins beside the optimum, where correct behaviour is to
        plateau immediately, and the assertion would be testing luck.
        """
        strategy = HillClimbing(magnitude=MutationMagnitude.SMALL)
        strategy.initialize(_parabola_space(), seed=5)
        start = CandidateGenome(genome_id="start", values={"gain": -8.0})
        strategy.observe([_fitness(start, _parabola(start.values))])

        results = _drive(strategy, _parabola, rounds=40, batch=3)
        best = max([*results, _fitness(start, _parabola(start.values))],
                   key=lambda item: item.scalar_fitness or -1e9)

        assert best.scalar_fitness > _parabola(start.values)
        assert abs(best.genome.values["gain"] - 3.0) < 8.0

    def test_hill_climbing_can_be_trapped_in_a_local_optimum(self) -> None:
        """Step 19, demonstrated rather than only asserted in a docstring.

        On a surface with a tall narrow peak and a broad low one, a climber
        started in the low basin stays there. Nothing in its output would
        distinguish that answer from a global best, which is why the other
        strategies exist.
        """
        def two_peaks(values: dict[str, float]) -> float:
            x = values["gain"]
            return max(1.0 - abs(x + 7.0), 5.0 - abs(x - 7.0) * 5.0)

        strategy = HillClimbing(magnitude=MutationMagnitude.SMALL)
        strategy.initialize(_parabola_space(), seed=5)
        start = CandidateGenome(genome_id="start", values={"gain": -7.0})
        strategy.observe([_fitness(start, two_peaks(start.values))])

        results = _drive(strategy, two_peaks, rounds=30, batch=3)
        best = max([*results, _fitness(start, two_peaks(start.values))],
                   key=lambda item: item.scalar_fitness or -1e9)

        # The global optimum is 5.0 at x = +7; the climber never leaves x < 0.
        assert best.scalar_fitness < 5.0
        assert best.genome.values["gain"] < 0.0

    def test_hill_climbing_stops_on_a_plateau(self) -> None:
        """Step 19's limitation, made visible rather than hidden."""
        strategy = HillClimbing(magnitude=MutationMagnitude.SMALL)
        strategy.initialize(_parabola_space(), seed=1)

        _drive(strategy, lambda values: 0.5, rounds=6, batch=3)
        stop, reason = strategy.should_stop()

        assert stop is True
        assert "local optimum" in reason

    def test_local_search_restarts_where_hill_climbing_stops(self) -> None:
        """The difference between the two: one escapes its basin, one does not.

        On a flat surface neither can improve. Hill climbing calls that a local
        optimum and stops; local search restarts elsewhere and keeps looking.
        """
        flat = LocalSearch()
        flat.initialize(_parabola_space(), seed=1)
        climber = HillClimbing()
        climber.initialize(_parabola_space(), seed=1)

        _drive(flat, lambda values: 0.5, rounds=6, batch=3)
        _drive(climber, lambda values: 0.5, rounds=6, batch=3)

        assert flat.should_stop()[0] is False
        assert climber.should_stop()[0] is True

    def test_local_search_explores_around_the_best_point(self) -> None:
        space = _parabola_space()
        strategy = LocalSearch(magnitude=MutationMagnitude.SMALL)
        strategy.initialize(space, seed=2)

        _drive(strategy, _parabola, rounds=6, batch=3)
        best = strategy.best
        assert best is not None
        proposals = strategy.propose(3, generation=7)

        for genome in proposals:
            differing = [
                name
                for name, value in genome.values.items()
                if value != best.genome.values[name]
            ]
            assert len(differing) <= 1


# ---------------------------------------------------------------------------
# Steps 20-21 - beam
# ---------------------------------------------------------------------------


class TestBeamSearch:
    def test_keeps_exactly_the_top_k(self) -> None:
        """Step 79: the retained set must be the best K, not the newest K."""
        strategy = BeamSearch(beam_width=3)
        strategy.initialize(_parabola_space(), seed=4)

        results = _drive(strategy, _parabola, rounds=4, batch=5)
        kept = {item.candidate_id for item in strategy.beam}
        ranked = sorted(results, key=lambda i: i.scalar_fitness or -1e9, reverse=True)

        assert len(kept) == 3
        assert kept == {item.candidate_id for item in ranked[:3]}

    def test_records_why_a_candidate_was_pruned(self) -> None:
        """Step 71: elimination must be explainable, not inferred from absence."""
        strategy = BeamSearch(beam_width=2)
        strategy.initialize(_parabola_space(), seed=4)

        _drive(strategy, _parabola, rounds=3, batch=4)

        assert strategy.pruned
        assert "beam width 2" in strategy.pruned[0][1]

    def test_a_wider_beam_retains_more(self) -> None:
        def run(width: int) -> int:
            strategy = BeamSearch(beam_width=width)
            strategy.initialize(_parabola_space(), seed=6)
            _drive(strategy, _parabola, rounds=3, batch=4)
            return len(strategy.beam)

        assert run(1) == 1
        assert run(4) == 4

    def test_rejects_a_beam_width_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            BeamSearch(beam_width=0)

    def test_reaches_the_optimum_region_with_enough_width(self) -> None:
        strategy = BeamSearch(beam_width=3, magnitude=MutationMagnitude.MEDIUM)
        strategy.initialize(_parabola_space(), seed=8)

        results = _drive(strategy, _parabola, rounds=12, batch=4)
        best = max(results, key=lambda item: item.scalar_fitness or -1e9)

        assert abs(best.genome.values["gain"] - 3.0) < 2.0


# ---------------------------------------------------------------------------
# Steps 24-27 - exploration and exploitation
# ---------------------------------------------------------------------------


class TestEpsilonGreedy:
    def test_exploration_decays_but_never_below_the_floor(self) -> None:
        strategy = EpsilonGreedy(exploration_rate=0.4, decay=0.5, minimum_rate=0.1)

        assert strategy.rate_at(0) == pytest.approx(0.4)
        assert strategy.rate_at(1) == pytest.approx(0.2)
        assert strategy.rate_at(9) == pytest.approx(0.1)

    def test_records_each_explore_or_exploit_decision(self) -> None:
        """Step 70: the reason a candidate exists must be recoverable."""
        strategy = EpsilonGreedy(exploration_rate=0.5)
        strategy.initialize(_parabola_space(), seed=3)

        _drive(strategy, _parabola, rounds=4, batch=3)

        assert strategy.decisions
        assert any("exploit" in entry for entry in strategy.decisions)
        assert any("explore" in entry for entry in strategy.decisions)

    def test_pure_exploitation_stays_near_the_best(self) -> None:
        strategy = EpsilonGreedy(exploration_rate=0.0, minimum_rate=0.0)
        strategy.initialize(_parabola_space(), seed=3)

        results = _drive(strategy, _parabola, rounds=10, batch=3)
        best = max(results, key=lambda item: item.scalar_fitness or -1e9)

        assert abs(best.genome.values["gain"] - 3.0) < 3.0

    def test_is_reproducible_from_its_seed(self) -> None:
        def run() -> list[float]:
            strategy = EpsilonGreedy(exploration_rate=0.3)
            strategy.initialize(_parabola_space(), seed=21)
            return [
                item.genome.values["gain"]
                for item in _drive(strategy, _parabola, rounds=5, batch=3)
            ]

        assert run() == run()


# ---------------------------------------------------------------------------
# Steps 31-35 - evolutionary
# ---------------------------------------------------------------------------


class TestEvolutionarySearch:
    def test_keeps_the_best_across_generations(self) -> None:
        """Elitism: a population that can lose its best answer is worse."""
        strategy = EvolutionarySearch(population_size=4, elite_count=1)
        strategy.initialize(_parabola_space(), seed=7)

        results = _drive(strategy, _parabola, rounds=8, batch=4)
        best = max(results, key=lambda item: item.scalar_fitness or -1e9)

        assert strategy.elites[0].candidate_id == best.candidate_id

    def test_population_never_exceeds_its_size(self) -> None:
        strategy = EvolutionarySearch(population_size=5)
        strategy.initialize(_parabola_space(), seed=7)

        _drive(strategy, _parabola, rounds=10, batch=4)

        assert len(strategy.population) <= 5

    def test_improves_over_its_first_generation(self) -> None:
        strategy = EvolutionarySearch(population_size=6, elite_count=2)
        strategy.initialize(_parabola_space(), seed=9)

        results = _drive(strategy, _parabola, rounds=12, batch=4)
        first_gen = [item.scalar_fitness for item in results[:4]]
        last_gen = [item.scalar_fitness for item in results[-4:]]

        assert max(last_gen) > max(first_gen)

    def test_refuses_an_all_elite_population(self) -> None:
        """With no room for a new candidate the population cannot change."""
        with pytest.raises(ValueError, match="room for at least one"):
            EvolutionarySearch(population_size=2, elite_count=2)


# ---------------------------------------------------------------------------
# Every strategy - shared guarantees
# ---------------------------------------------------------------------------


ALL_STRATEGIES = [
    lambda: RandomSearch(),
    lambda: GridSearch(steps=5, max_points=64),
    lambda: LocalSearch(),
    lambda: HillClimbing(),
    lambda: BeamSearch(beam_width=2),
    lambda: EpsilonGreedy(),
    lambda: EvolutionarySearch(population_size=4),
]


@pytest.mark.parametrize("factory", ALL_STRATEGIES)
class TestEveryStrategy:
    def test_never_proposes_the_same_point_twice(self, factory) -> None:  # type: ignore[no-untyped-def]
        """A duplicate's whole cost is the evaluation it would trigger."""
        space = _parabola_space()
        strategy = factory()
        strategy.initialize(space, seed=12)

        results = _drive(strategy, _parabola, rounds=6, batch=3)
        fingerprints = [item.genome.fingerprint(space) for item in results]

        assert len(fingerprints) == len(set(fingerprints))

    def test_never_proposes_a_point_outside_the_space(self, factory) -> None:  # type: ignore[no-untyped-def]
        space = _parabola_space()
        strategy = factory()
        strategy.initialize(space, seed=13)

        for item in _drive(strategy, _parabola, rounds=6, batch=3):
            item.genome.validate_against(space)

    def test_never_proposes_the_unedited_root(self, factory) -> None:  # type: ignore[no-untyped-def]
        """Its score is known; re-measuring it costs a full inference."""
        space = _parabola_space()
        strategy = factory()
        strategy.initialize(space, seed=14)

        for item in _drive(strategy, _parabola, rounds=6, batch=3):
            assert item.genome.is_neutral(space) is False

    def test_never_returns_more_than_requested(self, factory) -> None:  # type: ignore[no-untyped-def]
        strategy = factory()
        strategy.initialize(_parabola_space(), seed=15)

        assert len(strategy.propose(2, 0)) <= 2

    def test_state_is_json_serializable(self, factory) -> None:  # type: ignore[no-untyped-def]
        import json

        strategy = factory()
        strategy.initialize(_parabola_space(), seed=16)
        _drive(strategy, _parabola, rounds=2, batch=2)

        json.dumps(strategy.get_state())

    def test_is_reproducible_from_its_seed(self, factory) -> None:  # type: ignore[no-untyped-def]
        def run() -> list[float]:
            strategy = factory()
            strategy.initialize(_parabola_space(), seed=17)
            return [
                item.genome.values["gain"]
                for item in _drive(strategy, _parabola, rounds=4, batch=3)
            ]

        assert run() == run()


# ---------------------------------------------------------------------------
# Steps 39-46, 63 - tracking and stopping
# ---------------------------------------------------------------------------


class TestTracking:
    def _record(self, tracker: SearchTracker, gain: float, value: float, cid: str) -> None:
        genome = CandidateGenome(genome_id=cid, values={"gain": gain})
        tracker.record(
            _fitness(genome, value), generation=0, ledger=BudgetLedger()
        )

    def test_trajectory_is_monotonic_in_best_so_far(self) -> None:
        """The curve that answers "how much did this cost to find"."""
        tracker = SearchTracker(root_fitness=0.0)
        for index, value in enumerate([0.1, 0.05, 0.3, 0.2, 0.4]):
            self._record(tracker, float(index), value, f"c{index}")

        bests = [point.best_fitness for point in tracker.trajectory]

        assert bests == sorted(bests)
        assert bests[-1] == pytest.approx(0.4)

    def test_records_when_a_new_best_appears(self) -> None:
        tracker = SearchTracker()
        self._record(tracker, 1.0, 0.1, "a")
        self._record(tracker, 2.0, 0.5, "b")

        new_bests = [p.candidate_id for p in tracker.trajectory if p.is_new_best]

        assert new_bests == ["a", "b"]
        assert tracker.best is not None and tracker.best.candidate_id == "b"

    def test_best_record_captures_the_cost_at_discovery(self) -> None:
        """Step 39: what it had spent when the winner turned up."""
        tracker = SearchTracker(root_fitness=0.0)
        genome = CandidateGenome(genome_id="a", values={"gain": 1.0})
        ledger = BudgetLedger().with_evaluation(wall_seconds=3120.0)

        tracker.record(_fitness(genome, 0.5), generation=2, ledger=ledger)

        assert tracker.best is not None
        assert tracker.best.wall_seconds_at_discovery == 3120.0
        assert tracker.best.tribe_runs_at_discovery == 1
        assert tracker.best.improvement_over_root == pytest.approx(0.5)

    def test_events_are_appended_in_order(self) -> None:
        tracker = SearchTracker()
        tracker.event(SearchEventType.SEARCH_STARTED, "go")
        self._record(tracker, 1.0, 0.1, "a")

        types = [event.type for event in tracker.events]

        assert types[0] is SearchEventType.SEARCH_STARTED
        assert SearchEventType.CANDIDATE_EVALUATED in types
        assert SearchEventType.NEW_BEST_FOUND in types

    def test_metrics_flag_an_improvement_inside_the_noise_floor(self) -> None:
        tracker = SearchTracker(root_fitness=0.30)
        self._record(tracker, 1.0, 0.305, "a")

        metrics = tracker.metrics(BudgetLedger(), minimum_improvement=0.0157)

        assert metrics.absolute_improvement == pytest.approx(0.005)
        assert metrics.improvement_is_resolvable is False

    def test_metrics_report_the_failure_rate(self) -> None:
        tracker = SearchTracker()
        genome = CandidateGenome(genome_id="x", values={"gain": 1.0})
        tracker.record(
            CandidateFitness(
                candidate_id="x", genome=genome,
                status=FitnessStatus.EVALUATION_FAILED, reason="died",
            ),
            generation=0, ledger=BudgetLedger(),
        )
        self._record(tracker, 1.0, 0.5, "y")

        metrics = tracker.metrics(BudgetLedger(), minimum_improvement=0.0157)

        assert metrics.failure_rate == pytest.approx(0.5)


class TestStoppingPolicy:
    def _policy(self, **config: object) -> StoppingPolicy:
        return StoppingPolicy(
            SearchConfig(**config),  # type: ignore[arg-type]
            SearchBudget(max_candidates=100, max_generations=100),
        )

    def test_patience_fires_after_generations_without_a_believable_gain(self) -> None:
        """Step 82. Counting any gain would chase drift below the floor."""
        policy = self._policy(patience=3, minimum_improvement=0.01)
        genome = CandidateGenome(genome_id="a", values={"gain": 1.0})
        best = _fitness(genome, 0.5)

        for _ in range(4):
            policy.note_generation(best)
        stop = policy.check(BudgetLedger(), best)

        assert stop is not None
        assert stop[0] is StoppingReason.NO_IMPROVEMENT

    def test_a_real_gain_resets_patience(self) -> None:
        policy = self._policy(patience=2, minimum_improvement=0.01)
        genome = CandidateGenome(genome_id="a", values={"gain": 1.0})

        policy.note_generation(_fitness(genome, 0.10))
        policy.note_generation(_fitness(genome, 0.10))
        policy.note_generation(_fitness(genome, 0.50))

        assert policy.generations_without_gain == 0

    def test_a_gain_below_the_floor_does_not_reset_patience(self) -> None:
        policy = self._policy(patience=5, minimum_improvement=0.0157)
        genome = CandidateGenome(genome_id="a", values={"gain": 1.0})

        policy.note_generation(_fitness(genome, 0.100))
        policy.note_generation(_fitness(genome, 0.105))

        assert policy.generations_without_gain == 1

    def test_target_reached_stops_the_search(self) -> None:
        """Step 83."""
        policy = self._policy(target_fitness=0.90)
        genome = CandidateGenome(genome_id="a", values={"gain": 1.0})

        stop = policy.check(BudgetLedger(), _fitness(genome, 0.91))

        assert stop is not None and stop[0] is StoppingReason.TARGET_REACHED

    def test_budget_outranks_patience(self) -> None:
        """Both fired; the binding constraint is the one to report."""
        policy = StoppingPolicy(
            SearchConfig(patience=1), SearchBudget(max_candidates=2)
        )
        genome = CandidateGenome(genome_id="a", values={"gain": 1.0})
        policy.note_generation(_fitness(genome, 0.5))
        policy.note_generation(_fitness(genome, 0.5))

        stop = policy.check(BudgetLedger().with_proposed(2), _fitness(genome, 0.5))

        assert stop is not None and stop[0] is StoppingReason.MAX_CANDIDATES

    def test_a_user_stop_outranks_everything(self) -> None:
        policy = self._policy()
        policy.request_stop()

        stop = policy.check(BudgetLedger(), None)

        assert stop is not None and stop[0] is StoppingReason.USER_STOPPED

    def test_nothing_triggered_means_continue(self) -> None:
        assert self._policy().check(BudgetLedger(), None) is None

    def test_warns_before_a_limit_is_reached(self) -> None:
        budget = SearchBudget(max_candidates=10)
        ledger = BudgetLedger().with_proposed(9)

        warnings = budget_warnings(ledger, budget)

        assert any("max_candidates" in item for item in warnings)
