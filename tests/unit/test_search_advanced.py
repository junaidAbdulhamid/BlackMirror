"""Steps 22, 32-38, 51-53, 58-62, 72, 81, 85, 86.

Guardrails and Pareto carry the weight here. A guardrail that fails to veto lets
a search report an inadmissible candidate as its answer, and a frontier computed
wrongly lets a dominated candidate look like a trade-off worth considering.
"""

from __future__ import annotations

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
from blackmirror.search.cache import (
    CacheIdentity,
    CachingEvaluator,
    SearchEvaluationCache,
)
from blackmirror.search.fitness import (
    CandidateFitness,
    ComputeCost,
    FitnessStatus,
    best_infeasible,
    best_of,
)
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.guardrails import (
    Guardrail,
    GuardrailKind,
    GuardrailPolicy,
)
from blackmirror.search.pareto import (
    dominates,
    non_dominated_sort,
    pareto_front,
    select_by_pareto_rank,
)
from blackmirror.search.population import (
    EliminationReason,
    build_population,
    diversity_aware_selection,
    population_diversity,
    successive_halving,
)
from blackmirror.search.scheduler import (
    CandidateScheduler,
    HumanOverride,
    PriorityReason,
)
from blackmirror.search.space import ContentSearchSpace, continuous


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),
            continuous("brightness", Op.BRIGHTNESS, -1.0, 1.0, resolution=0.01),
        )
    )


def _objective(oid: str, direction: ObjectiveDirection) -> NeuralObjective:
    return NeuralObjective(
        objective_id=oid, name=oid, metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction,
    )


def _genome(cid: str, gain: float = 1.0, brightness: float = 0.0) -> CandidateGenome:
    return CandidateGenome(
        genome_id=cid, values={"gain": gain, "brightness": brightness}
    )


def _fit(
    cid: str,
    scalar: float,
    *,
    raw: dict[str, float | None] | None = None,
    gain: float = 1.0,
    brightness: float = 0.0,
    duration_change: float = 0.0,
) -> CandidateFitness:
    return CandidateFitness(
        candidate_id=cid,
        genome=_genome(cid, gain, brightness),
        status=FitnessStatus.EVALUATED,
        scalar_fitness=scalar,
        primary_objective_id="o1",
        raw_objectives=raw if raw is not None else {"o1": scalar},
        duration_change_seconds=duration_change,
    )


# ---------------------------------------------------------------------------
# Steps 61-62, 86 - guardrails
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_a_floor_breach_is_a_violation(self) -> None:
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.OBJECTIVE_FLOOR,
                       objective_id="stability", bound=0.5),)
        )

        result = policy.check({"stability": 0.4})

        assert result.feasible is False
        assert "below the floor" in result.violations[0].message

    def test_a_ceiling_breach_is_a_violation(self) -> None:
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.OBJECTIVE_CEILING,
                       objective_id="noise", bound=0.2),)
        )

        assert policy.check({"noise": 0.3}).feasible is False
        assert policy.check({"noise": 0.1}).feasible is True

    def test_a_relative_drop_is_measured_against_the_root(self) -> None:
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.MAX_RELATIVE_DROP,
                       objective_id="stability", bound=0.10),)
            , root_objectives={"stability": 1.0},
        )

        assert policy.check({"stability": 0.95}).feasible is True
        assert policy.check({"stability": 0.80}).feasible is False

    def test_a_duration_guardrail_uses_the_produced_length(self) -> None:
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.MAX_DURATION_CHANGE,
                       bound=0.5),)
        )

        assert policy.check({}, duration_change=0.4).feasible is True
        assert policy.check({}, duration_change=-1.0).feasible is False

    def test_a_missing_measurement_is_unchecked_not_passed(self) -> None:
        """Silently passing would let an unmeasured guardrail protect nothing."""
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.OBJECTIVE_FLOOR,
                       objective_id="stability", bound=0.5),)
        )

        result = policy.check({"other": 1.0})

        assert result.unchecked == ("g1",)
        assert result.feasible is True
        assert "unchecked" in result.summary

    def test_unchecked_can_be_made_disqualifying(self) -> None:
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.OBJECTIVE_FLOOR,
                       objective_id="stability", bound=0.5),),
            treat_unchecked_as_violation=True,
        )

        assert policy.check({}).feasible is False

    def test_a_guardrail_needs_the_objective_it_watches(self) -> None:
        with pytest.raises(ValueError, match="needs the objective"):
            Guardrail(guardrail_id="g", kind=GuardrailKind.OBJECTIVE_FLOOR, bound=0.5)

    def test_guardrail_ids_must_be_unique(self) -> None:
        rail = Guardrail(
            guardrail_id="g", kind=GuardrailKind.MAX_DURATION_CHANGE, bound=1.0
        )
        with pytest.raises(ValueError, match="unique"):
            GuardrailPolicy((rail, rail))


class TestGuardrailVeto:
    def test_the_highest_scorer_cannot_win_if_it_breached_a_guardrail(self) -> None:
        """Step 86, the property the whole mechanism exists for."""
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.OBJECTIVE_FLOOR,
                       objective_id="stability", bound=0.5),)
        )
        top = _fit("top", 0.9, raw={"o1": 0.9, "stability": 0.1})
        modest = _fit("modest", 0.4, raw={"o1": 0.4, "stability": 0.8})

        judged = [
            item.model_copy(update={"feasibility": policy.check(item.raw_objectives)})
            for item in (top, modest)
        ]
        winner = best_of(judged)

        assert winner is not None and winner.candidate_id == "modest"
        assert judged[0].is_usable is True
        assert judged[0].is_eligible is False

    def test_the_disqualified_leader_is_still_reportable(self) -> None:
        """A user needs told that the top score broke a rule, not shown nothing."""
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.OBJECTIVE_FLOOR,
                       objective_id="stability", bound=0.5),)
        )
        judged = [
            item.model_copy(update={"feasibility": policy.check(item.raw_objectives)})
            for item in (
                _fit("top", 0.9, raw={"o1": 0.9, "stability": 0.1}),
                _fit("modest", 0.4, raw={"o1": 0.4, "stability": 0.8}),
            )
        ]

        blocked = best_infeasible(judged)

        assert blocked is not None and blocked.candidate_id == "top"


# ---------------------------------------------------------------------------
# Steps 58-60, 85 - Pareto
# ---------------------------------------------------------------------------


class TestPareto:
    def _objectives(self) -> tuple[NeuralObjective, ...]:
        return (
            _objective("o1", ObjectiveDirection.MAXIMIZE),
            _objective("o2", ObjectiveDirection.MAXIMIZE),
        )

    def test_dominance_requires_at_least_as_good_everywhere(self) -> None:
        assert dominates((2.0, 2.0), (1.0, 1.0)) is True
        assert dominates((2.0, 1.0), (1.0, 1.0)) is True
        assert dominates((2.0, 0.5), (1.0, 1.0)) is False
        assert dominates((1.0, 1.0), (1.0, 1.0)) is False

    def test_the_frontier_matches_a_known_answer(self) -> None:
        """Step 85: a synthetic set whose non-dominated members are known."""
        results = [
            _fit("a", 1.0, raw={"o1": 3.0, "o2": 1.0}),
            _fit("b", 1.0, raw={"o1": 1.0, "o2": 3.0}),
            _fit("c", 1.0, raw={"o1": 2.0, "o2": 2.0}),
            _fit("d", 1.0, raw={"o1": 1.0, "o2": 1.0}),
        ]

        front = pareto_front(results, self._objectives())

        assert set(front.non_dominated) == {"a", "b", "c"}
        assert front.dominated == ("d",)
        assert set(front.dominated_by["d"]) >= {"a", "b", "c"}

    def test_a_minimize_objective_is_folded_correctly(self) -> None:
        """Lower is better there, so the frontier must not invert."""
        objectives = (
            _objective("o1", ObjectiveDirection.MAXIMIZE),
            _objective("o2", ObjectiveDirection.MINIMIZE),
        )
        results = [
            _fit("a", 1.0, raw={"o1": 3.0, "o2": 1.0}),
            _fit("b", 1.0, raw={"o1": 3.0, "o2": 5.0}),
        ]

        front = pareto_front(results, objectives)

        assert front.non_dominated == ("a",)
        assert front.dominated == ("b",)

    def test_infeasible_candidates_are_excluded_from_the_frontier(self) -> None:
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g1", kind=GuardrailKind.OBJECTIVE_FLOOR,
                       objective_id="o2", bound=1.0),)
        )
        results = [
            _fit("a", 1.0, raw={"o1": 5.0, "o2": 0.0}),
            _fit("b", 1.0, raw={"o1": 1.0, "o2": 3.0}),
        ]
        judged = [
            item.model_copy(update={"feasibility": policy.check(item.raw_objectives)})
            for item in results
        ]

        front = pareto_front(judged, self._objectives())

        assert front.non_dominated == ("b",)
        assert front.excluded == ("a",)
        assert front.warnings

    def test_a_missing_objective_excludes_rather_than_guesses(self) -> None:
        results = [
            _fit("a", 1.0, raw={"o1": 3.0, "o2": None}),
            _fit("b", 1.0, raw={"o1": 1.0, "o2": 3.0}),
        ]

        front = pareto_front(results, self._objectives())

        assert front.excluded == ("a",)

    def test_non_dominated_sort_produces_successive_fronts(self) -> None:
        results = [
            _fit("a", 1.0, raw={"o1": 3.0, "o2": 1.0}),
            _fit("b", 1.0, raw={"o1": 1.0, "o2": 3.0}),
            _fit("c", 1.0, raw={"o1": 2.0, "o2": 0.5}),
        ]

        fronts = non_dominated_sort(results, self._objectives())

        assert set(fronts[0]) == {"a", "b"}
        assert fronts[1] == ("c",)

    def test_selection_takes_whole_fronts_in_order(self) -> None:
        results = [
            _fit("a", 1.0, raw={"o1": 3.0, "o2": 1.0}),
            _fit("b", 1.0, raw={"o1": 1.0, "o2": 3.0}),
            _fit("c", 1.0, raw={"o1": 2.0, "o2": 0.5}),
        ]

        chosen = select_by_pareto_rank(results, self._objectives(), count=2)

        assert {item.candidate_id for item in chosen} == {"a", "b"}


# ---------------------------------------------------------------------------
# Steps 36-38, 81 - the evaluation cache
# ---------------------------------------------------------------------------


def _identity() -> CacheIdentity:
    return CacheIdentity(
        root_media_sha256="a" * 64,
        objective_definition_hash="b" * 64,
        space_version="1.0",
        materialization_version="1.0",
        scoring_version="1.0",
        model_fingerprint="tribe_v2|test",
    )


class CountingEvaluator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
        self.calls.append(genome.genome_id)
        return CandidateFitness(
            candidate_id=genome.genome_id, genome=genome,
            status=FitnessStatus.EVALUATED, scalar_fitness=genome.values["gain"],
            primary_objective_id="o1", raw_objectives={"o1": genome.values["gain"]},
            compute=ComputeCost(wall_seconds=3120.0, tribe_runs=1),
        )


class TestEvaluationCache:
    def test_an_identical_genome_is_evaluated_once(self, tmp_path: Path) -> None:
        """Step 81: one evaluation, one reuse."""
        space = _space()
        inner = CountingEvaluator()
        caching = CachingEvaluator(
            inner, SearchEvaluationCache(tmp_path, _identity()), space
        )

        first = caching.evaluate(_genome("c1", 3.0))
        second = caching.evaluate(_genome("c2", 3.0))

        assert inner.calls == ["c1"]
        assert second.compute.served_from_cache is True
        assert second.compute.tribe_runs == 0
        assert second.scalar_fitness == first.scalar_fitness

    def test_a_different_genome_is_not_a_hit(self, tmp_path: Path) -> None:
        inner = CountingEvaluator()
        caching = CachingEvaluator(
            inner, SearchEvaluationCache(tmp_path, _identity()), _space()
        )

        caching.evaluate(_genome("c1", 3.0))
        caching.evaluate(_genome("c2", 4.0))

        assert inner.calls == ["c1", "c2"]

    def test_the_cache_survives_a_new_process(self, tmp_path: Path) -> None:
        """The point of persisting it: a second search pays nothing."""
        space = _space()
        first = CountingEvaluator()
        CachingEvaluator(
            first, SearchEvaluationCache(tmp_path, _identity()), space
        ).evaluate(_genome("c1", 3.0))

        second = CountingEvaluator()
        result = CachingEvaluator(
            second, SearchEvaluationCache(tmp_path, _identity()), space
        ).evaluate(_genome("other", 3.0))

        assert second.calls == []
        assert result.compute.served_from_cache is True

    def test_a_changed_pipeline_version_invalidates_the_entry(
        self, tmp_path: Path
    ) -> None:
        """A number computed under different rules must not be reused."""
        space = _space()
        CachingEvaluator(
            CountingEvaluator(), SearchEvaluationCache(tmp_path, _identity()), space
        ).evaluate(_genome("c1", 3.0))

        newer = _identity().model_copy(update={"scoring_version": "2.0"})
        inner = CountingEvaluator()
        CachingEvaluator(
            inner, SearchEvaluationCache(tmp_path, newer), space
        ).evaluate(_genome("c1", 3.0))

        assert inner.calls == ["c1"]

    def test_failures_are_not_cached(self, tmp_path: Path) -> None:
        """A transient failure cached forever would be a permanent one."""
        class Failing:
            def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
                return CandidateFitness(
                    candidate_id=genome.genome_id, genome=genome,
                    status=FitnessStatus.EVALUATION_FAILED, reason="disk full",
                )

        cache = SearchEvaluationCache(tmp_path, _identity())
        CachingEvaluator(Failing(), cache, _space()).evaluate(_genome("c1", 3.0))

        assert cache.size() == 0


# ---------------------------------------------------------------------------
# Steps 22, 32-35 - populations
# ---------------------------------------------------------------------------


class TestPopulation:
    def test_records_why_each_candidate_was_eliminated(self) -> None:
        members = [_fit(f"c{i}", float(i), gain=float(i)) for i in range(5)]

        population = build_population(0, members, _space(), keep=2)

        assert population.survivors == ("c4", "c3")
        assert len(population.eliminated) == 3
        assert population.eliminated[0].reason is EliminationReason.OUTSIDE_BEAM
        assert "of 5" in population.eliminated[0].detail

    def test_a_guardrail_failure_is_recorded_as_its_own_reason(self) -> None:
        policy = GuardrailPolicy(
            (Guardrail(guardrail_id="g", kind=GuardrailKind.OBJECTIVE_FLOOR,
                       objective_id="o1", bound=1.0),)
        )
        members = [
            _fit("good", 5.0, raw={"o1": 5.0}, gain=5.0),
            _fit("bad", 9.0, raw={"o1": 0.0}, gain=9.0),
        ]
        judged = [
            item.model_copy(update={"feasibility": policy.check(item.raw_objectives)})
            for item in members
        ]

        population = build_population(0, judged, _space(), keep=1)

        assert population.survivors == ("good",)
        reasons = {item.reason for item in population.eliminated}
        assert EliminationReason.GUARDRAIL in reasons

    def test_diversity_is_none_below_two_members(self) -> None:
        """Zero would read as "fully collapsed", a claim about nothing."""
        assert population_diversity([_fit("a", 1.0)], _space()) is None

    def test_diversity_falls_as_a_population_converges(self) -> None:
        spread = [
            _fit("a", 1.0, gain=-10.0, brightness=-1.0),
            _fit("b", 1.0, gain=10.0, brightness=1.0),
        ]
        tight = [
            _fit("a", 1.0, gain=1.0, brightness=0.0),
            _fit("b", 1.0, gain=1.1, brightness=0.0),
        ]

        assert population_diversity(spread, _space()) > population_diversity(
            tight, _space()
        )

    def test_collapse_is_detected(self) -> None:
        members = [
            _fit("a", 1.0, gain=1.0, brightness=0.0),
            _fit("b", 1.0, gain=1.01, brightness=0.0),
        ]

        population = build_population(0, members, _space(), keep=2)

        assert population.has_collapsed is True

    def test_successive_halving_keeps_the_better_half(self) -> None:
        """Step 22. Elimination on full evaluations, not fake partial ones."""
        members = [_fit(f"c{i}", float(i), gain=float(i)) for i in range(8)]

        population = successive_halving(members, _space(), generation=1)

        assert len(population.survivors) == 4
        assert set(population.survivors) == {"c7", "c6", "c5", "c4"}
        assert population.eliminated[0].reason is EliminationReason.HALVED

    def test_halving_never_empties_the_population(self) -> None:
        population = successive_halving(
            [_fit("only", 1.0)], _space(), generation=0
        )

        assert len(population.survivors) == 1

    def test_diversity_selection_defaults_to_pure_fitness(self) -> None:
        members = [_fit(f"c{i}", float(i), gain=float(i)) for i in range(4)]

        chosen = diversity_aware_selection(members, _space(), count=2)

        assert [item.candidate_id for item in chosen] == ["c3", "c2"]

    def test_novelty_weight_spreads_the_selection(self) -> None:
        """The best is always kept; the second is chosen for distance."""
        members = [
            _fit("best", 10.0, gain=1.0),
            _fit("near", 9.9, gain=1.1),
            _fit("far", 9.0, gain=-9.0),
        ]

        chosen = diversity_aware_selection(
            members, _space(), count=2, novelty_weight=0.9
        )

        assert [item.candidate_id for item in chosen] == ["best", "far"]

    def test_novelty_weight_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="fraction between 0 and 1"):
            diversity_aware_selection([], _space(), count=1, novelty_weight=2.0)


# ---------------------------------------------------------------------------
# Steps 51-53, 72 - scheduling and override
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_pinned_candidates_run_first(self) -> None:
        override = HumanOverride()
        override.pin("c3")
        scheduler = CandidateScheduler(override=override)

        queued = scheduler.prioritize([_genome("c1"), _genome("c2"), _genome("c3")])

        assert queued[0].genome.genome_id == "c3"
        assert PriorityReason.PINNED in queued[0].reasons
        assert "pinned by a person" in queued[0].describe()

    def test_cached_candidates_run_before_expensive_ones(self) -> None:
        """Free information first: it can raise the bar for everything else."""
        scheduler = CandidateScheduler(
            is_cached=lambda genome: genome.genome_id == "c2"
        )

        queued = scheduler.prioritize([_genome("c1"), _genome("c2")])

        assert queued[0].genome.genome_id == "c2"

    def test_an_eliminated_candidate_never_runs(self) -> None:
        override = HumanOverride()
        override.eliminate("c2")
        scheduler = CandidateScheduler(override=override)

        queued = scheduler.prioritize([_genome("c1"), _genome("c2")])

        assert [item.genome.genome_id for item in queued] == ["c1"]
        assert scheduler.skipped == [("c2", "eliminated by a person")]

    def test_pinning_clears_a_prior_elimination(self) -> None:
        override = HumanOverride()
        override.eliminate("c1")
        override.pin("c1")

        assert override.is_eliminated("c1") is False

    def test_every_position_carries_its_reason(self) -> None:
        """Step 53: no opaque priority score deciding hour-long jobs."""
        queued = CandidateScheduler().prioritize([_genome("c1"), _genome("c2")])

        for item in queued:
            assert item.reasons
            assert item.describe()

    def test_a_stop_prevents_remaining_candidates(self) -> None:
        scheduler = CandidateScheduler()
        queued = scheduler.prioritize([_genome("c1"), _genome("c2")])
        seen: list[str] = []

        def evaluate(genome: CandidateGenome) -> CandidateFitness:
            seen.append(genome.genome_id)
            return _fit(genome.genome_id, 1.0)

        scheduler.run(queued, evaluate, should_continue=lambda: not seen)

        assert len(seen) == 1

    def test_a_failure_is_retried_then_returned(self) -> None:
        scheduler = CandidateScheduler(max_attempts=3)
        queued = scheduler.prioritize([_genome("c1")])
        attempts: list[int] = []

        def evaluate(genome: CandidateGenome) -> CandidateFitness:
            attempts.append(1)
            return CandidateFitness(
                candidate_id=genome.genome_id, genome=genome,
                status=FitnessStatus.EVALUATION_FAILED, reason="flaky",
            )

        results = scheduler.run(queued, evaluate)

        assert len(attempts) == 3
        assert results[0].status is FitnessStatus.EVALUATION_FAILED

    def test_a_rejection_is_not_retried(self) -> None:
        """Rejections are deterministic; retrying spends budget to relearn."""
        scheduler = CandidateScheduler(max_attempts=3)
        queued = scheduler.prioritize([_genome("c1")])
        attempts: list[int] = []

        def evaluate(genome: CandidateGenome) -> CandidateFitness:
            attempts.append(1)
            return CandidateFitness(
                candidate_id=genome.genome_id, genome=genome,
                status=FitnessStatus.REJECTED, reason="frozen modality",
            )

        scheduler.run(queued, evaluate)

        assert len(attempts) == 1

    def test_concurrent_running_preserves_priority_order(self) -> None:
        """A worker count change must not change the recorded order."""
        scheduler = CandidateScheduler(max_concurrent=3)
        genomes = [_genome(f"c{i}", float(i)) for i in range(1, 7)]
        queued = scheduler.prioritize(genomes)

        results = scheduler.run(queued, lambda g: _fit(g.genome_id, 1.0))

        assert [item.candidate_id for item in results] == [
            item.genome.genome_id for item in queued
        ]

    def test_concurrency_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            CandidateScheduler(max_concurrent=0)


class TestFidelity:
    """Step 23. The interface exists; the approximations deliberately do not."""

    def test_only_the_full_pass_is_implemented(self) -> None:
        from blackmirror.search.fidelity import FidelityLevel

        assert FidelityLevel.FULL.is_implemented is True
        assert FidelityLevel.LOW.is_implemented is False
        assert FidelityLevel.MEDIUM.is_implemented is False

    def test_using_an_unimplemented_level_raises_with_the_reason(self) -> None:
        """A fabricated cheap score would make every elimination untrustworthy."""
        from blackmirror.search.fidelity import FidelityLevel, require_implemented

        with pytest.raises(NotImplementedError, match="unvalidated number"):
            require_implemented(FidelityLevel.LOW)

    def test_the_full_level_passes_the_guard(self) -> None:
        from blackmirror.search.fidelity import FidelityLevel, require_implemented

        assert require_implemented(FidelityLevel.FULL) is FidelityLevel.FULL
