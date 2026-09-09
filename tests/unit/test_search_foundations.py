"""Steps 1-7: experiment, config, budget, hard admission, space, parameters.

The budget tests carry the most weight. An evaluation costs roughly 52 minutes
of inference on this machine, so a budget that leaks by three candidates leaks
by most of a working afternoon, and the failure is invisible until the machine
has already spent it.
"""

from __future__ import annotations

from random import Random

import pytest

from blackmirror.materialization.schemas import EditOperation
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.search.budget import (
    BudgetLedger,
    BudgetLimit,
    SearchBudget,
    admit,
)
from blackmirror.search.schemas import (
    MEASURED_NUISANCE_FLOOR,
    SearchConfig,
    SearchExperiment,
    SearchStatus,
    SearchStrategyName,
    StoppingReason,
)
from blackmirror.search.space import (
    ContentSearchSpace,
    ParameterType,
    SearchParameter,
    SearchSpaceError,
    categorical,
    continuous,
)


def _objective(objective_id: str = "o1") -> NeuralObjective:
    return NeuralObjective(
        objective_id=objective_id,
        name=objective_id,
        metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE,
    )


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", EditOperation.AUDIO_GAIN_DB, -6.0, 6.0),
            continuous("brightness", EditOperation.BRIGHTNESS, -0.2, 0.2),
        )
    )


def _experiment(**overrides: object) -> SearchExperiment:
    fields: dict[str, object] = {
        "search_id": "s1",
        "experiment_id": "exp",
        "root_run_id": "root",
        "root_media_path": "/media/root.mp4",
        "root_media_sha256": "a" * 64,
        "objective_set_hash": "b" * 64,
        "objective_definition_hash": "c" * 64,
        "objectives": (_objective(),),
        "space": _space(),
    }
    fields.update(overrides)
    return SearchExperiment(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step 7 - parameters
# ---------------------------------------------------------------------------


class TestSearchParameter:
    def test_rejects_a_type_nothing_can_materialize(self) -> None:
        """A knob that cannot become a file would never actually move."""
        with pytest.raises(ValueError, match="cannot be materialized"):
            SearchParameter(
                name="cta_time",
                type=ParameterType.TEXT_CANDIDATE,
                operation=EditOperation.BRIGHTNESS,
                choices=(1.0, 2.0),
            )

    def test_rejects_bounds_the_edit_operation_cannot_accept(self) -> None:
        """Caught here, not at materialization time hours later."""
        with pytest.raises(ValueError, match="outside the range"):
            continuous("gain", EditOperation.AUDIO_GAIN_DB, -6.0, 999.0)

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="low < high"):
            continuous("gain", EditOperation.AUDIO_GAIN_DB, 6.0, -6.0)

    def test_sampling_stays_inside_the_domain(self) -> None:
        parameter = continuous("gain", EditOperation.AUDIO_GAIN_DB, -6.0, 6.0)
        rng = Random(0)

        values = [parameter.sample(rng) for _ in range(200)]

        assert all(parameter.contains(value) for value in values)
        assert min(values) < -4 and max(values) > 4

    def test_sampling_is_reproducible_from_a_seed(self) -> None:
        parameter = continuous("gain", EditOperation.AUDIO_GAIN_DB, -6.0, 6.0)

        first = [parameter.sample(Random(7)) for _ in range(5)]
        second = [parameter.sample(Random(7)) for _ in range(5)]

        assert first == second

    def test_clamping_pulls_a_value_back_inside(self) -> None:
        """Strategies overshoot; a bound the user set is not negotiable."""
        parameter = continuous("gain", EditOperation.AUDIO_GAIN_DB, -6.0, 6.0)

        assert parameter.clamp(99.0) == 6.0
        assert parameter.clamp(-99.0) == -6.0

    def test_quantization_collapses_indistinguishable_values(self) -> None:
        """21.600 and 21.6001 are one candidate, not two evaluations."""
        parameter = continuous(
            "gain", EditOperation.AUDIO_GAIN_DB, -6.0, 6.0, resolution=0.01
        )

        assert parameter.quantize(3.14159) == parameter.quantize(3.1416)

    def test_categorical_clamps_to_the_nearest_choice(self) -> None:
        parameter = categorical("speed", EditOperation.SPEED, (0.9, 1.0, 1.1))

        assert parameter.clamp(1.06) == 1.1
        assert parameter.contains(1.05) is False

    def test_boolean_needs_exactly_two_choices(self) -> None:
        with pytest.raises(ValueError, match="exactly two choices"):
            SearchParameter(
                name="loud",
                type=ParameterType.BOOLEAN,
                operation=EditOperation.AUDIO_GAIN_DB,
                choices=(0.0, 3.0, 6.0),
            )


# ---------------------------------------------------------------------------
# Step 6 - the space
# ---------------------------------------------------------------------------


class TestContentSearchSpace:
    def test_rejects_two_parameters_driving_one_operation(self) -> None:
        """They would fight at materialization and ordering would decide."""
        with pytest.raises(ValueError, match="same edit operation"):
            ContentSearchSpace(
                parameters=(
                    continuous("gain_a", EditOperation.AUDIO_GAIN_DB, -3.0, 3.0),
                    continuous("gain_b", EditOperation.AUDIO_GAIN_DB, -6.0, 6.0),
                )
            )

    def test_sampling_produces_a_complete_point(self) -> None:
        space = _space()

        values = space.sample(Random(1))

        assert set(values) == {"gain", "brightness"}
        assert space.contains(values)

    def test_converts_a_point_into_edit_steps(self) -> None:
        space = _space()

        steps = space.to_edit_steps({"gain": 3.0, "brightness": 0.1})

        assert {step.operation for step in steps} == {
            EditOperation.AUDIO_GAIN_DB,
            EditOperation.BRIGHTNESS,
        }

    def test_refuses_to_decode_an_illegal_value(self) -> None:
        """Clamping here would make the recorded genome disagree with the file."""
        space = _space()

        with pytest.raises(SearchSpaceError, match="outside its declared domain"):
            space.to_edit_steps({"gain": 99.0, "brightness": 0.0})

    def test_refuses_to_decode_an_incomplete_point(self) -> None:
        with pytest.raises(SearchSpaceError, match="missing values"):
            _space().to_edit_steps({"gain": 1.0})

    def test_reports_grid_size_before_enumerating_it(self) -> None:
        """Step 16 needs this to refuse a grid larger than the budget."""
        space = _space()

        assert space.cardinality(steps=4) == 16.0
        assert len(space.grid(steps=4)) == 16

    def test_detects_a_point_that_would_change_nothing(self) -> None:
        space = _space()

        assert space.is_neutral({"gain": 0.0, "brightness": 0.0}) is True
        assert space.is_neutral({"gain": 3.0, "brightness": 0.0}) is False


# ---------------------------------------------------------------------------
# Steps 3-5 - budget, ledger, hard admission
# ---------------------------------------------------------------------------


class TestBudget:
    def test_requires_at_least_one_finite_limit(self) -> None:
        """Step 74. An unbounded search over hour-long evaluations is unsafe."""
        with pytest.raises(ValueError, match="at least one finite limit"):
            SearchBudget(
                max_candidates=None,
                max_generations=None,
                max_tribe_runs=None,
                max_wall_seconds=None,
                max_cost=None,
            )

    def test_cost_limit_without_a_rate_is_refused(self) -> None:
        """Nothing would ever accrue against it, so it is a false ceiling."""
        with pytest.raises(ValueError, match="cost_per_wall_hour"):
            SearchBudget(max_cost=10.0, cost_per_wall_hour=0.0)

    def test_remaining_reports_none_for_an_unlimited_dimension(self) -> None:
        budget = SearchBudget(max_candidates=5, max_generations=None)

        remaining = BudgetLedger().remaining(budget)

        assert remaining[BudgetLimit.MAX_CANDIDATES] == 5
        assert remaining[BudgetLimit.MAX_GENERATIONS] is None


class TestLedger:
    def test_a_cache_hit_is_an_evaluation_but_not_a_tribe_run(self) -> None:
        """Conflating them would flatter the compute figures in the report."""
        ledger = BudgetLedger().with_evaluation(wall_seconds=1.0, cached=True)

        assert ledger.candidates_evaluated == 1
        assert ledger.tribe_runs == 0
        assert ledger.cache_hits == 1

    def test_a_real_evaluation_counts_as_a_tribe_run(self) -> None:
        ledger = BudgetLedger().with_evaluation(wall_seconds=3120.0)

        assert ledger.tribe_runs == 1
        assert ledger.wall_seconds == 3120.0

    def test_failures_are_counted_without_stopping_the_ledger(self) -> None:
        ledger = BudgetLedger().with_evaluation(wall_seconds=5.0, failed=True)

        assert ledger.evaluation_failures == 1
        assert ledger.candidates_evaluated == 1

    def test_cache_hit_rate_is_none_before_anything_ran(self) -> None:
        """Zero would read as "the cache never helps", which is a claim."""
        assert BudgetLedger().cache_hit_rate is None

    def test_spending_returns_a_new_ledger(self) -> None:
        """Immutability is what makes a mid-crash checkpoint trustworthy."""
        original = BudgetLedger()

        spent = original.with_proposed(3)

        assert original.candidates_generated == 0
        assert spent.candidates_generated == 3

    def test_estimated_cost_uses_the_operator_rate(self) -> None:
        budget = SearchBudget(max_cost=100.0, cost_per_wall_hour=2.0)
        ledger = BudgetLedger().with_evaluation(wall_seconds=3600.0)

        assert ledger.estimated_cost(budget) == pytest.approx(2.0)


class TestHardAdmission:
    def test_admits_only_what_remains(self) -> None:
        """Step 5's worked example: 2 left, 5 requested, 2 admitted."""
        budget = SearchBudget(max_candidates=10, max_tribe_runs=2)
        ledger = BudgetLedger()

        decision = admit(5, ledger, budget)

        assert decision.admitted == 2
        assert decision.rejected == 3
        assert decision.binding_limit is BudgetLimit.MAX_TRIBE_RUNS

    def test_admits_nothing_once_exhausted(self) -> None:
        budget = SearchBudget(max_candidates=2)
        ledger = BudgetLedger().with_proposed(2)

        decision = admit(3, ledger, budget)

        assert decision.admitted == 0
        assert decision.is_exhausted is True
        assert "exhausted" in decision.reason

    def test_admits_the_whole_batch_when_affordable(self) -> None:
        decision = admit(4, BudgetLedger(), SearchBudget(max_candidates=10))

        assert decision.admitted == 4
        assert decision.binding_limit is None

    def test_never_admits_more_than_requested(self) -> None:
        decision = admit(2, BudgetLedger(), SearchBudget(max_candidates=1000))

        assert decision.admitted == 2

    def test_a_budget_cannot_be_overspent_across_repeated_batches(self) -> None:
        """The property that matters: total admitted never exceeds the ceiling."""
        budget = SearchBudget(max_candidates=7)
        ledger = BudgetLedger()
        total = 0

        for _ in range(10):
            decision = admit(3, ledger, budget)
            total += decision.admitted
            ledger = ledger.with_proposed(decision.admitted)

        assert total == 7


# ---------------------------------------------------------------------------
# Steps 1-2 - experiment and config
# ---------------------------------------------------------------------------


class TestSearchConfig:
    def test_default_minimum_improvement_is_the_measured_noise_floor(self) -> None:
        """Below this, an "improvement" is not distinguishable from a nuisance.

        Measured on this corpus: swapping between two mirrors of the same
        language-model weights moved whole-cortex mean by 0.0157 on average.
        """
        assert SearchConfig().minimum_improvement == MEASURED_NUISANCE_FLOOR
        assert SearchConfig().resolves_below_noise is False

    def test_a_threshold_below_the_floor_is_allowed_but_warned_about(self) -> None:
        config = SearchConfig(minimum_improvement=0.001)

        assert config.resolves_below_noise is True
        assert any("nuisance floor" in warning for warning in config.warnings())

    def test_warns_when_concurrency_would_contend_for_memory(self) -> None:
        config = SearchConfig(max_concurrent_candidates=4, candidate_batch_size=4)

        assert any("memory-bound" in warning for warning in config.warnings())

    def test_warns_when_exploration_is_switched_off(self) -> None:
        config = SearchConfig(exploration_rate=0.0, minimum_exploration_rate=0.0)

        assert any("local optimum" in warning for warning in config.warnings())

    def test_exploration_decays_but_never_below_its_floor(self) -> None:
        config = SearchConfig(
            exploration_rate=0.4, exploration_decay=0.5, minimum_exploration_rate=0.1
        )

        assert config.exploration_rate_at(0) == pytest.approx(0.4)
        assert config.exploration_rate_at(1) == pytest.approx(0.2)
        assert config.exploration_rate_at(10) == pytest.approx(0.1)

    def test_idle_worker_configuration_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sit idle"):
            SearchConfig(candidate_batch_size=1, max_concurrent_candidates=2)


class TestSearchExperiment:
    def test_several_objectives_need_a_declared_primary(self) -> None:
        """Otherwise the search would silently pick which one it maximises."""
        with pytest.raises(ValueError, match="without a primary"):
            _experiment(objectives=(_objective("o1"), _objective("o2")))

    def test_primary_objective_must_be_one_of_the_declared(self) -> None:
        with pytest.raises(ValueError, match="not among the declared"):
            _experiment(
                objectives=(_objective("o1"), _objective("o2")),
                primary_objective_id="o3",
            )

    def test_a_completed_search_must_say_why_it_stopped(self) -> None:
        with pytest.raises(ValueError, match="why it stopped"):
            _experiment(status=SearchStatus.COMPLETED)

    def test_defaults_to_random_search(self) -> None:
        """Step 14: the baseline every smarter strategy is measured against."""
        assert _experiment().config.strategy is SearchStrategyName.RANDOM_SEARCH

    def test_changing_config_preserves_the_previous_one(self) -> None:
        """Step 73. A past decision must not look like it followed new rules."""
        experiment = _experiment()

        updated = experiment.with_config(
            SearchConfig(exploration_rate=0.1), reason="narrowing"
        )

        assert updated.config.exploration_rate == 0.1
        assert updated.config.config_version == 2
        assert len(updated.config_history) == 1
        assert updated.config_history[0].config.exploration_rate == 0.3
        assert updated.config_history[0].reason == "narrowing"

    def test_warns_when_the_budget_outruns_the_space(self) -> None:
        small = ContentSearchSpace(
            parameters=(categorical("speed", EditOperation.SPEED, (0.9, 1.1)),)
        )

        experiment = _experiment(space=small, budget=SearchBudget(max_candidates=50))

        assert any("exhaust the space" in warning for warning in experiment.warnings())

    def test_carries_a_notice_that_rules_out_the_wrong_reading(self) -> None:
        notice = _experiment().interpretation_notice

        assert "not a global optimum" in notice
        assert "not a statement about human viewers" in notice

    def test_records_a_stopping_reason_when_completed(self) -> None:
        experiment = _experiment(
            status=SearchStatus.COMPLETED, stopping_reason=StoppingReason.MAX_CANDIDATES
        )

        assert experiment.stopping_reason is StoppingReason.MAX_CANDIDATES
