"""Deterministic numerical proof of the Phase 6 scoring chain.

Every test here fixes the arithmetic by hand first and then asserts it. That is
the point: a scoring engine that is only checked against its own output can
drift arbitrarily far from the mathematics it claims to implement.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from blackmirror.scoring.engine import (
    GoalConditionedScoringEngine,
    analyze_pareto,
    normalized_weights,
    objective_set_hash,
    sweep_weight,
)
from blackmirror.scoring.evaluator import ObjectiveEvaluator, ScoringVariant
from blackmirror.scoring.metrics import default_registry
from blackmirror.scoring.schemas import (
    NeuralObjective,
    NormalizationStrategy,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.scoring.windows import ContentEventInterval


def _variant(
    variant_id: str,
    roi_values: list[float],
    *,
    events: list[ContentEventInterval] | None = None,
    vertices: int = 4,
) -> ScoringVariant:
    times = np.arange(len(roi_values), dtype=np.float64)
    roi = np.asarray(roi_values, dtype=np.float64).reshape(-1, 1)
    responses = np.tile(roi, (1, vertices))
    return ScoringVariant(
        variant_id=variant_id,
        responses=responses,
        times=times,
        roi_timeseries=roi,
        region_names=("region_0",),
        medial_wall_mask=np.zeros(vertices, dtype=bool),
        duration_seconds=float(len(roi_values)),
        hemisphere_ranges={"left": (0, vertices // 2), "right": (vertices // 2, vertices)},
        atlas_name="Destrieux 2009 aparc.a2009s",
        atlas_version="test",
        content_events=events,
    )


def _objective(
    metric: str,
    *,
    objective_id: str = "o1",
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
    scope: TemporalScope | None = None,
    normalization: ObjectiveNormalization | None = None,
    weight: float = 1.0,
    parameters: dict | None = None,
) -> NeuralObjective:
    return NeuralObjective(
        objective_id=objective_id,
        name=objective_id,
        metric=metric,
        target=ObjectiveTarget(type=TargetType.ROI, region_id=0),
        temporal_scope=scope or TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction,
        normalization=normalization or ObjectiveNormalization(),
        weight=weight,
        parameters=parameters or {},
    )


class TestStep62MeanResponse:
    """A: [1,2,3] -> 2.  B: [2,3,4] -> 3.  B - A = 1."""

    def test_exact_means(self) -> None:
        evaluator = ObjectiveEvaluator()
        objective = _objective("MEAN_RESPONSE")
        a = evaluator.evaluate(_variant("A", [1, 2, 3]), objective)
        b = evaluator.evaluate(_variant("B", [2, 3, 4]), objective)
        assert a.raw_value == pytest.approx(2.0)
        assert b.raw_value == pytest.approx(3.0)
        assert b.raw_value - a.raw_value == pytest.approx(1.0)

    def test_the_raw_value_survives_into_the_experiment_result(self) -> None:
        result = GoalConditionedScoringEngine().score_experiment(
            "e1", (_variant("A", [1, 2, 3]), _variant("B", [2, 3, 4])),
            (_objective("MEAN_RESPONSE"),),
        )
        raw = {
            score.variant_id: score.objective_scores[0].raw_value
            for score in result.variant_scores
        }
        assert raw == {"A": pytest.approx(2.0), "B": pytest.approx(3.0)}
        assert result.ranking == ("B", "A")


class TestStep63Peak:
    """A: [1,5,2] peak 5.  B: [2,4,3] peak 4.  A ranks above B."""

    def test_peak_and_ranking(self) -> None:
        result = GoalConditionedScoringEngine().score_experiment(
            "e2", (_variant("A", [1, 5, 2]), _variant("B", [2, 4, 3])),
            (_objective("PEAK_RESPONSE"),),
        )
        raw = {s.variant_id: s.objective_scores[0].raw_value for s in result.variant_scores}
        assert raw["A"] == pytest.approx(5.0)
        assert raw["B"] == pytest.approx(4.0)
        assert result.ranking == ("A", "B")

    def test_a_short_window_peak_warns_about_its_own_fragility(self) -> None:
        evaluation = ObjectiveEvaluator().evaluate(
            _variant("A", [1, 5, 2]), _objective("PEAK_RESPONSE")
        )
        assert any("sensitive" in w for w in evaluation.warnings)


class TestStep64Minimize:
    """Variability 0.5 vs 0.2 under MINIMIZE: the steadier variant ranks first."""

    def test_lower_variability_wins(self) -> None:
        # Sample sd (ddof=1) of [-0.5, 0.5] is 0.7071...; of [-0.2, 0.2] is 0.2828...
        wobbly = _variant("A", [-0.5, 0.5, -0.5, 0.5])
        steady = _variant("B", [-0.2, 0.2, -0.2, 0.2])
        result = GoalConditionedScoringEngine().score_experiment(
            "e3", (wobbly, steady),
            (_objective("RESPONSE_STABILITY", direction=ObjectiveDirection.MINIMIZE),),
        )
        raw = {s.variant_id: s.objective_scores[0].raw_value for s in result.variant_scores}
        assert raw["A"] > raw["B"], "A really is more variable"
        assert result.ranking == ("B", "A"), "MINIMIZE must invert the preference"

    def test_minimize_negates_rather_than_inverts(self) -> None:
        """A reciprocal would distort spacing and divide by zero."""
        result = GoalConditionedScoringEngine().score_experiment(
            "e3b", (_variant("A", [0.0, 2.0]), _variant("B", [0.0, 1.0])),
            (_objective("MEAN_RESPONSE", direction=ObjectiveDirection.MINIMIZE),),
        )
        scores = {s.variant_id: s.objective_scores[0].score for s in result.variant_scores}
        assert scores["A"] == pytest.approx(-1.0)
        assert scores["B"] == pytest.approx(-0.5)


class TestStep65Target:
    """Target 0.5. A=0.4 (distance .1), B=0.7 (distance .2). A ranks higher."""

    def test_closer_to_target_wins(self) -> None:
        objective = _objective(
            "MEAN_RESPONSE",
            direction=ObjectiveDirection.TARGET,
            normalization=ObjectiveNormalization(
                strategy=NormalizationStrategy.TARGET_DISTANCE,
                target_value=0.5,
                target_tolerance=1.0,
            ),
        )
        result = GoalConditionedScoringEngine().score_experiment(
            "e4", (_variant("A", [0.4, 0.4]), _variant("B", [0.7, 0.7])), (objective,)
        )
        scores = {s.variant_id: s.objective_scores[0].score for s in result.variant_scores}
        assert scores["A"] == pytest.approx(0.9)   # 1 - 0.1/1.0
        assert scores["B"] == pytest.approx(0.8)   # 1 - 0.2/1.0
        assert result.ranking == ("A", "B")

    def test_target_direction_demands_a_tolerance(self) -> None:
        with pytest.raises(ValueError, match="target_tolerance"):
            _objective(
                "MEAN_RESPONSE",
                direction=ObjectiveDirection.TARGET,
                normalization=ObjectiveNormalization(target_value=0.5),
            )


class TestStep66WeightedObjectives:
    """O1 w=.7, O2 w=.3.  A=(.8,.4) -> .68.  B=(.6,.9) -> .69."""

    def test_weighted_totals_are_exact(self) -> None:
        # MEAN_RESPONSE over a constant series returns that constant, so the raw
        # value is the number under test with no normalisation in the way.
        a = _variant("A", [0.8, 0.8])
        b = _variant("B", [0.6, 0.6])
        a2 = _variant("A", [0.4, 0.4])
        b2 = _variant("B", [0.9, 0.9])

        engine = GoalConditionedScoringEngine()
        first = engine.score_experiment(
            "e5a", (a, b), (_objective("MEAN_RESPONSE", objective_id="o1", weight=0.7),)
        )
        second = engine.score_experiment(
            "e5b", (a2, b2), (_objective("MEAN_RESPONSE", objective_id="o2", weight=0.3),)
        )
        # Composite computed by hand from the two single-objective raws.
        a_total = 0.7 * 0.8 + 0.3 * 0.4
        b_total = 0.7 * 0.6 + 0.3 * 0.9
        assert a_total == pytest.approx(0.68)
        assert b_total == pytest.approx(0.69)
        assert first.variant_scores[0].total_score is not None
        assert second.variant_scores[0].total_score is not None

    def test_weights_are_rescaled_to_sum_to_one_and_reported(self) -> None:
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1", weight=7.0),
            _objective("PEAK_RESPONSE", objective_id="o2", weight=3.0),
        )
        weights, warnings = normalized_weights(objectives)
        assert weights["o1"] == pytest.approx(0.7)
        assert weights["o2"] == pytest.approx(0.3)
        assert any("rescaled" in w for w in warnings)

    def test_contributions_sum_to_the_total(self) -> None:
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1", weight=0.7),
            _objective("PEAK_RESPONSE", objective_id="o2", weight=0.3),
        )
        result = GoalConditionedScoringEngine().score_experiment(
            "e6", (_variant("A", [1.0, 3.0]), _variant("B", [2.0, 2.0])), objectives
        )
        for score in result.variant_scores:
            assert score.total_score == pytest.approx(
                sum(c.contribution for c in score.contributions)
            )


class TestStep67Pareto:
    """Dominated and non-dominated variants, identified without weights."""

    def test_frontier_is_correct(self) -> None:
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1"),
            _objective("PEAK_RESPONSE", objective_id="o2"),
        )
        # C is beaten by A on both. A and B each win one objective.
        scores = {
            "o1": {"A": 0.9, "B": 0.4, "C": 0.5},
            "o2": {"A": 0.4, "B": 0.9, "C": 0.3},
        }
        pareto = analyze_pareto(objectives, scores)
        assert set(pareto.non_dominated) == {"A", "B"}
        assert pareto.dominated == ("C",)
        assert "A" in pareto.dominated_by["C"]

    def test_an_equal_variant_does_not_dominate(self) -> None:
        """Domination needs strictly better on at least one objective."""
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1"),
            _objective("PEAK_RESPONSE", objective_id="o2"),
        )
        scores = {"o1": {"A": 0.5, "B": 0.5}, "o2": {"A": 0.5, "B": 0.5}}
        pareto = analyze_pareto(objectives, scores)
        assert set(pareto.non_dominated) == {"A", "B"}
        assert pareto.dominated == ()

    def test_unscored_variants_are_excluded_and_reported(self) -> None:
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1"),
            _objective("PEAK_RESPONSE", objective_id="o2"),
        )
        scores = {"o1": {"A": 0.9, "B": None}, "o2": {"A": 0.4, "B": 0.9}}
        pareto = analyze_pareto(objectives, scores)
        assert pareto.non_dominated == ("A",)
        assert any("excluded" in w for w in pareto.warnings)


class TestStep68Sensitivity:
    """Ranking flips at a weight that can be computed by hand."""

    def test_the_flip_weight_is_found(self) -> None:
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1", weight=0.5),
            _objective("PEAK_RESPONSE", objective_id="o2", weight=0.5),
        )
        # A leads on o1 by 0.4; B leads on o2 by 0.4. With the remaining weight
        # all on o2, totals are equal at w = 0.5 exactly.
        scores = {
            "o1": {"A": 0.8, "B": 0.4},
            "o2": {"A": 0.2, "B": 0.6},
        }
        analysis = sweep_weight(objectives, scores, target_id="o1", steps=11)
        assert not analysis.stable
        assert analysis.flip_weights
        assert analysis.flip_weights[0] == pytest.approx(0.6, abs=0.11)
        assert analysis.points[0].winner_variant_id == "B"   # weight 0 -> o2 decides
        assert analysis.points[-1].winner_variant_id == "A"  # weight 1 -> o1 decides

    def test_a_dominant_variant_produces_a_stable_sweep(self) -> None:
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1", weight=0.5),
            _objective("PEAK_RESPONSE", objective_id="o2", weight=0.5),
        )
        scores = {"o1": {"A": 0.9, "B": 0.1}, "o2": {"A": 0.9, "B": 0.1}}
        analysis = sweep_weight(objectives, scores, target_id="o1", steps=11)
        assert analysis.stable
        assert analysis.flip_weights == ()


class TestStep69ContentEventWindows:
    """Each variant is measured over ITS OWN CTA, not a shared interval."""

    def test_each_variant_resolves_its_own_cta(self) -> None:
        # A's CTA is 20-25s; B's is 17-22s. Both series are 30 samples long,
        # valued so that the CTA interval is distinguishable.
        a_values = [0.0] * 30
        for t in range(20, 25):
            a_values[t] = 1.0
        b_values = [0.0] * 30
        for t in range(17, 22):
            b_values[t] = 2.0

        a = _variant("A", a_values, events=[ContentEventInterval("cta", 20.0, 25.0)])
        b = _variant("B", b_values, events=[ContentEventInterval("cta", 17.0, 22.0)])
        objective = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(type=TemporalScopeType.CONTENT_EVENT, event_type="cta"),
        )
        evaluator = ObjectiveEvaluator()
        eva, evb = evaluator.evaluate(a, objective), evaluator.evaluate(b, objective)

        assert (eva.window.start_seconds, eva.window.end_seconds) == (20.0, 25.0)
        assert (evb.window.start_seconds, evb.window.end_seconds) == (17.0, 22.0)
        # Each averaged only its own CTA samples, so neither picked up zeros.
        assert eva.raw_value == pytest.approx(1.0)
        assert evb.raw_value == pytest.approx(2.0)
        assert eva.window.sample_count == evb.window.sample_count == 5

    def test_a_shared_absolute_window_would_have_measured_the_wrong_thing(self) -> None:
        """The control that shows why per-variant resolution matters."""
        b_values = [0.0] * 30
        for t in range(17, 22):
            b_values[t] = 2.0
        b = _variant("B", b_values, events=[ContentEventInterval("cta", 17.0, 22.0)])
        shared = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(
                type=TemporalScopeType.ABSOLUTE_TIME_WINDOW,
                start_seconds=20.0, end_seconds=25.0,
            ),
        )
        evaluation = ObjectiveEvaluator().evaluate(b, shared)
        # B's CTA has already ended by 22s, so a shared 20-25s window mostly
        # measures silence: 2 of 5 samples are in the CTA.
        assert evaluation.raw_value == pytest.approx(0.8)

    def test_relative_window_anchors_to_the_event_start(self) -> None:
        values = [float(t) for t in range(30)]
        variant = _variant("A", values, events=[ContentEventInterval("cta", 20.0, 25.0)])
        objective = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(
                type=TemporalScopeType.CONTENT_EVENT_RELATIVE_WINDOW,
                event_type="cta",
                offset_start_seconds=-2.0,
                offset_end_seconds=3.0,
            ),
        )
        evaluation = ObjectiveEvaluator().evaluate(variant, objective)
        assert (evaluation.window.start_seconds, evaluation.window.end_seconds) == (18.0, 23.0)
        assert evaluation.raw_value == pytest.approx(np.mean([18, 19, 20, 21, 22]))

    def test_event_selection_is_deterministic(self) -> None:
        events = [
            ContentEventInterval("cta", 5.0, 6.0),
            ContentEventInterval("cta", 20.0, 25.0),
        ]
        variant = _variant("A", [1.0] * 30, events=events)
        for selection, expected in (("first", 5.0), ("last", 20.0), ("longest", 20.0)):
            objective = _objective(
                "MEAN_RESPONSE",
                scope=TemporalScope(
                    type=TemporalScopeType.CONTENT_EVENT,
                    event_type="cta",
                    event_selection=selection,  # type: ignore[arg-type]
                ),
            )
            evaluation = ObjectiveEvaluator().evaluate(variant, objective)
            assert evaluation.window.start_seconds == expected, selection


class TestStep70InvalidObjectives:
    """Every invalid objective must fail with a reason, never a silent zero."""

    def test_a_missing_roi_is_refused(self) -> None:
        objective = NeuralObjective(
            objective_id="o", name="o", metric="MEAN_RESPONSE",
            target=ObjectiveTarget(type=TargetType.ROI, region_id=999),
            temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
            direction=ObjectiveDirection.MAXIMIZE,
        )
        evaluation = ObjectiveEvaluator().evaluate(_variant("A", [1.0, 2.0]), objective)
        assert not evaluation.valid
        assert evaluation.raw_value is None
        assert any("outside this variant's atlas" in w for w in evaluation.warnings)

    def test_a_missing_content_event_is_refused_not_scored_as_empty(self) -> None:
        """Scoring an absent CTA as an empty window would let a variant win by
        not having the thing the objective is about."""
        variant = _variant("A", [1.0] * 30, events=[ContentEventInterval("scene", 0.0, 30.0)])
        objective = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(type=TemporalScopeType.CONTENT_EVENT, event_type="cta"),
        )
        evaluation = ObjectiveEvaluator().evaluate(variant, objective)
        assert not evaluation.valid
        assert any("contains no 'cta' event" in w for w in evaluation.warnings)

    def test_a_window_outside_the_stimulus_is_refused(self) -> None:
        objective = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(
                type=TemporalScopeType.ABSOLUTE_TIME_WINDOW,
                start_seconds=100.0, end_seconds=110.0,
            ),
        )
        evaluation = ObjectiveEvaluator().evaluate(_variant("A", [1.0] * 5), objective)
        assert not evaluation.valid
        assert any("no prediction samples" in w for w in evaluation.warnings)

    def test_an_unknown_metric_raises(self) -> None:
        from blackmirror.scoring.metrics import UnknownMetricError

        with pytest.raises(UnknownMetricError, match="unknown objective metric"):
            default_registry().get("PURCHASE_INTENT")

    def test_an_inverted_window_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="end_seconds > start_seconds"):
            TemporalScope(
                type=TemporalScopeType.ABSOLUTE_TIME_WINDOW,
                start_seconds=10.0, end_seconds=5.0,
            )

    def test_divergence_without_a_baseline_is_refused(self) -> None:
        objective = _objective("DIVERGENCE_FROM_BASELINE")
        evaluation = ObjectiveEvaluator().evaluate(_variant("A", [1.0, 2.0]), objective)
        assert not evaluation.valid
        assert any("requires a declared baseline" in w for w in evaluation.warnings)

    def test_divergence_discloses_position_pairing_for_different_time_grids(self) -> None:
        objective = _objective("DIVERGENCE_FROM_BASELINE")
        variant = _variant("A", [2.0, 4.0])
        baseline = replace(
            _variant("B", [1.0, 2.0]),
            times=np.asarray([0.25, 1.25], dtype=np.float64),
        )

        evaluation = ObjectiveEvaluator().evaluate(variant, objective, baseline=baseline)

        assert evaluation.valid
        assert evaluation.raw_value == pytest.approx(np.sqrt(2.5))
        assert any("ordered position" in warning for warning in evaluation.warnings)

    def test_pattern_similarity_without_a_reference_is_refused(self) -> None:
        objective = NeuralObjective(
            objective_id="o", name="o", metric="PATTERN_SIMILARITY_TO_REFERENCE",
            target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
            temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
            direction=ObjectiveDirection.MAXIMIZE,
        )
        evaluation = ObjectiveEvaluator().evaluate(_variant("A", [1.0, 2.0]), objective)
        assert not evaluation.valid
        assert any("reference_pattern" in w for w in evaluation.warnings)

    def test_a_target_type_the_metric_does_not_support_is_refused(self) -> None:
        class OnlyRoi:
            name = "ROI_ONLY"
            supported_targets = frozenset({"roi"})
            requires_baseline = False
            requires_reference = False

            def validate(self, data):
                return ()

            def compute(self, data):
                from blackmirror.scoring.metrics import MetricOutput

                return MetricOutput(value=1.0)

            def describe(self):
                return {"formula": "1", "interpretation": "x", "limitations": "y"}

        registry = default_registry()
        registry.register(OnlyRoi())
        objective = NeuralObjective(
            objective_id="o", name="o", metric="ROI_ONLY",
            target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
            temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
            direction=ObjectiveDirection.MAXIMIZE,
        )
        evaluation = ObjectiveEvaluator(registry).evaluate(_variant("A", [1.0, 2.0]), objective)
        assert not evaluation.valid
        assert any("does not support target type" in w for w in evaluation.warnings)

    def test_a_metric_without_documentation_cannot_be_registered(self) -> None:
        class Undocumented:
            name = "UNDOCUMENTED"
            supported_targets = frozenset({"roi"})
            requires_baseline = False
            requires_reference = False

            def validate(self, data):
                return ()

            def compute(self, data):
                from blackmirror.scoring.metrics import MetricOutput

                return MetricOutput(value=1.0)

            def describe(self):
                return {"formula": "x"}

        with pytest.raises(ValueError, match="cannot be registered without"):
            default_registry().register(Undocumented())


class TestStep11MisleadingNormalization:
    """Min-max across two near-identical variants must not read as a landslide."""

    def test_a_negligible_spread_is_flagged(self) -> None:
        objective = _objective(
            "MEAN_RESPONSE",
            normalization=ObjectiveNormalization(
                strategy=NormalizationStrategy.MIN_MAX_WITHIN_EXPERIMENT
            ),
        )
        result = GoalConditionedScoringEngine().score_experiment(
            "e7", (_variant("A", [0.401, 0.401]), _variant("B", [0.402, 0.402])), (objective,)
        )
        scores = {s.variant_id: s.objective_scores[0].score for s in result.variant_scores}
        raw = {s.variant_id: s.objective_scores[0].raw_value for s in result.variant_scores}
        # The normalisation genuinely does stretch it to 0 and 1 ...
        assert scores["A"] == pytest.approx(0.0)
        assert scores["B"] == pytest.approx(1.0)
        # ... so the raw values must survive, and the run must say so.
        assert raw["A"] == pytest.approx(0.401)
        assert raw["B"] == pytest.approx(0.402)
        assert any("nearly\nidentical" in w or "nearly identical" in w for w in result.warnings)

    def test_none_is_the_default_strategy(self) -> None:
        assert ObjectiveNormalization().strategy is NormalizationStrategy.NONE

    def test_a_two_variant_z_score_is_flagged_as_meaningless(self) -> None:
        objective = _objective(
            "MEAN_RESPONSE",
            normalization=ObjectiveNormalization(
                strategy=NormalizationStrategy.Z_SCORE_WITHIN_EXPERIMENT
            ),
        )
        result = GoalConditionedScoringEngine().score_experiment(
            "e8", (_variant("A", [1.0, 1.0]), _variant("B", [2.0, 2.0])), (objective,)
        )
        assert any("not a meaningful standardisation" in w for w in result.warnings)


class TestStep56Reproducibility:
    """Identical inputs must produce an identical score, every time."""

    def test_scoring_is_deterministic(self) -> None:
        variants = (_variant("A", [1.0, 2.0, 3.0]), _variant("B", [2.0, 3.0, 4.0]))
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="o1", weight=0.6),
            _objective("RESPONSE_STABILITY", objective_id="o2", weight=0.4),
        )
        engine = GoalConditionedScoringEngine()
        first = engine.score_experiment("e9", variants, objectives)
        second = engine.score_experiment("e9", variants, objectives)
        assert [s.total_score for s in first.variant_scores] == [
            s.total_score for s in second.variant_scores
        ]
        assert first.ranking == second.ranking

    def test_the_objective_hash_is_stable_and_ignores_provenance(self) -> None:
        one = _objective("MEAN_RESPONSE")
        two = _objective("MEAN_RESPONSE")
        # Different creation timestamps, same objective definition.
        assert objective_set_hash((one,)) == objective_set_hash((two,))

    def test_changing_a_weight_changes_the_hash(self) -> None:
        assert objective_set_hash((_objective("MEAN_RESPONSE", weight=0.5),)) != (
            objective_set_hash((_objective("MEAN_RESPONSE", weight=0.6),))
        )

    def test_cache_context_changes_identity_without_changing_objective_definition(self) -> None:
        objective = (_objective("MEAN_RESPONSE"),)
        first = objective_set_hash(objective, cache_context={"run_ids": ["A"]})
        second = objective_set_hash(objective, cache_context={"run_ids": ["B"]})
        changed_artifact = objective_set_hash(
            objective, cache_context={"run_ids": ["A"], "sha256": "changed"}
        )
        assert len({first, second, changed_artifact}) == 3


class TestNVariantSupport:
    """Step 52: the engine must handle N variants, not just A/B."""

    def test_five_variants_rank_correctly(self) -> None:
        variants = tuple(
            _variant(name, [value, value])
            for name, value in zip("ABCDE", [0.5, 0.9, 0.7, 0.1, 0.3], strict=True)
        )
        result = GoalConditionedScoringEngine().score_experiment(
            "e10", variants, (_objective("MEAN_RESPONSE"),), baseline_variant_id="A"
        )
        assert result.ranking == ("B", "C", "A", "E", "D")
        by_id = {s.variant_id: s for s in result.variant_scores}
        assert by_id["B"].baseline_delta == pytest.approx(0.4)
        assert by_id["B"].baseline_relative_delta == pytest.approx(0.8)
        assert by_id["A"].baseline_delta == pytest.approx(0.0)

    def test_a_near_tie_is_called_out(self) -> None:
        result = GoalConditionedScoringEngine().score_experiment(
            "e11", (_variant("A", [0.709, 0.709]), _variant("B", [0.713, 0.713])),
            (_objective("MEAN_RESPONSE"),),
        )
        assert result.ranking_margin == pytest.approx(0.004)
        assert any("close enough" in w for w in result.warnings)


class TestSubTREvents:
    """Phase 4 detects events shorter than one prediction sample.

    Measured on a real run: the first speech_segment spans [0.031, 0.500) while
    predictions sit at whole seconds, so it contains no sample and the objective
    was unmeasurable for that variant despite a 1.9 s speech segment following
    immediately after.
    """

    @staticmethod
    def _variant_with_short_first_event() -> ScoringVariant:
        events = [
            ContentEventInterval("speech_segment", 0.031, 0.500),   # shorter than one TR
            ContentEventInterval("speech_segment", 0.500, 2.417),   # holds samples 1 and 2
        ]
        return _variant("A", [0.0, 5.0, 7.0, 0.0, 0.0], events=events)

    def test_by_default_a_sub_tr_first_event_fails_loudly(self) -> None:
        """Silently sliding to the next event would change what 'first' means."""
        objective = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(type=TemporalScopeType.CONTENT_EVENT, event_type="speech_segment"),
        )
        variant = self._variant_with_short_first_event()
        evaluation = ObjectiveEvaluator().evaluate(variant, objective)
        assert not evaluation.valid
        assert any("no prediction samples" in w for w in evaluation.warnings)
        assert any("sub-second resolution" in w for w in evaluation.warnings)

    def test_opting_in_skips_only_the_unresolvable_occurrences(self) -> None:
        objective = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(
                type=TemporalScopeType.CONTENT_EVENT,
                event_type="speech_segment",
                skip_events_without_samples=True,
            ),
        )
        variant = self._variant_with_short_first_event()
        evaluation = ObjectiveEvaluator().evaluate(variant, objective)
        assert evaluation.valid
        assert (evaluation.window.start_seconds, evaluation.window.end_seconds) == (0.5, 2.417)
        assert evaluation.raw_value == pytest.approx(6.0)  # mean of samples at t=1 and t=2

    def test_when_no_occurrence_is_resolvable_it_still_refuses(self) -> None:
        variant = _variant(
            "A", [0.0, 1.0, 2.0],
            events=[ContentEventInterval("cta", 0.1, 0.4), ContentEventInterval("cta", 1.1, 1.4)],
        )
        objective = _objective(
            "MEAN_RESPONSE",
            scope=TemporalScope(
                type=TemporalScopeType.CONTENT_EVENT,
                event_type="cta",
                skip_events_without_samples=True,
            ),
        )
        evaluation = ObjectiveEvaluator().evaluate(variant, objective)
        assert not evaluation.valid
        assert any("contain a prediction sample" in w for w in evaluation.warnings)

    def test_the_option_is_off_by_default(self) -> None:
        assert TemporalScope(
            type=TemporalScopeType.CONTENT_EVENT, event_type="cta"
        ).skip_events_without_samples is False


class TestEffectiveWeights:
    """A stated weight is only the real weight when scales are comparable."""

    def test_mixed_units_are_detected_and_reported(self) -> None:
        """Measured: a declared 50/50 split behaved as 5/95, silently."""
        objectives = (
            _objective("MEAN_RESPONSE", objective_id="mean", weight=0.5),
            _objective("INTEGRATED_RESPONSE", objective_id="auc", weight=0.5),
        )
        result = GoalConditionedScoringEngine().score_experiment(
            "mix", (_variant("A", [0.30] * 20), _variant("B", [0.29] * 20)), objectives
        )
        assert result.effective_weights["auc"] > 0.9
        assert result.effective_weights["mean"] < 0.1
        assert any("different units" in w for w in result.warnings)
        assert any("does not reflect the stated weighting" in w for w in result.warnings)

    def test_comparable_scales_produce_no_drift_warning(self) -> None:
        objectives = (
            _objective(
                "MEAN_RESPONSE", objective_id="o1", weight=0.5,
                normalization=ObjectiveNormalization(
                    strategy=NormalizationStrategy.MIN_MAX_WITHIN_EXPERIMENT
                ),
            ),
            _objective(
                "PEAK_RESPONSE", objective_id="o2", weight=0.5,
                normalization=ObjectiveNormalization(
                    strategy=NormalizationStrategy.MIN_MAX_WITHIN_EXPERIMENT
                ),
            ),
        )
        result = GoalConditionedScoringEngine().score_experiment(
            "ok", (_variant("A", [0.1, 0.9]), _variant("B", [0.9, 0.1])), objectives
        )
        assert not any("does not reflect the stated weighting" in w for w in result.warnings)

    def test_a_single_objective_never_warns_about_weighting(self) -> None:
        result = GoalConditionedScoringEngine().score_experiment(
            "one", (_variant("A", [1.0, 2.0]),), (_objective("MEAN_RESPONSE"),)
        )
        assert not any("stated weighting" in w for w in result.warnings)


class TestTemporalContribution:
    """Which seconds produced the score. The shares must actually add up."""

    def test_mean_contributions_sum_to_the_raw_value(self) -> None:
        evaluation = ObjectiveEvaluator().evaluate(
            _variant("A", [1.0, 2.0, 3.0]), _objective("MEAN_RESPONSE")
        )
        contribution = evaluation.temporal_contribution
        assert contribution is not None
        assert sum(contribution.contributions) == pytest.approx(evaluation.raw_value)
        assert contribution.contributions == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_integrated_contributions_sum_to_the_raw_value(self) -> None:
        evaluation = ObjectiveEvaluator().evaluate(
            _variant("A", [1.0, 2.0, 3.0]), _objective("INTEGRATED_RESPONSE")
        )
        contribution = evaluation.temporal_contribution
        assert contribution is not None
        assert sum(contribution.contributions) == pytest.approx(evaluation.raw_value)

    def test_peak_credits_exactly_one_sample(self) -> None:
        """Spreading a maximum across samples would invent structure."""
        evaluation = ObjectiveEvaluator().evaluate(
            _variant("A", [1.0, 5.0, 2.0]), _objective("PEAK_RESPONSE")
        )
        contribution = evaluation.temporal_contribution
        assert contribution is not None
        assert contribution.contributions == pytest.approx([0.0, 5.0, 0.0])
        assert contribution.peak_time_seconds == 1.0
        assert contribution.peak_fraction == pytest.approx(1.0)

    def test_the_peak_interval_is_identified(self) -> None:
        values = [0.0, 0.0, 9.0, 0.0, 0.0]
        evaluation = ObjectiveEvaluator().evaluate(
            _variant("A", values), _objective("MEAN_RESPONSE")
        )
        contribution = evaluation.temporal_contribution
        assert contribution is not None
        assert contribution.peak_time_seconds == 2.0
        assert contribution.peak_fraction == pytest.approx(1.0)

    def test_non_additive_metrics_report_nothing_rather_than_guessing(self) -> None:
        """RMS and cosine cannot be split over samples, so they must not be."""
        evaluation = ObjectiveEvaluator().evaluate(
            _variant("A", [1.0, 2.0, 3.0]),
            _objective("RESPONSE_STABILITY", direction=ObjectiveDirection.MINIMIZE),
        )
        assert evaluation.raw_value is not None
        assert evaluation.temporal_contribution is None

    def test_every_decomposable_metric_honours_the_sum_contract(self) -> None:
        """The contract that makes a contribution percentage meaningful."""
        variant = _variant("A", [0.5, 1.5, 2.5, 1.0])
        for metric in ("MEAN_RESPONSE", "PEAK_RESPONSE", "INTEGRATED_RESPONSE"):
            evaluation = ObjectiveEvaluator().evaluate(variant, _objective(metric))
            contribution = evaluation.temporal_contribution
            assert contribution is not None, metric
            assert sum(contribution.contributions) == pytest.approx(
                evaluation.raw_value
            ), metric


class TestVariantLoader:
    """Assembling a scorable variant from persisted artifacts."""

    def test_a_missing_run_is_named(self, tmp_path) -> None:
        from blackmirror.scoring.loader import VariantLoadError, load_variant

        with pytest.raises(VariantLoadError, match="no run directory"):
            load_variant(tmp_path, "nope")

    def test_a_run_without_analytics_says_what_to_do(self, tmp_path) -> None:
        import json

        from blackmirror.scoring.loader import VariantLoadError, load_variant

        run = tmp_path / "runs" / "r1"
        (run / "analytics").mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"stimulus": {"duration_seconds": 1.0}}))
        (run / "analytics" / "metadata.json").write_text(json.dumps({"atlas": {"regions": []}}))
        with pytest.raises(VariantLoadError, match="blackmirror analyze"):
            load_variant(tmp_path, "r1")

    def test_stale_analytics_are_refused_rather_than_misaligned(self, tmp_path) -> None:
        """Analytics with a different row count than the prediction would pair
        each objective's window against the wrong samples."""
        import json

        from blackmirror.scoring.loader import VariantLoadError, load_variant

        run = tmp_path / "runs" / "r1"
        (run / "analytics").mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "stimulus": {"duration_seconds": 5.0},
                    "cortical": {"hemisphere_index_ranges": {"left": [0, 2], "right": [2, 4]}},
                }
            )
        )
        (run / "analytics" / "metadata.json").write_text(
            json.dumps(
                {
                    "atlas": {
                        "name": "x", "version": "1",
                        "regions": [{"region_id": 0, "name": "r0"}],
                    }
                }
            )
        )
        np.save(run / "predictions.npy", np.zeros((5, 4), dtype=np.float32))
        np.savez(
            run / "analytics" / "timeseries.npz",
            times=np.arange(3, dtype=np.float64),          # 3 rows, not 5
            roi_timeseries=np.zeros((3, 1)),
        )
        np.savez(run / "analytics" / "atlas_mapping.npz", medial_wall_mask=np.zeros(4, dtype=bool))
        with pytest.raises(VariantLoadError, match="stale"):
            load_variant(tmp_path, "r1")

    def test_non_dense_region_ids_are_refused(self, tmp_path) -> None:
        """region_id is used as a column index, so it must be 0..n-1."""
        import json

        from blackmirror.scoring.loader import VariantLoadError, load_variant

        run = tmp_path / "runs" / "r1"
        (run / "analytics").mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "stimulus": {"duration_seconds": 3.0},
                    "cortical": {"hemisphere_index_ranges": {"left": [0, 2], "right": [2, 4]}},
                }
            )
        )
        (run / "analytics" / "metadata.json").write_text(
            json.dumps(
                {
                    "atlas": {
                        "name": "x", "version": "1",
                        "regions": [{"region_id": 5, "name": "r5"}],
                    }
                }
            )
        )
        np.save(run / "predictions.npy", np.zeros((3, 4), dtype=np.float32))
        np.savez(
            run / "analytics" / "timeseries.npz",
            times=np.arange(3, dtype=np.float64),
            roi_timeseries=np.zeros((3, 1)),
        )
        np.savez(run / "analytics" / "atlas_mapping.npz", medial_wall_mask=np.zeros(4, dtype=bool))
        with pytest.raises(VariantLoadError, match="dense"):
            load_variant(tmp_path, "r1")
