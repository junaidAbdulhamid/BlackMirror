"""Deterministic proof of the Phase 7 evidence chain.

Recommendation *wording* is never asserted. Every test checks structured
fields, because prose is the part most likely to change and least likely to
matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from blackmirror.optimization.engine import (
    OptimizationOrchestrator,
    content_series_from_arrays,
)
from blackmirror.optimization.evidence import ContentSeries, build_feature_evidence
from blackmirror.optimization.intervals import (
    analyse_gap,
    detect_intervals,
    select_references,
)
from blackmirror.optimization.schemas import (
    ContentIntervention,
    EditCost,
    EditInstruction,
    EvidenceConfidence,
    ExpectedDirection,
    IntervalKind,
    InterventionType,
    OptimizationConstraints,
    OptimizationRecommendation,
    OptimizationRequest,
    OptimizationStrategy,
    ReferenceStrategy,
    RiskLevel,
)
from blackmirror.optimization.validate import (
    deduplicate,
    rank,
    scan_language,
    validate,
)
from blackmirror.scoring.engine import GoalConditionedScoringEngine
from blackmirror.scoring.evaluator import ScoringVariant
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)

FEATURES = ("motion", "speech_present", "audio_energy", "visual_available")


def _variant(vid: str, values: list[float]) -> ScoringVariant:
    times = np.arange(len(values), dtype=np.float64)
    roi = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return ScoringVariant(
        variant_id=vid, responses=np.tile(roi, (1, 4)), times=times, roi_timeseries=roi,
        region_names=("r0",), medial_wall_mask=np.zeros(4, dtype=bool),
        duration_seconds=float(len(values)),
        hemisphere_ranges={"left": (0, 2), "right": (2, 4)},
    )


def _objective(
    oid: str = "o1",
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
    metric: str = "MEAN_RESPONSE",
    normalization: ObjectiveNormalization | None = None,
) -> NeuralObjective:
    return NeuralObjective(
        objective_id=oid, name=oid, metric=metric,
        target=ObjectiveTarget(type=TargetType.ROI, region_id=0),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction, normalization=normalization or ObjectiveNormalization(),
    )


def _series(vid: str, motion: list[float], speech: list[float]) -> ContentSeries:
    times = np.arange(len(motion), dtype=np.float64)
    matrix = np.column_stack(
        [motion, speech, [0.5] * len(motion), [1.0] * len(motion)]
    ).astype(np.float64)
    return content_series_from_arrays(vid, times, matrix, FEATURES)


class TestStep66EvidenceGrounding:
    """Source 0.30 vs reference 0.50, weak interval, motion 0.2 vs 0.5."""

    def test_evidence_names_the_feature_interval_and_delta(self) -> None:
        from blackmirror.optimization.schemas import OptimizationInterval

        interval = OptimizationInterval(
            interval_id="weak-o1-0", kind=IntervalKind.WEAK, objective_id="o1",
            start_seconds=20.0, end_seconds=25.0, sample_count=5,
            contribution_share=0.02, contribution=0.02, reason="low share",
        )
        source = _series("A", [0.2] * 30, [0.0] * 30)
        reference = _series("B", [0.5] * 30, [1.0] * 30)
        evidence, considered = build_feature_evidence(
            interval, source, [reference], prefix="W0"
        )
        motion = next(e for e in evidence if e.feature == "motion")
        assert motion.source_value == pytest.approx(0.2)
        assert motion.reference_value == pytest.approx(0.5)
        assert motion.delta == pytest.approx(0.3)
        assert motion.interval_start_seconds == 20.0
        assert motion.interval_end_seconds == 25.0
        assert motion.association == "temporal_comparison"
        assert considered == 3, "availability columns must not be compared"

    def test_a_difference_below_threshold_is_not_evidence(self) -> None:
        """Scanning 21 features will always turn up something tiny."""
        from blackmirror.optimization.schemas import OptimizationInterval

        interval = OptimizationInterval(
            interval_id="weak-o1-0", kind=IntervalKind.WEAK, objective_id="o1",
            start_seconds=0.0, end_seconds=5.0, sample_count=5,
            contribution_share=0.02, contribution=0.02, reason="low share",
        )
        evidence, _ = build_feature_evidence(
            interval, _series("A", [0.50] * 10, [0.0] * 10),
            [_series("B", [0.51] * 10, [0.0] * 10)], prefix="W0",
        )
        assert not [e for e in evidence if e.feature == "motion"]

    def test_consistency_counts_agreeing_references(self) -> None:
        from blackmirror.optimization.schemas import OptimizationInterval

        interval = OptimizationInterval(
            interval_id="weak-o1-0", kind=IntervalKind.WEAK, objective_id="o1",
            start_seconds=0.0, end_seconds=5.0, sample_count=5,
            contribution_share=0.02, contribution=0.02, reason="low share",
        )
        source = _series("A", [0.2] * 10, [0.0] * 10)
        agree = [_series("B", [0.6] * 10, [0.0] * 10), _series("C", [0.7] * 10, [0.0] * 10)]
        evidence, _ = build_feature_evidence(interval, source, agree, prefix="W0")
        assert next(e for e in evidence if e.feature == "motion").consistency == 1.0

        mixed = [_series("B", [0.9] * 10, [0.0] * 10), _series("C", [0.1] * 10, [0.0] * 10)]
        evidence, _ = build_feature_evidence(interval, source, mixed, prefix="W0")
        motion = [e for e in evidence if e.feature == "motion"]
        if motion:
            assert motion[0].consistency == 0.5


class TestGapAnalysis:
    """The formula must follow the direction, not be assumed."""

    def test_maximize_gap_is_reference_minus_source(self) -> None:
        engine = GoalConditionedScoringEngine()
        result = engine.score_experiment(
            "e", (_variant("A", [0.3] * 5), _variant("B", [0.5] * 5)), (_objective(),)
        )
        source = next(v for v in result.variant_scores if v.variant_id == "A")
        reference = next(v for v in result.variant_scores if v.variant_id == "B")
        gap = analyse_gap(_objective(), source, reference)
        assert gap.gap == pytest.approx(0.2)
        assert gap.formula == "reference - source"

    def test_minimize_gap_reverses(self) -> None:
        objective = _objective(direction=ObjectiveDirection.MINIMIZE)
        engine = GoalConditionedScoringEngine()
        result = engine.score_experiment(
            "e", (_variant("A", [0.5] * 5), _variant("B", [0.3] * 5)), (objective,)
        )
        source = next(v for v in result.variant_scores if v.variant_id == "A")
        reference = next(v for v in result.variant_scores if v.variant_id == "B")
        gap = analyse_gap(objective, source, reference)
        assert gap.gap == pytest.approx(0.2), "the source is 0.2 too high"
        assert gap.formula == "source - reference"

    def test_target_gap_is_distance_from_the_target(self) -> None:
        objective = _objective(
            direction=ObjectiveDirection.TARGET,
            normalization=ObjectiveNormalization(
                strategy="target_distance", target_value=0.5, target_tolerance=1.0
            ),
        )
        engine = GoalConditionedScoringEngine()
        result = engine.score_experiment("e", (_variant("A", [0.4] * 5),), (objective,))
        source = result.variant_scores[0]
        gap = analyse_gap(objective, source, None)
        assert gap.gap == pytest.approx(0.1)
        assert gap.formula == "abs(source - target)"
        assert gap.reference_variant_id is None

    def test_a_source_already_ahead_reports_a_note(self) -> None:
        engine = GoalConditionedScoringEngine()
        result = engine.score_experiment(
            "e", (_variant("A", [0.9] * 5), _variant("B", [0.2] * 5)), (_objective(),)
        )
        source = next(v for v in result.variant_scores if v.variant_id == "A")
        reference = next(v for v in result.variant_scores if v.variant_id == "B")
        gap = analyse_gap(_objective(), source, reference)
        assert gap.gap < 0
        assert "already matches or exceeds" in (gap.note or "")


class TestIntervalDetection:
    """Weakness is objective-relative, never a raw low value."""

    def test_a_flat_interval_contributes_less_than_a_peak(self) -> None:
        values = [0.0] * 10 + [1.0] * 10
        engine = GoalConditionedScoringEngine()
        result = engine.score_experiment("e", (_variant("A", values),), (_objective(),))
        intervals, _ = detect_intervals(_objective(), result.variant_scores[0])
        weak = [i for i in intervals if i.kind is IntervalKind.WEAK]
        assert weak, "the zero-contribution first half must be weak"
        assert weak[0].start_seconds == 0.0
        assert weak[0].end_seconds <= 10.0

    def test_strong_intervals_are_detected_for_preservation(self) -> None:
        values = [0.0] * 18 + [5.0, 5.0]
        engine = GoalConditionedScoringEngine()
        result = engine.score_experiment("e", (_variant("A", values),), (_objective(),))
        intervals, _ = detect_intervals(_objective(), result.variant_scores[0])
        strong = [i for i in intervals if i.kind is IntervalKind.STRONG]
        assert strong
        assert strong[0].start_seconds >= 18.0

    def test_a_metric_without_decomposition_refuses(self) -> None:
        """Stability is an RMS-like quantity; it cannot be split over samples."""
        objective = _objective(metric="RESPONSE_STABILITY", direction=ObjectiveDirection.MINIMIZE)
        engine = GoalConditionedScoringEngine()
        result = engine.score_experiment("e", (_variant("A", [1.0, 2.0, 3.0]),), (objective,))
        intervals, warnings = detect_intervals(objective, result.variant_scores[0])
        assert intervals == []
        assert any("no temporal decomposition" in w for w in warnings)


class TestReferenceSelection:
    """A reference is never chosen silently."""

    @staticmethod
    def _result():
        engine = GoalConditionedScoringEngine()
        return engine.score_experiment(
            "e",
            (_variant("A", [0.3] * 5), _variant("B", [0.5] * 5), _variant("C", [0.4] * 5)),
            (_objective(),),
            baseline_variant_id="A",
        )

    def test_best_score_returns_only_higher_scoring_variants(self) -> None:
        ids, _ = select_references(self._result(), "A", "best_score")
        assert ids == ["B", "C"], "ordered best first, and the source is excluded"

    def test_no_better_variant_is_reported_not_hidden(self) -> None:
        ids, warnings = select_references(self._result(), "B", "best_score")
        assert ids == []
        assert any("no variant scores higher" in w for w in warnings)

    def test_manual_selection_drops_unscored_ids_with_a_warning(self) -> None:
        ids, warnings = select_references(
            self._result(), "A", "manual", manual_ids=("B", "ghost")
        )
        assert ids == ["B"]
        assert any("ghost" in w for w in warnings)

    def test_baseline_strategy_requires_a_baseline(self) -> None:
        ids, warnings = select_references(self._result(), "A", "baseline", baseline_id=None)
        assert ids == []
        assert any("no scored baseline" in w for w in warnings)


def _recommendation(
    rid: str = "R1",
    *,
    intervention_type: InterventionType = InterventionType.PACE_EDIT,
    start: float = 20.0,
    end: float = 25.0,
    confidence: float = 0.8,
    cost: EditCost = EditCost.LOW,
    title: str = "Increase visual movement during 20.00-25.00s",
    rationale: str = "Reference variants measured higher motion during this interval.",
    evidence_ids: tuple[str, ...] = ("E1",),
    is_bundle: bool = False,
    modality: str = "visual",
) -> OptimizationRecommendation:
    intervention = ContentIntervention(
        intervention_id=f"{rid}-I0",
        type=intervention_type,
        target_interval_start=start,
        target_interval_end=end,
        description=title,
        rationale=rationale,
        evidence_ids=evidence_ids,
        objective_id="o1",
        expected_direction=ExpectedDirection.TEST_FOR_INCREASE,
        edit_instructions=(EditInstruction(operation="adjust_feature", feature="motion"),),
        edit_cost=cost,
        modality=modality,
    )
    return OptimizationRecommendation(
        recommendation_id=rid,
        title=title,
        objective_id="o1",
        source_variant_id="A",
        interventions=(intervention,),
        target_interval_start=start,
        target_interval_end=end,
        rationale=rationale,
        evidence_ids=evidence_ids,
        evidence_confidence=EvidenceConfidence(
            value=confidence, reference_consistency=1.0, feature_difference_strength=0.5,
            objective_gap_strength=0.5, evidence_count=1, formula="test",
        ),
        expected_direction=ExpectedDirection.TEST_FOR_INCREASE,
        risk=RiskLevel.LOW,
        edit_cost=cost,
        is_bundle=is_bundle,
    )


def _evidence_map(*ids: str) -> dict:
    from blackmirror.optimization.schemas import Evidence, EvidenceKind

    return {
        item: Evidence(
            evidence_id=item, kind=EvidenceKind.CONTENT_FEATURE_DELTA, description="d"
        )
        for item in ids
    }


class TestStep67Constraints:
    """A constraint the user set is not a preference the optimizer may trade."""

    def test_a_frozen_modality_rejects_its_intervention(self) -> None:
        constraints = OptimizationConstraints(frozen_modalities=("audio",))
        problems = validate(
            _recommendation(intervention_type=InterventionType.AUDIO_EDIT),
            _evidence_map("E1"),
            constraints,
        )
        assert any("audio" in p and "froze" in p for p in problems)

    def test_a_preserved_interval_rejects_an_overlapping_edit(self) -> None:
        constraints = OptimizationConstraints(preserve_intervals=((22.0, 24.0),))
        problems = validate(_recommendation(start=20.0, end=25.0), _evidence_map("E1"), constraints)
        assert any("preserved interval" in p for p in problems)

    def test_a_non_overlapping_edit_survives(self) -> None:
        constraints = OptimizationConstraints(preserve_intervals=((0.0, 10.0),))
        result = validate(
            _recommendation(start=20.0, end=25.0), _evidence_map("E1"), constraints
        )
        assert result == []

    def test_forbidding_voiceover_rejects_speech_edits(self) -> None:
        constraints = OptimizationConstraints(forbid_new_voiceover=True)
        problems = validate(
            _recommendation(intervention_type=InterventionType.SPEECH_EDIT),
            _evidence_map("E1"),
            constraints,
        )
        assert any("forbade" in p for p in problems)


class TestStep68CausalLanguage:
    """Once a causal claim reaches a UI, nobody reads the caveat under it."""

    @pytest.mark.parametrize(
        "text",
        [
            "This will increase purchase intent.",
            "This change guarantees a stronger response.",
            "Moving the CTA causes higher engagement.",
            "This will improve conversions.",
            "This is the optimal edit.",
            "This creates emotion in the viewer.",
        ],
    )
    def test_outcome_claims_are_caught(self, text: str) -> None:
        assert scan_language(text), f"should have been flagged: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "Increase visual movement during 20.00-25.00s",
            "Higher-scoring variants measured more motion during this interval.",
            "This is a candidate intervention to test.",
            "Associated with a higher value on the selected objective.",
        ],
    )
    def test_hypothesis_language_passes(self, text: str) -> None:
        assert scan_language(text) == [], f"should not have been flagged: {text}"

    def test_a_causal_recommendation_is_rejected_not_softened(self) -> None:
        problems = validate(
            _recommendation(rationale="This will increase conversions."),
            _evidence_map("E1"),
            OptimizationConstraints(),
        )
        assert any("unsupported claim" in p for p in problems)


class TestStep69EvidenceRequired:
    def test_a_recommendation_citing_unknown_evidence_is_rejected(self) -> None:
        problems = validate(
            _recommendation(evidence_ids=("MISSING",)),
            _evidence_map("E1"),
            OptimizationConstraints(),
        )
        assert any("does not exist" in p for p in problems)

    def test_an_intervention_without_edit_instructions_is_rejected(self) -> None:
        recommendation = _recommendation()
        stripped = recommendation.interventions[0].model_copy(update={"edit_instructions": ()})
        broken = recommendation.model_copy(update={"interventions": (stripped,)})
        problems = validate(broken, _evidence_map("E1"), OptimizationConstraints())
        assert any("no machine-readable edit instruction" in p for p in problems)

    def test_an_empty_interval_is_rejected(self) -> None:
        problems = validate(
            _recommendation(start=20.0, end=20.0), _evidence_map("E1"), OptimizationConstraints()
        )
        assert any("empty" in p for p in problems)


class TestStep70Deduplication:
    """Same intervention type over the same interval is one test, not two."""

    def test_structurally_identical_recommendations_collapse(self) -> None:
        kept, dropped = deduplicate(
            [
                _recommendation("R1", title="Increase visual movement", confidence=0.6),
                _recommendation("R2", title="Add more motion", confidence=0.9),
            ]
        )
        assert len(kept) == 1
        assert kept[0].recommendation_id == "R2", "the better-supported one survives"
        assert dropped[0]["recommendation_id"] == "R1"
        assert "duplicate of R2" in dropped[0]["reason"]

    def test_different_intervals_are_not_duplicates(self) -> None:
        kept, dropped = deduplicate(
            [_recommendation("R1", start=0.0, end=5.0), _recommendation("R2", start=20.0, end=25.0)]
        )
        assert len(kept) == 2
        assert dropped == []

    def test_different_types_are_not_duplicates(self) -> None:
        kept, _ = deduplicate(
            [
                _recommendation("R1", intervention_type=InterventionType.PACE_EDIT),
                _recommendation("R2", intervention_type=InterventionType.SPEECH_EDIT),
            ]
        )
        assert len(kept) == 2


class TestStep71Ranking:
    """priority = evidence_confidence * information_value / cost_weight."""

    def test_priority_is_computed_exactly(self) -> None:
        ranked = rank(
            [_recommendation("R1", confidence=0.8, cost=EditCost.LOW)],
            max_recommendations=5,
        )
        assert ranked[0].priority == pytest.approx(0.8 * 1.0 / 1.0)

    def test_a_cheaper_edit_outranks_an_equally_supported_expensive_one(self) -> None:
        ranked = rank(
            [
                _recommendation("EXPENSIVE", confidence=0.8, cost=EditCost.HIGH,
                                intervention_type=InterventionType.SHOT_EDIT),
                _recommendation("CHEAP", confidence=0.8, cost=EditCost.LOW),
            ],
            max_recommendations=5,
        )
        assert ranked[0].recommendation_id == "CHEAP"

    def test_a_bundle_is_discounted_against_an_equal_atomic_test(self) -> None:
        """A bundle answers a vaguer question, so it ranks below an atomic one."""
        ranked = rank(
            [
                _recommendation("BUNDLE", confidence=0.9, cost=EditCost.LOW, is_bundle=True,
                                intervention_type=InterventionType.SHOT_EDIT),
                _recommendation("ATOMIC", confidence=0.7, cost=EditCost.LOW),
            ],
            max_recommendations=5,
        )
        assert ranked[0].recommendation_id == "ATOMIC"
        assert ranked[0].priority == pytest.approx(0.7)
        assert ranked[1].priority == pytest.approx(0.9 * 0.6)

    def test_diversification_avoids_five_variations_of_one_idea(self) -> None:
        recommendations = [
            _recommendation(f"P{i}", confidence=0.9 - i * 0.01, start=float(i * 10),
                            end=float(i * 10 + 5))
            for i in range(4)
        ] + [_recommendation("SPEECH", confidence=0.5,
                             intervention_type=InterventionType.SPEECH_EDIT)]
        ranked = rank(recommendations, max_recommendations=2)
        types = {i.type for r in ranked for i in r.interventions}
        assert len(types) == 2, "the second slot goes to a different intervention type"

    def test_ranking_is_deterministic(self) -> None:
        recommendations = [
            _recommendation("R1", confidence=0.5),
            _recommendation("R2", confidence=0.5, start=0.0, end=5.0),
        ]
        first = [r.recommendation_id for r in rank(recommendations, max_recommendations=5)]
        second = [r.recommendation_id for r in rank(recommendations, max_recommendations=5)]
        assert first == second, "ties must break deterministically"


class TestOrchestrator:
    """The whole chain, on synthetic data with a planted difference."""

    @staticmethod
    def _setup():
        # A is weak in its first half; B is strong throughout and has motion there.
        engine = GoalConditionedScoringEngine()
        score = engine.score_experiment(
            "exp",
            # A: flat first half (weak), modest middle, a clear peak at the end
            # (strong). B is uniformly strong and carries speech in A's weak window.
            (_variant("A", [0.0] * 10 + [1.0] * 8 + [6.0] * 2), _variant("B", [2.0] * 20)),
            (_objective(),),
            baseline_variant_id="A",
        )
        content = {
            "A": _series("A", [0.2] * 20, [0.0] * 20),
            "B": _series("B", [0.6] * 20, [1.0] * 20),
        }
        events = {
            "A": [{"event_type": "scene", "start_time": 0.0, "end_time": 20.0}],
            "B": [
                {"event_type": "scene", "start_time": 0.0, "end_time": 20.0},
                {"event_type": "speech_segment", "start_time": 0.0, "end_time": 8.0},
            ],
        }
        return score, content, events

    def _request(self, **overrides) -> OptimizationRequest:
        base = {
            "experiment_id": "exp",
            "source_variant_id": "A",
            "objective_set_hash": "0" * 16,
            "reference_strategy": ReferenceStrategy.BEST_SCORE,
            "strategy": OptimizationStrategy.MINIMAL_EDIT,
        }
        return OptimizationRequest(**{**base, **overrides})

    def test_the_full_chain_produces_grounded_recommendations(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(), score, content=content, events=events
        )
        assert result.reference_variant_ids == ("B",)
        assert result.weak_intervals, "A's flat first half must be weak"
        assert result.recommendations
        for recommendation in result.recommendations:
            assert recommendation.evidence_ids
            known = {e.evidence_id for e in result.evidence}
            assert set(recommendation.evidence_ids) <= known, "every citation must resolve"
            assert recommendation.hypothesis_id is not None

    def test_every_recommendation_has_a_traceable_hypothesis(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(), score, content=content, events=events
        )
        hypothesis_ids = {h.hypothesis_id for h in result.hypotheses}
        for recommendation in result.recommendations:
            assert recommendation.hypothesis_id in hypothesis_ids
        for hypothesis in result.hypotheses:
            assert hypothesis.outcome == "untested", "Phase 7 never sets an outcome"
            assert "hypothesis" in hypothesis.statement.lower()

    def test_traces_link_recommendation_to_evidence_and_interval(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(), score, content=content, events=events
        )
        assert result.traces
        for trace in result.traces:
            assert trace.evidence_ids
            assert trace.reference_variant_ids == ("B",)

    def test_the_feature_scan_count_is_reported(self) -> None:
        """Scanning many features will surface differences by chance."""
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(), score, content=content, events=events
        )
        assert result.metadata.features_considered > 0
        assert any("by chance" in w for w in result.warnings)

    def test_constraints_are_enforced_end_to_end(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(
                constraints=OptimizationConstraints(frozen_modalities=("visual", "speech"))
            ),
            score,
            content=content,
            events=events,
        )
        for recommendation in result.recommendations:
            for intervention in recommendation.interventions:
                assert intervention.modality not in {"visual", "speech"}
        assert result.rejected, "the frozen interventions must be recorded as rejected"

    def test_no_recommendation_contains_causal_language(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(), score, content=content, events=events
        )
        for recommendation in result.recommendations:
            assert scan_language(recommendation.title) == []
            assert scan_language(recommendation.rationale) == []
            for intervention in recommendation.interventions:
                assert scan_language(intervention.description) == []
                assert scan_language(intervention.rationale) == []

    def test_step72_reproducibility(self) -> None:
        """Same context, same parameters, same evidence graph and ranking."""
        score, content, events = self._setup()
        orchestrator = OptimizationOrchestrator()
        first = orchestrator.optimize(self._request(), score, content=content, events=events)
        second = orchestrator.optimize(self._request(), score, content=content, events=events)
        assert [e.evidence_id for e in first.evidence] == [
            e.evidence_id for e in second.evidence
        ]
        assert [r.recommendation_id for r in first.recommendations] == [
            r.recommendation_id for r in second.recommendations
        ]
        assert [r.priority for r in first.recommendations] == [
            r.priority for r in second.recommendations
        ]

    def test_an_unknown_source_variant_is_refused(self) -> None:
        score, content, events = self._setup()
        with pytest.raises(ValueError, match="not in this score result"):
            OptimizationOrchestrator().optimize(
                self._request(source_variant_id="ghost"), score, content=content, events=events
            )

    def test_a_source_with_no_better_reference_says_so(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(source_variant_id="B"), score, content=content, events=events
        )
        assert any("no variant scores higher" in w for w in result.warnings)

    def test_exploratory_strategy_may_bundle_but_labels_the_cost(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(strategy=OptimizationStrategy.EXPLORATORY, max_recommendations=10),
            score,
            content=content,
            events=events,
        )
        bundles = [r for r in result.recommendations if r.is_bundle]
        for bundle in bundles:
            assert bundle.risk is RiskLevel.HIGH
            assert any("attributed" in w for w in bundle.warnings)

    def test_strong_intervals_are_available_for_preservation(self) -> None:
        score, content, events = self._setup()
        result = OptimizationOrchestrator().optimize(
            self._request(), score, content=content, events=events
        )
        assert result.strong_intervals, "A's high second half must be marked strong"
        for interval in result.strong_intervals:
            assert interval.kind is IntervalKind.STRONG


class TestCausalFilterPrecision:
    """The filter must catch assertions without rejecting disclaimers.

    Found by the orchestrator test: the generator's own honest wording -- "not a
    demonstrated cause" -- was rejected, because the pattern matched the noun
    rather than an assertion. A filter that rejects the text keeping a
    recommendation honest is worse than no filter.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "This is a temporal comparison, not a demonstrated cause.",
            "Presence is a measured difference, not a demonstrated cause.",
            "No causal relationship is established here.",
        ],
    )
    def test_disclaimers_are_not_flagged(self, text: str) -> None:
        assert scan_language(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Moving the CTA causes higher response",
            "This will cause a stronger response",
            "The edit caused the improvement",
        ],
    )
    def test_assertions_are_still_flagged(self, text: str) -> None:
        assert scan_language(text)
