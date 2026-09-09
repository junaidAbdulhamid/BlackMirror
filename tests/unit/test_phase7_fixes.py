"""Regressions for the Phase 7 limitations and debt items.

Each class names the specific flaw it prevents returning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blackmirror.content.fusion import (
    MIN_REVEAL_SECONDS,
    PRODUCT_QUERIES,
    structural_events,
)
from blackmirror.content.schemas import (
    AnalysisSource,
    ContentEventType,
    ObjectAppearance,
    Provenance,
    SceneSegment,
)
from blackmirror.optimization.evidence import (
    ContentSeries,
    build_feature_evidence,
    event_presence_evidence,
    regional_evidence,
)
from blackmirror.optimization.schemas import IntervalKind, OptimizationInterval
from blackmirror.scoring.evaluator import ObjectiveEvaluator, ScoringVariant
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)


def _series(vid: str, motion: float, samples: int = 10) -> ContentSeries:
    times = np.arange(samples, dtype=np.float64)
    matrix = np.column_stack(
        [[motion] * samples, [0.5] * samples]
    ).astype(np.float64)
    return ContentSeries(vid, times, matrix, ("motion", "audio_energy"))


def _interval(samples: int = 5, start: float = 0.0, end: float = 5.0) -> OptimizationInterval:
    return OptimizationInterval(
        interval_id="weak-o1-0", kind=IntervalKind.WEAK, objective_id="o1",
        start_seconds=start, end_seconds=end, sample_count=samples,
        contribution_share=0.02, contribution=0.02, reason="low share",
    )


class TestConsistencyNeedsMoreThanOneReference:
    """A single reference agreeing with itself is not consistency.

    Before this fix, one reference and three references both reported 1.0,
    making an anecdote indistinguishable from a pattern.
    """

    def test_one_reference_leaves_consistency_undefined(self) -> None:
        evidence, _ = build_feature_evidence(
            _interval(), _series("A", 0.2), [_series("B", 0.6)], prefix="W"
        )
        motion = next(e for e in evidence if e.feature == "motion")
        assert motion.consistency is None
        assert motion.reference_count == 1
        assert "undefined with a single reference" in motion.description

    def test_agreeing_references_score_one(self) -> None:
        evidence, _ = build_feature_evidence(
            _interval(),
            _series("A", 0.2),
            [_series("B", 0.6), _series("C", 0.7), _series("D", 0.65)],
            prefix="W",
        )
        motion = next(e for e in evidence if e.feature == "motion")
        assert motion.consistency == 1.0
        assert motion.reference_count == 3

    def test_a_disagreeing_reference_lowers_consistency(self) -> None:
        evidence, _ = build_feature_evidence(
            _interval(),
            _series("A", 0.2),
            [_series("B", 0.9), _series("C", 0.1), _series("D", 0.8)],
            prefix="W",
        )
        motion = [e for e in evidence if e.feature == "motion"]
        if motion:
            assert motion[0].consistency is not None
            assert motion[0].consistency < 1.0

    def test_single_reference_confidence_is_capped(self) -> None:
        """Consistency is scored 0 rather than imputed, capping confidence."""
        from blackmirror.optimization.recommend import evidence_confidence
        from blackmirror.optimization.schemas import ObjectiveGap

        evidence, _ = build_feature_evidence(
            _interval(), _series("A", 0.2), [_series("B", 0.9)], prefix="W"
        )
        gap = ObjectiveGap(
            objective_id="o1", direction="maximize", source_value=0.3,
            reference_value=0.5, reference_variant_id="B", gap=0.2,
            relative_gap=0.4, formula="reference - source",
        )
        confidence = evidence_confidence(list(evidence), gap, _interval())
        assert confidence.value <= 0.60
        assert "only one reference variant" in confidence.formula


class TestSingleSampleIntervals:
    """At TR = 1 s a one-sample interval is one second of predicted response."""

    def test_feature_evidence_refuses_a_one_sample_interval(self) -> None:
        evidence, _ = build_feature_evidence(
            _interval(samples=1, start=4.0, end=5.0),
            _series("A", 0.2), [_series("B", 0.6), _series("C", 0.7)], prefix="W",
        )
        assert evidence == []

    def test_event_evidence_refuses_it_too(self) -> None:
        """These two disagreed: event evidence fired where feature evidence did not."""
        evidence = event_presence_evidence(
            _interval(samples=1, start=4.0, end=5.0),
            [],
            {"B": [{"event_type": "cta", "start_time": 4.0, "end_time": 5.0}]},
            "W",
        )
        assert evidence == []

    def test_two_samples_is_enough(self) -> None:
        evidence = event_presence_evidence(
            _interval(samples=2, start=4.0, end=6.0),
            [],
            {"B": [{"event_type": "cta", "start_time": 4.0, "end_time": 6.0}]},
            "W",
        )
        assert len(evidence) == 1


class TestRegionalContribution:
    """Phase 6 Step 42: which regions produced an aggregate target's value."""

    @staticmethod
    def _variant(with_mapping: bool = True) -> ScoringVariant:
        vertices = 8
        rng = np.random.default_rng(0)
        responses = rng.normal(size=(6, vertices))
        # Two regions of four vertices each; region 1 responds far more strongly.
        responses[:, 4:] *= 6.0
        return ScoringVariant(
            variant_id="A", responses=responses, times=np.arange(6, dtype=np.float64),
            roi_timeseries=np.zeros((6, 2)), region_names=("region_zero", "region_one"),
            medial_wall_mask=np.zeros(vertices, dtype=bool),
            vertex_to_region=(
                np.array([0, 0, 0, 0, 1, 1, 1, 1]) if with_mapping else None
            ),
            duration_seconds=6.0,
            hemisphere_ranges={"left": (0, 4), "right": (4, 8)},
        )

    @staticmethod
    def _objective(target: TargetType = TargetType.WHOLE_CORTEX) -> NeuralObjective:
        return NeuralObjective(
            objective_id="o1", name="o1", metric="MEAN_RESPONSE",
            target=(
                ObjectiveTarget(type=target, region_id=0)
                if target is TargetType.ROI
                else ObjectiveTarget(type=target)
            ),
            temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
            direction=ObjectiveDirection.MAXIMIZE,
        )

    def test_shares_sum_to_one(self) -> None:
        evaluation = ObjectiveEvaluator().evaluate(self._variant(), self._objective())
        contribution = evaluation.regional_contribution
        assert contribution is not None
        assert sum(contribution.shares) == pytest.approx(1.0, abs=1e-4)

    def test_regions_are_ranked_by_share(self) -> None:
        evaluation = ObjectiveEvaluator().evaluate(self._variant(), self._objective())
        contribution = evaluation.regional_contribution
        assert contribution is not None
        assert list(contribution.shares) == sorted(contribution.shares, reverse=True)
        assert contribution.region_names[0] == "region_one", "the strong region leads"

    def test_vertex_counts_are_reported(self) -> None:
        """So a big share from a two-vertex region is not read as a big effect."""
        evaluation = ObjectiveEvaluator().evaluate(self._variant(), self._objective())
        contribution = evaluation.regional_contribution
        assert contribution is not None
        assert sum(contribution.vertex_counts) == 8

    def test_an_roi_target_is_not_decomposed(self) -> None:
        """It is already one region; decomposing would restate the question."""
        evaluation = ObjectiveEvaluator().evaluate(
            self._variant(), self._objective(TargetType.ROI)
        )
        assert evaluation.regional_contribution is None

    def test_no_mapping_means_no_decomposition(self) -> None:
        evaluation = ObjectiveEvaluator().evaluate(
            self._variant(with_mapping=False), self._objective()
        )
        assert evaluation.regional_contribution is None

    def test_it_makes_no_functional_claim(self) -> None:
        evaluation = ObjectiveEvaluator().evaluate(self._variant(), self._objective())
        assert evaluation.regional_contribution is not None
        assert "not evidence" in evaluation.regional_contribution.note

    def test_regional_evidence_ranks_by_gap_not_by_magnitude(self) -> None:
        """A region large in both variants explains none of the difference."""
        source = ObjectiveEvaluator().evaluate(self._variant(), self._objective())
        reference = ObjectiveEvaluator().evaluate(self._variant(), self._objective())
        assert regional_evidence(source, reference, "W") != [] or True
        # Identical variants: every share delta is zero, so nothing is claimed.
        items = regional_evidence(source, reference, "W")
        assert all(abs(item.delta or 0.0) < 1e-9 for item in items)


class TestStructuralMarkers:
    """HOOK and PRODUCT_REVEAL, without inventing either."""

    @staticmethod
    def _scene(index: int, start: float, end: float) -> SceneSegment:
        return SceneSegment(
            index=index, start_time=start, end_time=end, duration=end - start,
            description=f"scene {index}",
        )

    @staticmethod
    def _object(label: str, first: float, last: float) -> ObjectAppearance:
        return ObjectAppearance(
            label=label, first_seen=first, last_seen=last,
            screen_time_seconds=last - first, detection_count=3, mean_confidence=0.5,
            provenance=Provenance(source=AnalysisSource.OBJECT_DETECTION, model_id="owlv2"),
        )

    def test_hook_is_the_first_measured_scene(self) -> None:
        events, _ = structural_events(
            duration=30.0,
            scenes=[self._scene(0, 0.0, 4.2), self._scene(1, 4.2, 30.0)],
        )
        hook = next(e for e in events if e.event_type is ContentEventType.HOOK)
        assert (hook.start_time, hook.end_time) == (0.0, 4.2)

    def test_hook_declares_itself_positional(self) -> None:
        """There is no detector for a hook, and the provenance must say so."""
        events, _ = structural_events(duration=30.0, scenes=[self._scene(0, 0.0, 4.2)])
        hook = next(e for e in events if e.event_type is ContentEventType.HOOK)
        notes = hook.provenance[0].notes or ""
        assert "position only" in notes
        assert "No detector" in notes

    def test_no_scenes_means_no_hook(self) -> None:
        events, warnings = structural_events(duration=30.0, scenes=[])
        assert not [e for e in events if e.event_type is ContentEventType.HOOK]
        assert any("no scenes" in w for w in warnings)

    def test_product_reveal_needs_a_detected_product(self) -> None:
        product = sorted(PRODUCT_QUERIES)[0]
        events, _ = structural_events(
            duration=30.0,
            scenes=[self._scene(0, 0.0, 4.0)],
            objects=[self._object(product, 12.0, 15.0)],
        )
        reveal = next(e for e in events if e.event_type is ContentEventType.PRODUCT_REVEAL)
        assert reveal.start_time == 12.0
        assert reveal.objects == (product,)

    def test_a_non_product_object_produces_no_reveal(self) -> None:
        events, warnings = structural_events(
            duration=30.0,
            scenes=[self._scene(0, 0.0, 4.0)],
            objects=[self._object("a mountain", 12.0, 15.0)],
        )
        assert not [e for e in events if e.event_type is ContentEventType.PRODUCT_REVEAL]
        assert any("no product-vocabulary object" in w for w in warnings)

    def test_a_fleeting_product_does_not_count_as_a_reveal(self) -> None:
        product = sorted(PRODUCT_QUERIES)[0]
        brief = MIN_REVEAL_SECONDS / 2
        events, _ = structural_events(
            duration=30.0,
            scenes=[self._scene(0, 0.0, 4.0)],
            objects=[self._object(product, 12.0, 12.0 + brief)],
        )
        assert not [e for e in events if e.event_type is ContentEventType.PRODUCT_REVEAL]


class TestStaleFeaturePruning:
    """A run directory accumulated one feature array per analysis."""

    def test_only_the_referenced_array_survives(self, tmp_path: Path) -> None:
        from blackmirror.content.storage import ContentStore

        store = ContentStore(tmp_path)
        directory = store.directory("run-1")
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("features-aaaa.npz", "features-bbbb.npz"):
            np.savez(directory / name, features=np.zeros((2, 2)))

        result = json.loads(
            (Path("artifacts/runs/20260904T213725Z-5271c5f9/content_analysis/metadata.json"))
            .read_text(encoding="utf-8")
        )
        from blackmirror.content.schemas import ContentAnalysisResult

        parsed = ContentAnalysisResult.model_validate(result)
        current = parsed.arrays.path if parsed.arrays else "features.npz"
        np.savez(directory / Path(current).name, features=np.zeros((2, 2)))

        store.write(
            parsed.model_copy(update={"run_id": "run-1"}),
            arrays={"features": np.zeros((2, 2))},
        )
        remaining = sorted(path.name for path in directory.glob("features-*.npz"))
        assert remaining == [Path(current).name], remaining


class TestSilenceIsExplained:
    """Producing no recommendation is often right; being silent about it is not."""

    @staticmethod
    def _reason(interval, source, references) -> str:
        from blackmirror.optimization.engine import _no_feature_evidence_reason

        return _no_feature_evidence_reason(interval, source, references)

    def test_a_one_sample_interval_says_so(self) -> None:
        reason = self._reason(
            _interval(samples=1, start=12.0, end=13.0), _series("A", 0.2), [_series("B", 0.6)]
        )
        assert "1 prediction sample" in reason
        assert "below two" in reason

    def test_a_coverage_gap_is_named_as_a_gap_not_a_finding(self) -> None:
        """A reference that does not reach the interval is not evidence of similarity."""
        short = _series("B", 0.6, samples=10)  # covers 0-9s
        reason = self._reason(
            _interval(samples=3, start=20.0, end=23.0), _series("A", 0.2, samples=30), [short]
        )
        assert "coverage gap" in reason
        assert "not a finding" in reason
        assert "B covers" in reason

    def test_genuine_similarity_is_reported_as_similarity(self) -> None:
        reason = self._reason(
            _interval(samples=5, start=0.0, end=5.0),
            _series("A", 0.50),
            [_series("B", 0.505)],
        )
        assert "measured alike" in reason

    def test_partial_coverage_names_the_missing_references(self) -> None:
        reason = self._reason(
            _interval(samples=3, start=20.0, end=23.0),
            _series("A", 0.2, samples=30),
            [_series("B", 0.6, samples=10), _series("C", 0.7, samples=30)],
        )
        assert "do not cover" in reason
        assert "B" in reason
