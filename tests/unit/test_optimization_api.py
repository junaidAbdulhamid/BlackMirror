"""Persistence, human approval and the Phase 7 API contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blackmirror.optimization.engine import OptimizationOrchestrator
from blackmirror.optimization.schemas import (
    ApprovalState,
    ContentIntervention,
    EditInstruction,
    ExpectedDirection,
    InterventionType,
    OptimizationRequest,
    RecommendationReview,
)
from blackmirror.optimization.storage import OptimizationStore, build_spec
from blackmirror.scoring.engine import GoalConditionedScoringEngine
from blackmirror.scoring.evaluator import ScoringVariant
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)

fastapi = pytest.importorskip("fastapi")


def _variant(vid: str, values: list[float]) -> ScoringVariant:
    times = np.arange(len(values), dtype=np.float64)
    roi = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return ScoringVariant(
        variant_id=vid, responses=np.tile(roi, (1, 4)), times=times, roi_timeseries=roi,
        region_names=("r0",), medial_wall_mask=np.zeros(4, dtype=bool),
        duration_seconds=float(len(values)),
        hemisphere_ranges={"left": (0, 2), "right": (2, 4)},
    )


def _result():
    objective = NeuralObjective(
        objective_id="o1", name="o1", metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.ROI, region_id=0),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE,
    )
    score = GoalConditionedScoringEngine().score_experiment(
        "exp",
        (_variant("A", [0.0] * 10 + [1.0] * 8 + [6.0] * 2), _variant("B", [2.0] * 20)),
        (objective,),
    )
    from blackmirror.optimization.engine import content_series_from_arrays

    names = ("motion", "speech_present", "audio_energy", "visual_available")
    def series(vid: str, motion: float, speech: float):
        times = np.arange(20, dtype=np.float64)
        matrix = np.column_stack(
            [[motion] * 20, [speech] * 20, [0.5] * 20, [1.0] * 20]
        ).astype(np.float64)
        return content_series_from_arrays(vid, times, matrix, names)

    request = OptimizationRequest(
        experiment_id="exp", source_variant_id="A", objective_set_hash="0" * 16
    )
    return OptimizationOrchestrator().optimize(
        request,
        score,
        content={"A": series("A", 0.2, 0.0), "B": series("B", 0.7, 1.0)},
        events={
            "A": [{"event_type": "scene", "start_time": 0.0, "end_time": 20.0}],
            "B": [{"event_type": "speech_segment", "start_time": 0.0, "end_time": 9.0}],
        },
    )


class TestOptimizationStore:
    def test_round_trip_preserves_recommendations(self, tmp_path: Path) -> None:
        store = OptimizationStore(tmp_path)
        result = _result()
        store.write(result, "abc12345")
        back = store.read("exp", "abc12345")
        assert [r.recommendation_id for r in back.recommendations] == [
            r.recommendation_id for r in result.recommendations
        ]

    def test_each_artifact_is_written_separately(self, tmp_path: Path) -> None:
        store = OptimizationStore(tmp_path)
        path = store.write(_result(), "abc12345")
        names = {item.name for item in path.iterdir()}
        assert {
            "request.json", "metadata.json", "weak_intervals.json", "evidence.json",
            "recommendations.json", "hypotheses.json", "traces.json",
        } <= names

    def test_reviews_survive_re_optimization(self, tmp_path: Path) -> None:
        """A human decision is not derived data and must not be recomputed away."""
        store = OptimizationStore(tmp_path)
        result = _result()
        store.write(result, "abc12345")
        target = result.recommendations[0].recommendation_id
        store.record_review(
            "exp", "abc12345",
            RecommendationReview(recommendation_id=target, state=ApprovalState.REJECTED,
                                 reason="violates brand"),
        )
        store.write(result, "abc12345")  # re-optimize with the same key
        reviews = store.reviews("exp", "abc12345")
        assert len(reviews) == 1
        assert reviews[0].reason == "violates brand"

    def test_a_second_decision_replaces_the_first(self, tmp_path: Path) -> None:
        store = OptimizationStore(tmp_path)
        result = _result()
        store.write(result, "abc12345")
        target = result.recommendations[0].recommendation_id
        store.record_review(
            "exp", "abc12345",
            RecommendationReview(recommendation_id=target, state=ApprovalState.REJECTED),
        )
        reviews = store.record_review(
            "exp", "abc12345",
            RecommendationReview(recommendation_id=target, state=ApprovalState.APPROVED),
        )
        assert len(reviews) == 1
        assert reviews[0].state is ApprovalState.APPROVED

    def test_path_traversal_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid experiment id"):
            OptimizationStore(tmp_path).directory("../escape", "abc12345")


class TestCandidateSpec:
    """Recommendations never become candidates on their own."""

    def test_no_approval_means_no_candidate(self) -> None:
        result = _result()
        with pytest.raises(ValueError, match="approved or modified"):
            build_spec(result, [], proposed_variant_id="cand-1")

    def test_a_rejection_does_not_produce_a_candidate(self) -> None:
        result = _result()
        reviews = [
            RecommendationReview(
                recommendation_id=result.recommendations[0].recommendation_id,
                state=ApprovalState.REJECTED,
            )
        ]
        with pytest.raises(ValueError, match="approved or modified"):
            build_spec(result, reviews, proposed_variant_id="cand-1")

    def test_an_approved_recommendation_becomes_a_spec(self) -> None:
        result = _result()
        recommendation = result.recommendations[0]
        spec = build_spec(
            result,
            [
                RecommendationReview(
                    recommendation_id=recommendation.recommendation_id,
                    state=ApprovalState.APPROVED,
                )
            ],
            proposed_variant_id="cand-1",
        )
        assert spec.parent_variant_id == "A"
        assert spec.hypothesis_ids == (recommendation.hypothesis_id,)
        assert spec.edit_instructions
        assert spec.target_objective_id == recommendation.objective_id

    def test_a_modified_review_supplies_its_own_interventions(self) -> None:
        """The user's edit is used, and the original stays in the stored result."""
        result = _result()
        recommendation = result.recommendations[0]
        replacement = ContentIntervention(
            intervention_id="user-1",
            type=InterventionType.TIMING_EDIT,
            target_interval_start=6.0,
            target_interval_end=8.0,
            description="User-adjusted timing",
            rationale="Reviewer preferred a different placement.",
            evidence_ids=recommendation.evidence_ids,
            objective_id=recommendation.objective_id,
            expected_direction=ExpectedDirection.TEST_FOR_INCREASE,
            edit_instructions=(EditInstruction(operation="move_event", to_time=6.5),),
        )
        spec = build_spec(
            result,
            [
                RecommendationReview(
                    recommendation_id=recommendation.recommendation_id,
                    state=ApprovalState.MODIFIED,
                    modified_interventions=(replacement,),
                )
            ],
            proposed_variant_id="cand-2",
        )
        assert spec.interventions == (replacement,)
        assert result.recommendations[0].interventions != (replacement,), (
            "the original recommendation must be preserved unchanged"
        )

    def test_strong_intervals_become_preserve_regions(self) -> None:
        result = _result()
        spec = build_spec(
            result,
            [
                RecommendationReview(
                    recommendation_id=result.recommendations[0].recommendation_id,
                    state=ApprovalState.APPROVED,
                )
            ],
            proposed_variant_id="cand-3",
        )
        assert spec.preserve_intervals, "an interval already carrying the objective is preserved"

    def test_a_spec_makes_no_improvement_claim(self) -> None:
        result = _result()
        spec = build_spec(
            result,
            [
                RecommendationReview(
                    recommendation_id=result.recommendations[0].recommendation_id,
                    state=ApprovalState.APPROVED,
                )
            ],
            proposed_variant_id="cand-4",
        )
        assert "No claim is made" in spec.note
        assert spec.expected_direction.value.startswith("test_for")


class TestOptimizationApi:
    @staticmethod
    def _client(tmp_path: Path):
        from fastapi.testclient import TestClient

        from blackmirror.api import app as module
        from blackmirror.api.optimization_loader import OptimizationLoader

        client = TestClient(module.app)
        module.app.dependency_overrides[module.optimization_loader_dep] = (
            lambda: OptimizationLoader(tmp_path)
        )
        return client, module

    def test_optimizing_without_a_stored_score_is_a_conflict(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            response = client.post(
                "/api/experiments/exp/optimize",
                json={
                    "request": {
                        "experiment_id": "exp", "source_variant_id": "A",
                        "objective_set_hash": "0" * 16,
                    }
                },
            )
            assert response.status_code == 409
            assert "score the experiment first" in response.json()["detail"]
        finally:
            module.app.dependency_overrides.clear()

    def test_a_mismatched_experiment_id_is_rejected(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            response = client.post(
                "/api/experiments/other/optimize",
                json={
                    "request": {
                        "experiment_id": "exp", "source_variant_id": "A",
                        "objective_set_hash": "0" * 16,
                    }
                },
            )
            assert response.status_code == 422
        finally:
            module.app.dependency_overrides.clear()

    def test_manual_reference_without_ids_is_rejected(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            response = client.post(
                "/api/experiments/exp/optimize",
                json={
                    "request": {
                        "experiment_id": "exp", "source_variant_id": "A",
                        "objective_set_hash": "0" * 16,
                        "reference_strategy": "manual",
                    }
                },
            )
            assert response.status_code == 422
        finally:
            module.app.dependency_overrides.clear()

    def test_listing_an_unknown_experiment_is_empty_not_an_error(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            assert client.get("/api/experiments/nothing/optimization").json() == []
        finally:
            module.app.dependency_overrides.clear()
