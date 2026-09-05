"""Persistence, explanation and API contract for Phase 6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blackmirror.api.scoring_loader import ScoringLoader
from blackmirror.scoring.engine import GoalConditionedScoringEngine
from blackmirror.scoring.evaluator import ScoringVariant
from blackmirror.scoring.explanation import explain
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.scoring.storage import ScoringStore

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


def _objective(oid: str = "o1", weight: float = 1.0) -> NeuralObjective:
    return NeuralObjective(
        objective_id=oid, name=f"objective {oid}", metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.ROI, region_id=0),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE, weight=weight,
    )


def _result(experiment_id: str = "e1"):
    return GoalConditionedScoringEngine().score_experiment(
        experiment_id, (_variant("A", [1.0, 2.0]), _variant("B", [2.0, 3.0])),
        (_objective(),), baseline_variant_id="A",
    )


class TestScoringStore:
    def test_round_trip_preserves_the_ranking(self, tmp_path: Path) -> None:
        store = ScoringStore(tmp_path)
        result = _result()
        store.write(result)
        back = store.read("e1", result.metadata.objective_set_hash)
        assert back.ranking == result.ranking
        assert back.metadata.scoring_version == result.metadata.scoring_version

    def test_a_stored_score_is_not_silently_replaced(self, tmp_path: Path) -> None:
        """A stored score is evidence; overwriting it would break a citation."""
        store = ScoringStore(tmp_path)
        result = _result()
        store.write(result)
        with pytest.raises(FileExistsError, match="already exists"):
            store.write(result)
        store.write(result, overwrite=True)  # explicit replacement is allowed

    def test_the_objective_set_is_stored_separately_for_reuse(self, tmp_path: Path) -> None:
        store = ScoringStore(tmp_path)
        result = _result()
        path = store.write(result)
        objectives = json.loads((path / "objective_set.json").read_text())
        assert objectives[0]["metric"] == "MEAN_RESPONSE"

    def test_the_same_objective_set_lands_in_the_same_place(self, tmp_path: Path) -> None:
        """Which is what makes the cache lookup correct rather than convenient."""
        store = ScoringStore(tmp_path)
        first, second = _result(), _result()
        assert first.metadata.objective_set_hash == second.metadata.objective_set_hash
        assert store.directory("e1", first.metadata.objective_set_hash) == store.directory(
            "e1", second.metadata.objective_set_hash
        )

    def test_a_corrupt_entry_does_not_hide_healthy_ones(self, tmp_path: Path) -> None:
        store = ScoringStore(tmp_path)
        result = _result()
        store.write(result)
        broken = store.root / "e1" / "scoring" / "0123456789abcdef"
        broken.mkdir(parents=True)
        (broken / "metadata.json").write_text("{truncated")
        assert len(store.list_for_experiment("e1")) == 1

    def test_path_traversal_is_refused(self, tmp_path: Path) -> None:
        store = ScoringStore(tmp_path)
        with pytest.raises(ValueError, match="invalid experiment id"):
            store.directory("../escape", "0" * 16)

    def test_explanation_types_effect_size_unavailability(self) -> None:
        comparison = explain(_result()).objectives[0]
        assert comparison.paired_effect.status == "unavailable"
        assert comparison.paired_effect.value is None
        assert "autocorrelated" in " ".join(comparison.paired_effect.assumptions)


def test_cache_identity_includes_runs_and_artifact_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blackmirror.api.scoring_loader as module

    fingerprints = {"A": "a1", "B": "b1"}
    monkeypatch.setattr(
        module,
        "_input_fingerprint",
        lambda _root, run_id, **_kwargs: fingerprints[run_id],
    )
    monkeypatch.setattr(
        module,
        "load_variant",
        lambda _root, run_id, **_kwargs: _variant(run_id, [1.0, 2.0]),
    )
    loader = ScoringLoader(tmp_path)
    first = loader.create("experiment", ("A",), (_objective(),))
    second = loader.create("experiment", ("B",), (_objective(),))
    fingerprints["A"] = "a2"
    changed = loader.create("experiment", ("A",), (_objective(),))
    keys = {
        first.metadata.objective_set_hash,
        second.metadata.objective_set_hash,
        changed.metadata.objective_set_hash,
    }
    assert len(keys) == 3
    assert len(loader.scores("experiment")) == 3


class TestExplanation:
    def test_it_reports_the_leader_without_causal_language(self) -> None:
        explanation = explain(_result())
        joined = " ".join(explanation.statements).lower()
        assert "ranked highest" in joined
        for banned in ("because", "caused", "persuasive", "more effective"):
            assert banned not in joined, banned

    def test_conflicting_objectives_are_surfaced(self) -> None:
        variants = (_variant("A", [3.0, 3.0]), _variant("B", [1.0, 1.0]))
        objectives = (
            _objective("o1", weight=0.5),
            NeuralObjective(
                objective_id="o2", name="minimise it", metric="MEAN_RESPONSE",
                target=ObjectiveTarget(type=TargetType.ROI, region_id=0),
                temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
                direction=ObjectiveDirection.MINIMIZE, weight=0.5,
            ),
        )
        result = GoalConditionedScoringEngine().score_experiment("e2", variants, objectives)
        explanation = explain(result)
        assert any("objectives disagree" in s for s in explanation.statements)

    def test_the_non_causal_caveat_is_always_attached(self) -> None:
        explanation = explain(_result())
        assert any("not shown to have caused it" in c for c in explanation.caveats)
        assert any("no p-value" in c for c in explanation.caveats)

    def test_content_context_is_copied_not_inferred(self) -> None:
        descriptions = {
            "B": [
                {
                    "event_type": "cta", "start_time": 0.0, "end_time": 3.0,
                    "speech_text": "Start today.", "visual_description": "product close-up",
                    "objects": ["a product package"],
                }
            ]
        }
        explanation = explain(_result(), content_descriptions=descriptions)
        context = explanation.context.get("B")
        assert context is not None
        assert context.speech_text == "Start today."
        assert context.objects == ("a product package",)

    def test_a_relative_difference_is_withheld_near_zero(self) -> None:
        result = GoalConditionedScoringEngine().score_experiment(
            "e3", (_variant("A", [0.0, 0.0]), _variant("B", [1.0, 1.0])), (_objective(),)
        )
        explanation = explain(result)
        assert explanation.objectives[0].absolute_difference == pytest.approx(1.0)
        assert explanation.objectives[0].relative_difference is None


class TestScoringApi:
    @staticmethod
    def _client(tmp_path: Path):
        from fastapi.testclient import TestClient

        from blackmirror.api import app as app_module

        client = TestClient(app_module.app)
        app_module.app.dependency_overrides[app_module.scoring_loader_dep] = (
            lambda: __import__(
                "blackmirror.api.scoring_loader", fromlist=["ScoringLoader"]
            ).ScoringLoader(tmp_path)
        )
        return client, app_module

    def test_metrics_endpoint_documents_every_metric(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            payload = client.get("/api/objectives/metrics").json()
            assert len(payload) == 6
            assert "REFERENCE_PATTERN_SIMILARITY" not in payload
            for description in payload.values():
                assert {"formula", "interpretation", "limitations"} <= set(description)
        finally:
            module.app.dependency_overrides.clear()

    def test_an_unknown_run_is_a_conflict_not_a_crash(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            response = client.post(
                "/api/experiments/e1/score",
                json={
                    "run_ids": ["missing-run"],
                    "objectives": [_objective().model_dump(mode="json")],
                },
            )
            assert response.status_code == 409
            assert "no run directory" in response.json()["detail"]
        finally:
            module.app.dependency_overrides.clear()

    @pytest.mark.parametrize("unsafe_id", [".", ".."])
    def test_dot_path_run_ids_are_rejected(self, tmp_path: Path, unsafe_id: str) -> None:
        client, module = self._client(tmp_path)
        try:
            response = client.post(
                "/api/experiments/e1/score",
                json={
                    "run_ids": [unsafe_id],
                    "objectives": [_objective().model_dump(mode="json")],
                },
            )
            assert response.status_code == 422
        finally:
            module.app.dependency_overrides.clear()

    def test_a_psychological_metric_is_rejected(self, tmp_path: Path) -> None:
        """The allowlist must hold at the API boundary, not only internally."""
        client, module = self._client(tmp_path)
        try:
            objective = _objective().model_dump(mode="json")
            objective["metric"] = "PURCHASE_INTENT"
            response = client.post(
                "/api/experiments/e1/score",
                json={"run_ids": ["any-run"], "objectives": [objective]},
            )
            assert response.status_code == 422
        finally:
            module.app.dependency_overrides.clear()

    def test_duplicate_run_ids_are_rejected(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            response = client.post(
                "/api/experiments/e1/score",
                json={
                    "run_ids": ["r1", "r1"],
                    "objectives": [_objective().model_dump(mode="json")],
                },
            )
            assert response.status_code == 422
        finally:
            module.app.dependency_overrides.clear()

    def test_a_missing_explanation_is_a_404(self, tmp_path: Path) -> None:
        client, module = self._client(tmp_path)
        try:
            response = client.get(f"/api/experiments/e1/scores/{'0' * 16}/explanation")
            assert response.status_code == 404
        finally:
            module.app.dependency_overrides.clear()
