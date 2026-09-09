"""The Phase 8 request builder and the HTTP surface over the state machine.

These are the two pieces that decide whether Phase 8 is reachable at all. The
state machine was already covered; what was not covered was that a human can
produce a request it accepts, and that starting one does not block a web worker
for the hours a TRIBE pass takes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blackmirror.optimization.engine import OptimizationOrchestrator, content_series_from_arrays
from blackmirror.optimization.schemas import (
    ApprovalState,
    ExpectedDirection,
    OptimizationRequest,
    RecommendationReview,
)
from blackmirror.optimization.storage import OptimizationStore, build_spec
from blackmirror.resimulation.request_builder import RequestBuildError, build_request
from blackmirror.resimulation.schemas import ResimulationResult, RunStatus, Stage, StopReason
from blackmirror.resimulation.storage import ResimulationStore
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
from blackmirror.scoring.storage import ScoringStore

fastapi = pytest.importorskip("fastapi")

OPTIMIZATION_KEY = "abc12345"


# ---------------------------------------------------------------------------
# A real Phase 6 score and Phase 7 optimization on disk, not stand-ins.
# The builder's whole job is reading those records, so faking them would test
# nothing.
# ---------------------------------------------------------------------------


def _variant(variant_id: str, values: list[float]) -> ScoringVariant:
    times = np.arange(len(values), dtype=np.float64)
    roi = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return ScoringVariant(
        variant_id=variant_id,
        responses=np.tile(roi, (1, 4)),
        times=times,
        roi_timeseries=roi,
        region_names=("r0",),
        medial_wall_mask=np.zeros(4, dtype=bool),
        duration_seconds=float(len(values)),
        hemisphere_ranges={"left": (0, 2), "right": (2, 4)},
    )


def _objective(objective_id: str, direction: ObjectiveDirection) -> NeuralObjective:
    return NeuralObjective(
        objective_id=objective_id,
        name=objective_id,
        metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.ROI, region_id=0),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction,
    )


def _series(variant_id: str, motion: float, speech: float):
    names = ("motion", "speech_present", "audio_energy", "visual_available")
    times = np.arange(20, dtype=np.float64)
    matrix = np.column_stack(
        [[motion] * 20, [speech] * 20, [0.5] * 20, [1.0] * 20]
    ).astype(np.float64)
    return content_series_from_arrays(variant_id, times, matrix, names)


def _seed(
    root: Path,
    *,
    objectives: tuple[NeuralObjective, ...] | None = None,
    reference_values: list[float] | None = None,
) -> tuple[str, str]:
    """Write a real score and optimization, approve one recommendation.

    `reference_values` exists because "better" is direction-dependent: the
    reference variant has to actually out-score the source under the declared
    objective, or Phase 7 correctly produces nothing to approve.

    Returns the stored objective-set hash and the approved candidate id.
    """
    declared = objectives or (_objective("o1", ObjectiveDirection.MAXIMIZE),)
    score = GoalConditionedScoringEngine().score_experiment(
        "exp",
        (
            _variant("A", [0.0] * 10 + [1.0] * 8 + [6.0] * 2),
            _variant("B", reference_values or [2.0] * 20),
        ),
        declared,
    )
    ScoringStore(root).write(score)
    request = OptimizationRequest(
        experiment_id="exp",
        source_variant_id="A",
        objective_set_hash=score.metadata.objective_set_hash,
    )
    result = OptimizationOrchestrator().optimize(
        request,
        score,
        content={"A": _series("A", 0.2, 0.0), "B": _series("B", 0.7, 1.0)},
        events={
            "A": [{"event_type": "scene", "start_time": 0.0, "end_time": 20.0}],
            "B": [{"event_type": "speech_segment", "start_time": 0.0, "end_time": 9.0}],
        },
    )
    store = OptimizationStore(root)
    store.write(result, OPTIMIZATION_KEY)
    assert result.recommendations, "fixture needs at least one recommendation to approve"
    reviews = [
        RecommendationReview(
            recommendation_id=result.recommendations[0].recommendation_id,
            state=ApprovalState.APPROVED,
            reason="fixture approval",
        )
    ]
    for review in reviews:
        store.record_review("exp", OPTIMIZATION_KEY, review)
    spec = build_spec(result, reviews, proposed_variant_id="cand1")
    store.write_spec("exp", OPTIMIZATION_KEY, spec)
    return score.metadata.objective_set_hash, spec.proposed_variant_id


def _media(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.mp4"
    variant = tmp_path / "variant.mp4"
    source.write_bytes(b"source bytes")
    variant.write_bytes(b"variant bytes")
    return source, variant


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------


class TestRequestBuilder:
    def test_derives_objectives_hashes_and_maps_from_stored_records(
        self, tmp_path: Path
    ) -> None:
        """The parts a human cannot type are read, not supplied."""
        objective_set_hash, candidate = _seed(tmp_path)
        source, variant = _media(tmp_path)

        request = build_request(
            tmp_path,
            resimulation_id="loop-1",
            experiment_id="exp",
            optimization_request_key=OPTIMIZATION_KEY,
            proposed_variant_id=candidate,
            variant_media=variant,
            source_media=source,
        )

        assert request.objective_set_hash == objective_set_hash
        assert [item.objective_id for item in request.objectives] == ["o1"]
        assert request.parent_run_id == "A"
        assert request.binding.variant_sha256 != request.binding.source_sha256
        # Every hypothesis is mapped; none was left to a silent default.
        assert set(request.hypothesis_objective_ids) == set(
            request.proposed_variant.hypothesis_ids
        )
        assert set(request.hypothesis_expected_directions) == set(
            request.proposed_variant.hypothesis_ids
        )

    def test_direction_map_follows_the_objective_not_the_wish(self, tmp_path: Path) -> None:
        """A minimize objective must produce a test-for-decrease hypothesis.

        The request schema refuses a contradiction between the two. Deriving the
        direction from the stored objective is what keeps that check from being
        something a caller has to satisfy by trial and error.
        """
        _seed(
            tmp_path,
            objectives=(_objective("o1", ObjectiveDirection.MINIMIZE),),
            reference_values=[0.2] * 20,
        )
        source, variant = _media(tmp_path)

        request = build_request(
            tmp_path,
            resimulation_id="loop-1",
            experiment_id="exp",
            optimization_request_key=OPTIMIZATION_KEY,
            proposed_variant_id="cand1",
            variant_media=variant,
            source_media=source,
        )

        assert set(request.hypothesis_expected_directions.values()) == {
            ExpectedDirection.TEST_FOR_DECREASE
        }

    def test_resolves_source_media_from_the_parent_run_manifest(self, tmp_path: Path) -> None:
        """The comparison anchors to the bytes that produced the parent run."""
        _seed(tmp_path)
        source, variant = _media(tmp_path)
        manifest = tmp_path / "runs" / "A" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"stimulus": {"path": str(source)}}), encoding="utf-8")

        request = build_request(
            tmp_path,
            resimulation_id="loop-1",
            experiment_id="exp",
            optimization_request_key=OPTIMIZATION_KEY,
            proposed_variant_id="cand1",
            variant_media=variant,
        )

        assert request.binding.source_path == str(source.resolve())

    def test_missing_parent_manifest_names_the_fix(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        _, variant = _media(tmp_path)

        with pytest.raises(RequestBuildError, match="pass source_media explicitly"):
            build_request(
                tmp_path,
                resimulation_id="loop-1",
                experiment_id="exp",
                optimization_request_key=OPTIMIZATION_KEY,
                proposed_variant_id="cand1",
                variant_media=variant,
            )

    def test_unapproved_candidate_is_refused_with_what_exists(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        source, variant = _media(tmp_path)

        with pytest.raises(RequestBuildError, match="approved review"):
            build_request(
                tmp_path,
                resimulation_id="loop-1",
                experiment_id="exp",
                optimization_request_key=OPTIMIZATION_KEY,
                proposed_variant_id="never-approved",
                variant_media=variant,
                source_media=source,
            )

    def test_missing_optimization_says_to_run_phase_7(self, tmp_path: Path) -> None:
        _, variant = _media(tmp_path)

        with pytest.raises(RequestBuildError, match="run Phase 7 first"):
            build_request(
                tmp_path,
                resimulation_id="loop-1",
                experiment_id="exp",
                optimization_request_key=OPTIMIZATION_KEY,
                proposed_variant_id="cand1",
                variant_media=variant,
                source_media=variant,
            )

    def test_identical_media_is_refused(self, tmp_path: Path) -> None:
        """Re-simulating a file against itself would measure only model noise."""
        _seed(tmp_path)
        source, _ = _media(tmp_path)
        twin = tmp_path / "twin.mp4"
        twin.write_bytes(source.read_bytes())

        with pytest.raises(ValueError, match="content hashes must differ"):
            build_request(
                tmp_path,
                resimulation_id="loop-1",
                experiment_id="exp",
                optimization_request_key=OPTIMIZATION_KEY,
                proposed_variant_id="cand1",
                variant_media=twin,
                source_media=source,
            )


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path: Path):
    from fastapi.testclient import TestClient

    from blackmirror.api.app import app, resimulation_loader_dep
    from blackmirror.api.resimulation_loader import ResimulationLoader

    loader = ResimulationLoader(tmp_path, inference=object())  # type: ignore[arg-type]
    app.dependency_overrides[resimulation_loader_dep] = lambda: loader
    yield TestClient(app), loader, tmp_path
    app.dependency_overrides.clear()


def _bound_state(root: Path, tmp_path: Path) -> ResimulationResult:
    _seed(root)
    source, variant = _media(tmp_path)
    request = build_request(
        root,
        resimulation_id="loop-1",
        experiment_id="exp",
        optimization_request_key=OPTIMIZATION_KEY,
        proposed_variant_id="cand1",
        variant_media=variant,
        source_media=source,
    )
    from blackmirror.resimulation.orchestrator import _cache_key

    state = ResimulationResult(request=request, cache_key=_cache_key(request))
    ResimulationStore(root).checkpoint(state)
    return state


class TestResimulationRoutes:
    def test_start_returns_the_bound_state_without_waiting_for_the_pipeline(
        self, api, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pass takes hours; the request must not.

        The route is allowed to queue work, so this asserts the response
        describes the checkpoint on disk (stage BOUND, nothing measured) rather
        than a finished run.
        """
        client, loader, root = api
        _seed(root)
        source, variant = _media(root)
        submitted: list[str] = []
        monkeypatch.setattr(
            loader, "_submit", lambda resimulation_id, work: submitted.append(resimulation_id)
        )

        response = client.post(
            "/api/experiments/exp/resimulations",
            json={
                "resimulation_id": "loop-1",
                "optimization_request_key": OPTIMIZATION_KEY,
                "proposed_variant_id": "cand1",
                "variant_media_path": str(variant),
                "source_media_path": str(source),
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["state"]["current_stage"] == int(Stage.BOUND)
        assert body["state"]["status"] == RunStatus.ACTIVE.value
        assert body["state"]["objective_deltas"] == []
        assert submitted == ["loop-1"]
        # And it is durable, not merely returned.
        assert ResimulationStore(root).exists("loop-1")

    def test_start_without_an_approved_candidate_is_a_conflict(self, api) -> None:
        client, _loader, root = api
        _seed(root)
        source, variant = _media(root)

        response = client.post(
            "/api/experiments/exp/resimulations",
            json={
                "resimulation_id": "loop-1",
                "optimization_request_key": OPTIMIZATION_KEY,
                "proposed_variant_id": "not-approved",
                "variant_media_path": str(variant),
                "source_media_path": str(source),
            },
        )

        assert response.status_code == 409
        assert "approved review" in response.json()["detail"]

    def test_missing_media_is_reported_as_not_found(self, api) -> None:
        client, _loader, root = api
        _seed(root)

        response = client.post(
            "/api/experiments/exp/resimulations",
            json={
                "resimulation_id": "loop-1",
                "optimization_request_key": OPTIMIZATION_KEY,
                "proposed_variant_id": "cand1",
                "variant_media_path": str(root / "absent.mp4"),
                "source_media_path": str(root / "absent.mp4"),
            },
        )

        assert response.status_code == 404

    def test_get_and_list_report_worker_liveness(self, api, tmp_path: Path) -> None:
        """`active` with no worker is the state that needs a resume, so it shows."""
        client, _loader, root = api
        _bound_state(root, tmp_path)

        one = client.get("/api/resimulations/loop-1").json()
        listed = client.get("/api/resimulations").json()

        assert one["state"]["request"]["resimulation_id"] == "loop-1"
        assert one["worker_alive"] is False
        assert [item["state"]["request"]["resimulation_id"] for item in listed] == ["loop-1"]

    def test_list_filters_by_experiment(self, api, tmp_path: Path) -> None:
        client, _loader, root = api
        _bound_state(root, tmp_path)

        assert client.get("/api/resimulations", params={"experiment_id": "exp"}).json()
        assert client.get("/api/resimulations", params={"experiment_id": "other"}).json() == []

    def test_unknown_resimulation_is_not_found(self, api) -> None:
        client, _loader, _root = api

        assert client.get("/api/resimulations/absent").status_code == 404

    def test_stop_records_a_durable_request_observed_between_stages(
        self, api, tmp_path: Path
    ) -> None:
        """Stopping is a request, not a kill: a half-written stage is worse."""
        client, _loader, root = api
        _bound_state(root, tmp_path)

        response = client.post("/api/resimulations/loop-1/stop", json={"reason": "cancelled"})

        assert response.status_code == 200
        assert response.json()["state"]["requested_stop_reason"] == StopReason.CANCELLED.value
        assert ResimulationStore(root).stop_requested("loop-1") is StopReason.CANCELLED

    def test_stop_rejects_a_reason_the_state_machine_reserves_for_itself(
        self, api, tmp_path: Path
    ) -> None:
        client, _loader, root = api
        _bound_state(root, tmp_path)

        response = client.post("/api/resimulations/loop-1/stop", json={"reason": "completed"})

        assert response.status_code == 422

    def test_resume_queues_work_and_reports_current_state(
        self, api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, loader, root = api
        _bound_state(root, tmp_path)
        submitted: list[str] = []
        monkeypatch.setattr(
            loader, "_submit", lambda resimulation_id, work: submitted.append(resimulation_id)
        )

        response = client.post("/api/resimulations/loop-1/resume")

        assert response.status_code == 200
        assert submitted == ["loop-1"]
        assert response.json()["state"]["current_stage"] == int(Stage.BOUND)


class TestLoaderConcurrency:
    def test_a_second_start_while_one_is_running_is_refused(self, tmp_path: Path) -> None:
        """One machine, one TRIBE pass. A queued duplicate would only swap."""
        from blackmirror.api.resimulation_loader import ResimulationLoader, ResimulationLoadError

        loader = ResimulationLoader(tmp_path, inference=object())  # type: ignore[arg-type]

        class Pending:
            def done(self) -> bool:
                return False

        loader._jobs["loop-1"] = Pending()  # type: ignore[assignment]

        with pytest.raises(ResimulationLoadError, match="already running"):
            loader._submit("loop-1", lambda orchestrator: None)  # type: ignore[arg-type,return-value]

    def test_worker_liveness_is_derived_not_stored(self, tmp_path: Path) -> None:
        """A dead worker leaves `active` in the file; only the lock disproves it."""
        store = ResimulationStore(tmp_path)

        assert store.active_owner("loop-1") is None


class TestConflictReporting:
    def test_rebinding_an_id_to_different_media_is_a_conflict_not_a_bad_request(
        self, api, tmp_path: Path
    ) -> None:
        """The request is well formed; it disagrees with what is already stored.

        The distinction matters to a caller: 422 says "fix your request", 409
        says "that id already means something else". Silently repointing the id
        would leave finished artifacts attributed to a request that never
        produced them.
        """
        client, _loader, root = api
        _bound_state(root, tmp_path)
        other = tmp_path / "other.mp4"
        other.write_bytes(b"a third file entirely")

        response = client.post(
            "/api/experiments/exp/resimulations",
            json={
                "resimulation_id": "loop-1",
                "optimization_request_key": OPTIMIZATION_KEY,
                "proposed_variant_id": "cand1",
                "variant_media_path": str(other),
                "source_media_path": str(tmp_path / "source.mp4"),
            },
        )

        assert response.status_code == 409
        assert "already bound" in response.json()["detail"]

    def test_binding_a_run_never_constructs_an_inference_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reads and binds must not pay for a model backend they never use.

        Resolving a device and building a backend on every stop or bind would
        undo the point of deferring it, and the cost is invisible until a
        machine without a working backend cannot even list its runs.
        """
        from blackmirror.api.resimulation_loader import ResimulationLoader

        built: list[str] = []

        def explode() -> object:
            built.append("constructed")
            raise AssertionError("inference service must not be built for a bind")

        loader = ResimulationLoader(tmp_path)
        monkeypatch.setattr(loader, "_service", explode)
        _seed(tmp_path)
        source, variant = _media(tmp_path)
        request = build_request(
            tmp_path,
            resimulation_id="loop-1",
            experiment_id="exp",
            optimization_request_key=OPTIMIZATION_KEY,
            proposed_variant_id="cand1",
            variant_media=variant,
            source_media=source,
        )

        state = loader.create(request, execute=False)
        loader.stop(state.request.resimulation_id, StopReason.CANCELLED)

        assert built == []
