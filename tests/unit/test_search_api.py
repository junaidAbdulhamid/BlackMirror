"""Step 75: the HTTP surface over a search.

Starting a search is queued rather than awaited, so these tests drive the loop
to completion first and then assert what the routes serve. What is under test is
the contract, not the loop, which has its own suite.
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
from blackmirror.search.budget import SearchBudget
from blackmirror.search.fitness import CandidateFitness, ComputeCost, FitnessStatus
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.orchestrator import NeuralSearchOrchestrator
from blackmirror.search.schemas import SearchConfig, SearchExperiment
from blackmirror.search.space import ContentSearchSpace, continuous
from blackmirror.search.strategy import RandomSearch

fastapi = pytest.importorskip("fastapi")


def _objective(oid: str = "o1") -> NeuralObjective:
    return NeuralObjective(
        objective_id=oid, name=oid, metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE,
    )


def _experiment(objectives: tuple[NeuralObjective, ...] = (), **overrides: object):
    declared = objectives or (_objective(),)
    fields: dict[str, object] = {
        "search_id": "s1",
        "experiment_id": "exp",
        "root_run_id": "root",
        "root_media_path": "/media/root.mp4",
        "root_media_sha256": "a" * 64,
        "objective_set_hash": "b" * 64,
        "objective_definition_hash": "c" * 64,
        "objectives": declared,
        "primary_objective_id": declared[0].objective_id if len(declared) > 1 else None,
        "space": ContentSearchSpace(
            parameters=(continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),)
        ),
        "config": SearchConfig(random_seed=3, candidate_batch_size=2, patience=100),
        "budget": SearchBudget(max_candidates=4, max_generations=10),
    }
    fields.update(overrides)
    return SearchExperiment(**fields)  # type: ignore[arg-type]


class StubEvaluator:
    def __init__(self, *, extra: dict[str, float] | None = None) -> None:
        self.extra = extra or {}

    def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
        value = -((genome.values["gain"] - 3.0) ** 2)
        return CandidateFitness(
            candidate_id=genome.genome_id, genome=genome,
            status=FitnessStatus.EVALUATED, scalar_fitness=value,
            primary_objective_id="o1",
            raw_objectives={"o1": value, **self.extra},
            candidate_run_id=f"run-{genome.genome_id}",
            compute=ComputeCost(wall_seconds=1.0, tribe_runs=1),
        )


@pytest.fixture
def api(tmp_path: Path):
    from fastapi.testclient import TestClient

    from blackmirror.api.app import app, search_loader_dep
    from blackmirror.api.search_loader import SearchLoader

    loader = SearchLoader(tmp_path)
    app.dependency_overrides[search_loader_dep] = lambda: loader
    yield TestClient(app), loader, tmp_path
    app.dependency_overrides.clear()


def _completed(tmp_path: Path, loader, **overrides: object) -> NeuralSearchOrchestrator:
    """Run a search to completion so its artifacts exist."""
    experiment = _experiment(**overrides)
    orchestrator = NeuralSearchOrchestrator(
        experiment, RandomSearch(), StubEvaluator(),  # type: ignore[arg-type]
        artifact_root=tmp_path, root_fitness=-9.0,
    )
    loader.start(orchestrator, execute=False)
    orchestrator.run()
    return orchestrator


class TestReads:
    def test_lists_recorded_searches(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search").json()

        assert [item["search_id"] for item in body] == ["s1"]
        assert body[0]["status"] == "completed"
        assert body[0]["strategy"] == "random_search"

    def test_serves_one_search_summary(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search/s1").json()

        assert body["best_candidate_id"]
        assert body["metrics"]["evaluations"] == 4

    def test_serves_the_best_candidate(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search/s1/best").json()

        assert body["candidate_id"]
        assert "fitness" in body

    def test_serves_the_trajectory_for_the_optimization_plot(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search/s1/trajectory").json()

        assert len(body) == 4
        assert [point["best_fitness"] for point in body] == sorted(
            point["best_fitness"] for point in body
        )

    def test_serves_the_budget_and_ledger(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search/s1/budget").json()

        assert body["budget"]["max_candidates"] == 4
        assert body["ledger"]["candidates_evaluated"] == 4

    def test_serves_the_event_history(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search/s1/events").json()

        types = {event["type"] for event in body}
        assert "search_started" in types
        assert "search_completed" in types

    def test_serves_the_population_per_generation(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search/s1/population").json()

        assert body
        assert "survivors" in body[0]
        assert "eliminated" in body[0]

    def test_serves_the_final_report(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.get("/api/search/s1/report").json()

        assert body["best_observed"]
        assert body["caveats"]
        assert "lineage" in body

    def test_an_unknown_search_is_not_found(self, api) -> None:
        client, _loader, _tmp = api

        assert client.get("/api/search/absent").status_code == 404


class TestPareto:
    def test_a_single_objective_search_has_no_frontier(self, api) -> None:
        """404 with an explanation, not an empty object implying a trade-off."""
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        response = client.get("/api/search/s1/pareto")

        assert response.status_code == 404
        assert "single objective" in response.json()["detail"]

    def test_a_multi_objective_search_serves_its_frontier(self, api) -> None:
        client, loader, tmp_path = api
        experiment = _experiment(objectives=(_objective("o1"), _objective("o2")))
        orchestrator = NeuralSearchOrchestrator(
            experiment, RandomSearch(), StubEvaluator(extra={"o2": 0.5}),  # type: ignore[arg-type]
            artifact_root=tmp_path, root_fitness=-9.0,
        )
        loader.start(orchestrator, execute=False)
        orchestrator.run()

        body = client.get("/api/search/s1/pareto").json()

        assert body["objective_ids"] == ["o1", "o2"]
        assert body["non_dominated"]


class TestControl:
    def test_stop_is_refused_for_a_search_not_running_here(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)
        loader._running.clear()

        response = client.post("/api/search/s1/stop")

        assert response.status_code == 409

    def test_pause_and_resume_toggle_the_flag(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        client.post("/api/search/s1/pause")
        assert loader._overrides["s1"].paused is True

        client.post("/api/search/s1/resume")
        assert loader._overrides["s1"].paused is False

    def test_pinning_a_candidate_is_recorded(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.post("/api/search/s1/pin", json={"candidate_id": "c0002"}).json()

        assert body["pinned"] == ["c0002"]

    def test_eliminating_a_candidate_is_recorded(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        body = client.post(
            "/api/search/s1/eliminate", json={"candidate_id": "c0003"}
        ).json()

        assert body["eliminated"] == ["c0003"]

    def test_a_control_route_needs_a_candidate_id(self, api) -> None:
        client, loader, tmp_path = api
        _completed(tmp_path, loader)

        assert client.post("/api/search/s1/pin", json={}).status_code == 422

    def test_starting_a_running_search_again_is_refused(self, api, tmp_path) -> None:
        _client, loader, _tmp = api
        orchestrator = NeuralSearchOrchestrator(
            _experiment(), RandomSearch(), StubEvaluator(),  # type: ignore[arg-type]
            artifact_root=tmp_path, root_fitness=-9.0,
        )
        loader.start(orchestrator, execute=False)

        class Pending:
            def done(self) -> bool:
                return False

        loader._jobs["s1"] = Pending()  # type: ignore[assignment]

        from blackmirror.api.search_loader import SearchLoadError

        with pytest.raises(SearchLoadError, match="already running"):
            loader.start(orchestrator)


class TestArtifactLayout:
    def test_the_full_layout_is_written(self, api) -> None:
        """Step 76. Each file a reader may want to open on its own."""
        _client, loader, tmp_path = api
        _completed(tmp_path, loader)
        directory = tmp_path / "search" / "s1"

        for name in (
            "config.json", "search_space.json", "budget.json", "state.json",
            "trajectory.json", "events.json", "final_report.json",
        ):
            assert (directory / name).is_file(), name
        assert list((directory / "evaluations").glob("*.json"))
        assert list((directory / "population").glob("generation-*.json"))

    def test_a_pareto_file_is_written_only_when_it_means_something(self, api) -> None:
        _client, loader, tmp_path = api
        _completed(tmp_path, loader)

        assert not (tmp_path / "search" / "s1" / "pareto_front.json").exists()

    def test_the_store_refuses_an_unsafe_artifact_name(self, tmp_path: Path) -> None:
        from blackmirror.search.storage import SearchStore

        with pytest.raises(ValueError, match="unsafe artifact name"):
            SearchStore(tmp_path).write_named("s1", "../escape.json", {})

    def test_the_store_refuses_a_path_traversing_search_id(self, tmp_path: Path) -> None:
        from blackmirror.search.storage import SearchStore

        with pytest.raises(ValueError, match="invalid search id"):
            SearchStore(tmp_path).directory("../../etc")
