"""Step 57 and Steps 84-88: surrogate persistence and the HTTP surface over it.

The store deliberately writes a *recipe* rather than a pickled estimator, so
what these tests assert is that everything needed to rebuild a model survives a
round trip, and that a reader who was not there when the search ran can still
tell how far the model should be trusted.

The routes are read-only. Every number they serve is an estimate produced by a
model, never a measurement from the pipeline, and the tests check that the
distinction is still visible at the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

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
from blackmirror.search.space import ContentSearchSpace, continuous
from blackmirror.surrogate.encoder import GenomeFeatureEncoder
from blackmirror.surrogate.orchestrator import RoundRecord
from blackmirror.surrogate.schemas import (
    DatasetIdentity,
    SurrogateDataset,
    SurrogateTrainingRecord,
)
from blackmirror.surrogate.storage import SurrogateStore, persist
from blackmirror.surrogate.trainer import SurrogateTrainer, TrustPolicy

fastapi = pytest.importorskip("fastapi")


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),
            continuous("brightness", Op.BRIGHTNESS, -1.0, 1.0, resolution=0.01),
        )
    )


def _identity() -> DatasetIdentity:
    return DatasetIdentity(
        root_media_sha256="a" * 64,
        objective_definition_hash="b" * 64,
        search_space_version="1.0",
        materialization_version="1.0",
        scoring_version="1.0",
        model_fingerprint="tribe_v2|test",
    )


def _objective() -> NeuralObjective:
    return NeuralObjective(
        objective_id="o1",
        name="o1",
        metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE,
    )


def _surface(values: dict[str, float]) -> float:
    return -((values["gain"] - 3.0) ** 2) - 10.0 * (values["brightness"] - 0.4) ** 2


def _dataset(n: int = 12, *, seed: int = 0) -> SurrogateDataset:
    rng = Random(seed)
    space = _space()
    dataset = SurrogateDataset(identity=_identity())
    for index in range(n):
        values = space.sample(rng)
        target = _surface(values)
        dataset = dataset.with_record(
            SurrogateTrainingRecord(
                candidate_id=f"c{index:03d}",
                genome=values,
                objective_values={"o1": target},
                directional_target=target,
                primary_objective_id="o1",
                sequence=index,
            )
        )
    return dataset


def _rounds() -> list[RoundRecord]:
    return [
        RoundRecord(
            round_number=1,
            mode="bootstrap",
            candidate_id="c000",
            genome={"gain": 1.0, "brightness": 0.1},
            actual_score=-5.0,
            selection_reason="bootstrap: covering the space before the model is fitted",
        ),
        RoundRecord(
            round_number=2,
            mode="surrogate",
            candidate_id="c001",
            genome={"gain": 2.5, "brightness": 0.35},
            predicted_score=-1.2,
            predicted_uncertainty=0.4,
            actual_score=-0.9,
            prediction_error=0.3,
            acquisition="expected_improvement",
            acquisition_value=0.22,
            selection_reason="expected improvement over the incumbent",
        ),
    ]


def _persisted(tmp_path: Path, search_id: str = "search-1"):
    """A search's surrogate artifacts, written the way the loop writes them."""
    dataset = _dataset()
    space = _space()
    encoder = GenomeFeatureEncoder.from_space(space)
    trainer = SurrogateTrainer(encoder, policy=TrustPolicy(min_samples=8, min_spearman=0.2))
    trained = trainer.fit(dataset)
    store = persist(
        tmp_path,
        search_id,
        dataset=dataset,
        encoder=encoder,
        trained=trained,
        rounds=_rounds(),
        report={"headline": "best observed", "search_id": search_id},
    )
    return store, dataset, encoder, trained


class TestStoreRoundTrip:
    def test_everything_written_can_be_read_back(self, tmp_path: Path) -> None:
        store, dataset, encoder, trained = _persisted(tmp_path)

        assert store.exists
        reloaded = store.read_dataset()
        assert reloaded is not None
        assert reloaded.size == dataset.size
        # The hash is the contract: identical observations, identical dataset.
        assert reloaded.dataset_hash() == dataset.dataset_hash()

        back = store.read_encoder()
        assert back is not None
        assert back.schema_fingerprint() == encoder.schema_fingerprint()

        assert len(store.read_rounds()) == 2
        assert store.read_report() == {"headline": "best observed", "search_id": "search-1"}
        latest = store.latest_model()
        assert latest is not None
        assert latest.model_version == trained.metadata.model_version

    def test_the_recipe_is_enough_to_rebuild_the_same_model(self, tmp_path: Path) -> None:
        """Step 82. Reproducible beats reloadable: no pickle is involved."""
        store, _, _, trained = _persisted(tmp_path)

        dataset = store.read_dataset()
        encoder = store.read_encoder()
        assert dataset is not None and encoder is not None

        rebuilt = SurrogateTrainer(
            encoder, policy=TrustPolicy(min_samples=8, min_spearman=0.2)
        ).fit(dataset)

        assert rebuilt.metadata.training_dataset_hash == trained.metadata.training_dataset_hash
        assert rebuilt.trusted == trained.trusted
        assert rebuilt.prefix_validation.spearman == trained.prefix_validation.spearman

    def test_no_pickled_estimator_is_written(self, tmp_path: Path) -> None:
        store, _, _, _ = _persisted(tmp_path)
        written = [path for path in store.root.rglob("*") if path.is_file()]

        assert written, "nothing was persisted"
        assert all(path.suffix == ".json" for path in written), (
            "a pickled estimator is tied to the scikit-learn version that wrote it"
        )

    def test_model_versions_are_ordered_by_version_not_by_write_time(
        self, tmp_path: Path
    ) -> None:
        store, _, _, trained = _persisted(tmp_path)
        for version in (12, 3):
            store.write_model(trained.metadata.model_copy(update={"model_version": version}))

        # Zero-padded filenames, so 3 sorts before 12 rather than lexically after.
        assert [item.model_version for item in store.read_models()] == [1, 3, 12]
        assert store.latest_model() is not None
        assert store.latest_model().model_version == 12  # type: ignore[union-attr]

    def test_a_half_written_file_is_never_left_behind(self, tmp_path: Path) -> None:
        """The write is atomic, so a crash mid-write cannot corrupt the artifact."""
        store = SurrogateStore(tmp_path, "search-1")
        store.write_dataset(_dataset())
        good = store.read_dataset()

        with pytest.raises(RuntimeError):
            store.write_report({"boom": _Explodes()})

        assert store.read_dataset() is not None
        assert store.read_dataset().dataset_hash() == good.dataset_hash()  # type: ignore[union-attr]
        assert not list(store.root.glob(".*tmp")), "a temporary file survived the failure"

    def test_unreadable_artifacts_are_reported_as_absent_not_as_empty(
        self, tmp_path: Path
    ) -> None:
        store, _, _, _ = _persisted(tmp_path)
        (store.root / "dataset" / "records.json").write_text("{not json", encoding="utf-8")

        # None means "cannot say", which is different from a dataset of size zero.
        assert store.read_dataset() is None

    @pytest.mark.parametrize("bad", ["..", ".", "../etc", "a/b", "", "x" * 129])
    def test_a_search_id_cannot_escape_the_artifact_root(
        self, tmp_path: Path, bad: str
    ) -> None:
        with pytest.raises(ValueError, match="invalid search id"):
            SurrogateStore(tmp_path, bad)


class _Explodes:
    def __repr__(self) -> str:
        raise RuntimeError("cannot serialize")


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from blackmirror.api import app as app_module

    class _Settings:
        artifact_dir = tmp_path

    class _Loader:
        settings = _Settings()

    monkeypatch.setattr(app_module, "get_loader", lambda: _Loader())
    return TestClient(app_module.app), tmp_path


class TestSurrogateRoutes:
    def test_the_summary_reports_trust_alongside_the_model(self, api) -> None:
        client, tmp_path = api
        _persisted(tmp_path)

        body = client.get("/api/search/search-1/surrogate").json()

        assert body["search_id"] == "search-1"
        assert body["training_samples"] == 12
        assert body["model_versions"] == 1
        assert body["model"]["model_type"] == "gaussian_process"
        # Whether the model may choose is served next to the model itself, so a
        # reader cannot pick up the prediction without the caveat attached.
        assert "trusted" in body and "trust_reason" in body
        assert body["target_spread"] is not None

    def test_metrics_expose_both_validation_schemes(self, api) -> None:
        client, tmp_path = api
        _persisted(tmp_path)

        body = client.get("/api/search/search-1/surrogate/metrics").json()

        assert "cross_validation" in body and "prefix_validation" in body
        assert body["state"]
        assert "trust_reason" in body

    def test_the_dataset_route_serves_only_real_observations(self, api) -> None:
        client, tmp_path = api
        _persisted(tmp_path)

        body = client.get("/api/search/search-1/surrogate/dataset").json()

        assert len(body["records"]) == 12
        assert body["identity"]["model_fingerprint"] == "tribe_v2|test"

    def test_every_model_version_is_listed(self, api) -> None:
        client, tmp_path = api
        _persisted(tmp_path)

        body = client.get("/api/search/search-1/surrogate/models").json()

        assert [item["model_version"] for item in body] == [1]

    def test_acquisition_rounds_pair_predicted_with_actual(self, api) -> None:
        client, tmp_path = api
        _persisted(tmp_path)

        body = client.get("/api/search/search-1/acquisition").json()

        assert [item["round_number"] for item in body] == [1, 2]
        bootstrap, chosen = body
        # A bootstrap round has no prediction to be right or wrong about.
        assert bootstrap["predicted_score"] is None
        assert chosen["predicted_score"] == -1.2
        assert chosen["actual_score"] == -0.9
        assert chosen["prediction_error"] == pytest.approx(0.3)

    def test_the_report_route_serves_what_was_written(self, api) -> None:
        client, tmp_path = api
        _persisted(tmp_path)

        assert client.get("/api/search/search-1/surrogate/report").json() == {
            "headline": "best observed",
            "search_id": "search-1",
        }

    def test_a_search_with_no_surrogate_is_404_with_an_explanation(self, api) -> None:
        client, _ = api

        response = client.get("/api/search/nonexistent/surrogate")

        assert response.status_code == 404
        # The absence is expected early in a search, so it says so rather than
        # reading as a failure.
        assert "optimization loop" in response.json()["detail"]

    def test_missing_diagnostics_are_404_rather_than_an_empty_object(self, api) -> None:
        client, tmp_path = api
        store, _, _, _ = _persisted(tmp_path, "search-2")
        (store.root / "diagnostics" / "metrics.json").unlink()

        response = client.get("/api/search/search-2/surrogate/metrics")

        assert response.status_code == 404
        assert "no diagnostics yet" in response.json()["detail"]

    def test_a_missing_report_is_404_rather_than_a_fabricated_one(self, api) -> None:
        client, tmp_path = api
        store, _, _, _ = _persisted(tmp_path, "search-3")
        (store.root / "report.json").unlink()

        response = client.get("/api/search/search-3/surrogate/report")

        assert response.status_code == 404

    def test_a_traversing_search_id_is_refused(self, api) -> None:
        client, tmp_path = api
        _persisted(tmp_path)
        secret = tmp_path / "secret.json"
        secret.write_text(json.dumps({"private": True}), encoding="utf-8")

        response = client.get("/api/search/..%2F..%2Fsecret/surrogate")

        assert response.status_code in {404, 422}
        assert "private" not in response.text
