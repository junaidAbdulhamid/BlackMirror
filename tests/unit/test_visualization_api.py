"""Visualization API: serialization, binary transport, and failure modes.

Uses a synthetic artifact tree so the suite stays fast and does not depend on
whichever runs happen to exist on a developer's machine.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from blackmirror.api.app import app, loader_dep  # noqa: E402
from blackmirror.api.loader import MeshAlignmentError, VisualizationLoader  # noqa: E402
from blackmirror.config.settings import Settings  # noqa: E402

PER_HEMISPHERE = 8
TOTAL = PER_HEMISPHERE * 2
N_TIME = 5


def _write_mesh(root: Path) -> None:
    """A tiny two-hemisphere surface with valid, fully-connected topology."""
    mesh = root / "mesh" / "fsaverage5"
    mesh.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    faces = np.array(
        [[i, (i + 1) % PER_HEMISPHERE, (i + 2) % PER_HEMISPHERE] for i in range(PER_HEMISPHERE)],
        dtype=np.int32,
    )
    for hemisphere in ("left", "right"):
        for surface in ("inflated", "pial"):
            np.save(
                mesh / f"{hemisphere}_{surface}_vertices.npy",
                rng.normal(size=(PER_HEMISPHERE, 3)).astype(np.float32),
            )
        np.save(mesh / f"{hemisphere}_faces.npy", faces)
    np.save(mesh / "medial_wall_mask.npy", np.zeros(TOTAL, dtype=bool))
    (mesh / "mapping.json").write_text(
        json.dumps({"surface_space": "fsaverage5", "vertices_per_hemisphere": PER_HEMISPHERE})
    )


def _write_run(root: Path, run_id: str, *, vertex_count: int = TOTAL) -> np.ndarray:
    from blackmirror.schemas.metadata import ModelMetadata, RunProvenance
    from blackmirror.schemas.prediction import (
        ArtifactPaths,
        CorticalMetadata,
        PerformanceMetrics,
        PredictionArrayMetadata,
        PredictionResult,
        PredictionValidation,
        TemporalMetadata,
    )
    from blackmirror.schemas.stimulus import MediaType, StimulusInput

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    predictions = rng.normal(size=(N_TIME, vertex_count)).astype(np.float32)
    np.save(run_dir / "predictions.npy", predictions)

    # Deliberately non-uniform: mirrors TRIBE dropping event-free segments.
    starts = np.array([0.0, 1.0, 2.0, 7.0, 8.0], dtype=np.float64)
    np.savez_compressed(
        run_dir / "temporal.npz",
        segment_start_seconds=starts,
        segment_duration_seconds=np.ones(N_TIME),
        segment_n_events=np.ones(N_TIME, dtype=np.int64),
        stimulus_time_seconds=starts,
        bold_acquisition_time_seconds=starts + 5.0,
    )

    media = root / "clip.mp4"
    media.write_bytes(b"\x00" * 128)

    result = PredictionResult(
        run_id=run_id,
        created_at=dt.datetime(2026, 9, 3, tzinfo=dt.UTC),
        stimulus=StimulusInput(
            path=media,
            filename="clip.mp4",
            media_type=MediaType.VIDEO,
            file_size_bytes=128,
            duration_seconds=9.0,
            sha256="a" * 64,
        ),
        model=ModelMetadata(
            name="TRIBE v2",
            backend="tribe_v2",
            model_id="facebook/tribev2",
            license="CC-BY-NC-4.0",
            loaded_device="cpu",
            dtype="float32",
            subject_conditioning="average_subjects",
            feature_extractors={"text": "meta-llama/Llama-3.2-3B"},
        ),
        prediction=PredictionArrayMetadata(
            shape=(N_TIME, vertex_count),
            dtype="float32",
            axis_names=("time", "vertex"),
            temporal_samples=N_TIME,
            cortical_features=vertex_count,
            semantics="Predicted cortical response.",
            artifact_path=Path("predictions.npy"),
        ),
        temporal=TemporalMetadata(
            n_time_points=N_TIME,
            tr_seconds=1.0,
            sampling_rate_hz=1.0,
            timeline_is_contiguous=False,
            segments_were_filtered=True,
            n_segments_total=100,
            n_segments_kept=N_TIME,
            hemodynamic_offset_seconds=5.0,
            hemodynamic_offset_verified=True,
            output_is_stimulus_aligned=True,
            arrays_artifact_path=Path("temporal.npz"),
        ),
        cortical=CorticalMetadata(
            surface_space="fsaverage5",
            vertex_count=vertex_count,
            vertices_per_hemisphere=vertex_count // 2,
            hemisphere_order=("left", "right"),
            hemisphere_index_ranges={
                "left": (0, vertex_count // 2),
                "right": (vertex_count // 2, vertex_count),
            },
        ),
        validation=PredictionValidation(
            valid=True,
            shape=(N_TIME, vertex_count),
            dtype="float32",
            nan_count=0,
            inf_count=0,
            finite_count=predictions.size,
            min=float(predictions.min()),
            max=float(predictions.max()),
            mean=float(predictions.mean()),
            std=float(predictions.std()),
            p01=float(np.percentile(predictions, 1)),
            p50=float(np.percentile(predictions, 50)),
            p99=float(np.percentile(predictions, 99)),
        ),
        performance=PerformanceMetrics(
            preprocessing_seconds=0.1,
            model_load_seconds=0.1,
            inference_seconds=0.1,
            postprocessing_seconds=0.1,
            validation_seconds=0.1,
            persistence_seconds=0.1,
            total_seconds=0.6,
        ),
        provenance=RunProvenance(
            run_id=run_id,
            blackmirror_version="0.1.0",
            python_version="3.12",
            platform="test",
            device="cpu",
            dtype="float32",
            random_seed=1,
            stimulus_sha256="a" * 64,
            model_fingerprint="fp",
            preprocessing_config={},
            cache_key="c" * 64,
        ),
        artifacts=ArtifactPaths(
            run_dir=run_dir,
            manifest=Path("manifest.json"),
            stimulus=Path("stimulus.json"),
            predictions=Path("predictions.npy"),
            prediction_metadata=Path("prediction_metadata.json"),
            temporal=Path("temporal.npz"),
            validation=Path("validation.json"),
            performance=Path("performance.json"),
            provenance=Path("provenance.json"),
        ),
    )
    (run_dir / "manifest.json").write_text(result.model_dump_json(indent=2))
    return predictions


@pytest.fixture
def predictions(tmp_path: Path) -> np.ndarray:
    """The exact array written to disk, so tests can assert byte fidelity."""
    _write_mesh(tmp_path)
    return _write_run(tmp_path, "run-ok")


@pytest.fixture
def client(tmp_path: Path, predictions: np.ndarray) -> TestClient:
    loader = VisualizationLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
    app.dependency_overrides[loader_dep] = lambda: loader
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- Metadata -------------------------------------------------------------


def test_health_scopes_inference_to_phase8_only() -> None:
    """Phase 8 made the blanket "no inference" claim false, so it states the scope.

    The claim matters: it is what tells an operator that this process can start
    a multi-hour TRIBE pass, and exactly which routes can do it.
    """
    body = TestClient(app).get("/api/health").json()

    assert body["runs_inference"] is True
    assert body["inference_scope"] == "phase8_resimulation_only"


def test_run_summary_serializes_the_contract(client: TestClient) -> None:
    body = client.get("/api/runs/run-ok").json()

    assert body["run_id"] == "run-ok"
    assert body["cortical"]["vertex_count"] == TOTAL
    assert body["cortical"]["hemispheres"] == [
        {"name": "left", "start": 0, "end": PER_HEMISPHERE},
        {"name": "right", "start": PER_HEMISPHERE, "end": TOTAL},
    ]
    assert body["temporal"]["segments_were_filtered"] is True
    assert body["temporal"]["n_segments_total"] == 100
    assert body["temporal"]["output_is_stimulus_aligned"] is True
    assert "not measurements" in body["interpretation_notice"].lower()


def test_summary_contains_no_bulk_arrays(client: TestClient) -> None:
    """Arrays travel as binary; JSON stays small and readable."""
    raw = client.get("/api/runs/run-ok").text
    assert len(raw) < 20_000

    def walk(node: object) -> None:
        if isinstance(node, list):
            assert len(node) < 64
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(json.loads(raw))


def test_scale_recommends_diverging_for_signed_data(client: TestClient) -> None:
    scale = client.get("/api/runs/run-ok").json()["scale"]
    assert scale["negative_fraction"] > 0.15
    assert scale["diverging_recommended"] is True
    assert scale["abs_max"] == pytest.approx(max(abs(scale["p01"]), abs(scale["p99"])))


def test_run_listing(client: TestClient) -> None:
    runs = client.get("/api/runs").json()
    assert [r["run_id"] for r in runs] == ["run-ok"]
    assert runs[0]["shape"] == [N_TIME, TOTAL]


# --- Binary transport -----------------------------------------------------


def test_predictions_are_served_as_raw_float32(
    client: TestClient, predictions: np.ndarray
) -> None:
    """Bytes must survive the round trip unmodified — this is raw scientific data."""
    response = client.get("/api/runs/run-ok/predictions")

    assert response.headers["X-Array-Dtype"] == "float32"
    assert response.headers["X-Array-Shape"] == f"{N_TIME},{TOTAL}"
    assert len(response.content) == N_TIME * TOTAL * 4

    array = np.frombuffer(response.content, dtype=np.float32).reshape(N_TIME, TOTAL)
    np.testing.assert_array_equal(array, predictions)


def test_timeline_uses_segment_times_not_index_times(client: TestClient) -> None:
    """The regression that would silently desynchronise playback."""
    times = np.frombuffer(client.get("/api/runs/run-ok/timeline").content, dtype=np.float32)
    np.testing.assert_allclose(times, [0.0, 1.0, 2.0, 7.0, 8.0])
    assert not np.allclose(times, np.arange(N_TIME))


def test_mesh_buffers_are_typed_correctly(client: TestClient) -> None:
    vertices = client.get("/api/runs/run-ok/mesh/left/inflated/vertices")
    assert vertices.headers["X-Array-Dtype"] == "float32"
    assert vertices.headers["X-Array-Shape"] == f"{PER_HEMISPHERE},3"

    faces = client.get("/api/runs/run-ok/mesh/left/faces")
    assert faces.headers["X-Array-Dtype"] == "int32"
    indices = np.frombuffer(faces.content, dtype=np.int32)
    assert indices.max() < PER_HEMISPHERE

    wall = client.get("/api/runs/run-ok/mesh/medial-wall")
    assert wall.headers["X-Array-Dtype"] == "uint8"
    assert len(wall.content) == TOTAL


def test_binary_headers_are_exposed_cross_origin(client: TestClient) -> None:
    """Without this the browser cannot read the dtype/shape it needs."""
    exposed = client.get("/api/runs/run-ok/predictions").headers[
        "Access-Control-Expose-Headers"
    ]
    assert "X-Array-Dtype" in exposed and "X-Array-Shape" in exposed


def test_stimulus_is_streamable(client: TestClient) -> None:
    response = client.get("/api/runs/run-ok/stimulus")
    assert response.status_code == 200
    assert response.headers["Accept-Ranges"] == "bytes"


# --- Failure modes --------------------------------------------------------


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404


def test_malformed_run_id_is_rejected(client: TestClient) -> None:
    assert client.get("/api/runs/..%2F..%2Fetc").status_code in (404, 422)


def test_misaligned_run_is_refused_not_rendered(tmp_path: Path) -> None:
    """Scientific correctness beats showing something.

    A prediction whose vertex count does not match the mesh must fail loudly.
    """
    _write_mesh(tmp_path)
    _write_run(tmp_path, "run-bad", vertex_count=TOTAL + 4)
    loader = VisualizationLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
    app.dependency_overrides[loader_dep] = lambda: loader
    try:
        response = TestClient(app).get("/api/runs/run-bad")
        assert response.status_code == 409
        assert "vertices" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_missing_mesh_is_reported_clearly(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-ok")  # no mesh written
    loader = VisualizationLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
    with pytest.raises(MeshAlignmentError, match="mesh"):
        loader.build_summary("run-ok")


# --- Surface-space consistency ------------------------------------------


def test_mesh_descriptors_use_the_runs_declared_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mesh URLs and descriptors must follow the run, not a hard-coded default.

    A run predicting onto a different surface would otherwise be described with
    fsaverage5 geometry — the wrong mesh, silently.
    """
    _write_mesh(tmp_path)
    _write_run(tmp_path, "run-ok")

    loader = VisualizationLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))

    seen: list[str] = []
    original = loader.load_mesh_array

    def spy(hemisphere: str, name: str, space: str = "fsaverage5"):  # type: ignore[no-untyped-def]
        seen.append(space)
        return original(hemisphere, name, space)

    monkeypatch.setattr(loader, "load_mesh_array", spy)
    summary = loader.build_summary("run-ok")

    assert seen, "no mesh arrays were loaded"
    assert set(seen) == {"fsaverage5"}
    assert summary.cortical.surface_space == "fsaverage5"
    # The declared space travels into the descriptions clients read.
    assert "fsaverage5" in summary.arrays["left_faces"].description


def test_missing_mesh_for_the_declared_space_is_refused(tmp_path: Path) -> None:
    """A run naming a space we have not exported must fail, not fall back."""
    _write_mesh(tmp_path)
    predictions = _write_run(tmp_path, "run-other-space")

    manifest_path = tmp_path / "runs" / "run-other-space" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cortical"]["surface_space"] = "fsaverage6"
    manifest_path.write_text(json.dumps(manifest))
    assert predictions.size

    loader = VisualizationLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
    with pytest.raises(MeshAlignmentError, match="fsaverage6"):
        loader.build_summary("run-other-space")


# --- Durations / coverage -----------------------------------------------


def test_durations_are_served(client: TestClient) -> None:
    response = client.get("/api/runs/run-ok/durations")
    assert response.headers["X-Array-Dtype"] == "float32"
    values = np.frombuffer(response.content, dtype=np.float32)
    np.testing.assert_allclose(values, np.ones(N_TIME))


def test_durations_are_described_in_the_summary(client: TestClient) -> None:
    descriptor = client.get("/api/runs/run-ok").json()["arrays"]["durations"]
    assert descriptor["shape"] == [N_TIME]
    assert "coverage" in descriptor["description"] or "covers" in descriptor["description"]


def test_durations_fall_back_to_the_tr_when_absent(tmp_path: Path) -> None:
    """A run without the array still yields usable coverage information."""
    _write_mesh(tmp_path)
    _write_run(tmp_path, "run-ok")
    (tmp_path / "runs" / "run-ok" / "temporal.npz").unlink()

    loader = VisualizationLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
    array = loader.load_durations("run-ok")
    values = np.frombuffer(array.data, dtype=np.float32)
    assert values.shape == (N_TIME,)
    np.testing.assert_allclose(values, 1.0)
