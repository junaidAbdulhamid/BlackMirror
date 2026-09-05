"""Content analysis API: serialization, 404 behaviour, and safety."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from blackmirror.api.app import app, content_loader_dep  # noqa: E402
from blackmirror.api.content_loader import ContentLoader  # noqa: E402
from blackmirror.config.settings import Settings  # noqa: E402
from blackmirror.content.schemas import (  # noqa: E402
    AnalysisSource,
    ContentAnalysisMetadata,
    ContentAnalysisResult,
    ContentArrayArtifact,
    ContentEvent,
    ContentEventType,
    ContentMetrics,
    NeuralContentAssociation,
    Provenance,
    ShotSegment,
    TranscriptSegment,
    VideoMetadata,
)
from blackmirror.content.storage import ContentStore  # noqa: E402

ASR = Provenance(source=AnalysisSource.TRIBE_EVENTS)


def _write(tmp_path: Path, run_id: str = "run-1") -> ContentAnalysisResult:
    result = ContentAnalysisResult(
        run_id=run_id,
        stimulus_id="s1",
        stimulus_filename="clip.mp4",
        media=VideoMetadata(duration_seconds=10.0, width=640, height=360, fps=24.0),
        modalities_analysed=("visual", "speech"),
        shots=(ShotSegment(index=0, start_time=0, end_time=10, duration=10),),
        transcript=(
            TranscriptSegment(
                index=0, start_time=1, end_time=4, text="Keep going", provenance=ASR
            ),
        ),
        events=(
            ContentEvent(
                event_id="e0", event_type=ContentEventType.SPEECH_SEGMENT,
                start_time=0, end_time=5, speech_text="Keep going",
                visual_description="a gym", modalities=("visual", "speech"),
            ),
            ContentEvent(
                event_id="e1", event_type=ContentEventType.FUSED_INTERVAL,
                start_time=5, end_time=10, visual_description="a street",
            ),
        ),
        associations=(
            NeuralContentAssociation(
                neural_event_id="n1", neural_event_type="response_peak",
                neural_time=3.0, neural_score=1.4,
                window_start=1.0, window_end=5.0, context_window_seconds=2.0,
                content_event_ids=("e0",), speech_context=("Keep going",),
            ),
        ),
        metrics=ContentMetrics(duration_seconds=10.0, shot_count=1, scene_count=1),
        arrays=ContentArrayArtifact(
            path="features.npz", keys={}, frame_count=40,
            feature_names=("motion", "audio_energy"),
        ),
        metadata=ContentAnalysisMetadata(
            analysis_version="1.0", stimulus_sha256="a" * 64, cache_key="k1"
        ),
    )
    ContentStore(tmp_path).write(
        result, {"features": np.ones((40, 2), dtype=np.float32),
                 "feature_times": np.linspace(0, 10, 40, dtype=np.float32)}
    )
    return result


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    _write(tmp_path)
    loader = ContentLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
    app.dependency_overrides[content_loader_dep] = lambda: loader
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestEndpoints:
    def test_full_analysis(self, client: TestClient) -> None:
        body = client.get("/api/runs/run-1/content-analysis").json()
        assert body["run_id"] == "run-1"
        assert body["metrics"]["shot_count"] == 1
        assert len(body["events"]) == 2

    def test_events(self, client: TestClient) -> None:
        events = client.get("/api/runs/run-1/content-analysis/events").json()
        assert [e["event_id"] for e in events] == ["e0", "e1"]

    def test_transcript(self, client: TestClient) -> None:
        transcript = client.get("/api/runs/run-1/content-analysis/transcript").json()
        assert transcript[0]["text"] == "Keep going"

    def test_scenes_returns_shots_too(self, client: TestClient) -> None:
        body = client.get("/api/runs/run-1/content-analysis/scenes").json()
        assert "scenes" in body and "shots" in body
        assert len(body["shots"]) == 1

    def test_context_at_an_instant(self, client: TestClient) -> None:
        body = client.get("/api/runs/run-1/content-analysis/context?time=2.0").json()
        assert body["covered"] is True
        assert body["speech"] == "Keep going"
        assert len(body["events"]) == 1

    def test_context_with_a_window_widens_the_query(self, client: TestClient) -> None:
        body = client.get(
            "/api/runs/run-1/content-analysis/context?time=5.0&window=2.0"
        ).json()
        assert len(body["events"]) == 2

    def test_context_outside_the_timeline_reports_not_covered(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/runs/run-1/content-analysis/context?time=50").json()
        assert body["covered"] is False
        assert body["events"] == []

    def test_associations_carry_the_non_causal_notice(self, client: TestClient) -> None:
        """The guarantee that must survive serialization to the UI."""
        associations = client.get("/api/runs/run-1/content-analysis/associations").json()
        assert associations[0]["content_event_ids"] == ["e0"]
        text = associations[0]["interpretation"].lower()
        assert "co-occurrence" in text
        assert "not" in text

    def test_features_are_binary_with_typed_headers(self, client: TestClient) -> None:
        response = client.get("/api/runs/run-1/content-analysis/features")
        assert response.headers["X-Array-Dtype"] == "float32"
        assert response.headers["X-Array-Shape"] == "40,2"
        assert response.headers["X-Feature-Names"] == "motion,audio_energy"
        array = np.frombuffer(response.content, dtype=np.float32)
        assert array.size == 80

    def test_binary_headers_are_exposed_cross_origin(self, client: TestClient) -> None:
        exposed = client.get("/api/runs/run-1/content-analysis/features").headers[
            "Access-Control-Expose-Headers"
        ]
        assert "X-Feature-Names" in exposed

    def test_feature_times_are_returned_exactly(self, client: TestClient) -> None:
        response = client.get("/api/runs/run-1/content-analysis/features/times")
        assert response.status_code == 200
        assert response.headers["X-Array-Shape"] == "40"
        times = np.frombuffer(response.content, dtype=np.float32)
        np.testing.assert_allclose(times, np.linspace(0, 10, 40, dtype=np.float32))

    def test_summary_json_carries_no_bulk_arrays(self, client: TestClient) -> None:
        raw = client.get("/api/runs/run-1/content-analysis").text
        assert len(raw) < 40_000


class TestFailureModes:
    def test_unanalysed_run_404s_with_instructions(self, client: TestClient) -> None:
        """A page load must never trigger the pipeline."""
        response = client.get("/api/runs/never-analysed/content-analysis")
        assert response.status_code == 404
        assert "analyze-content" in response.json()["detail"]

    def test_every_subresource_404s_consistently(self, client: TestClient) -> None:
        for path in ("events", "transcript", "scenes", "associations", "features"):
            response = client.get(f"/api/runs/never-analysed/content-analysis/{path}")
            assert response.status_code == 404, path

    def test_context_404s_for_an_unanalysed_run(self, client: TestClient) -> None:
        assert (
            client.get(
                "/api/runs/never-analysed/content-analysis/context?time=1"
            ).status_code
            == 404
        )

    def test_keyframe_traversal_is_refused(self, tmp_path: Path) -> None:
        """A crafted name must not escape the run's keyframe directory."""
        _write(tmp_path)
        loader = ContentLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
        from blackmirror.errors import ArtifactReadError

        with pytest.raises(ArtifactReadError):
            loader.keyframe_path("run-1", "../../../../etc/passwd")

    def test_missing_signal_is_reported(self, tmp_path: Path) -> None:
        _write(tmp_path)
        loader = ContentLoader(Settings(artifact_dir=tmp_path, log_level="WARNING"))
        from blackmirror.errors import ArtifactReadError

        with pytest.raises(ArtifactReadError, match="not present"):
            loader.signal("run-1", "does_not_exist")
