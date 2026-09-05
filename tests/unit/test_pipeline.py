"""End-to-end pipeline behaviour, exercised through the mock backend.

The point of these tests is the *abstraction*: the service, artifact store,
validation and contract must all work against a backend that is not TRIBE.
If they do, swapping the model later really is a one-class change.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from blackmirror.config.settings import Settings
from blackmirror.errors import InferenceError, PostprocessingError, PreprocessingError
from blackmirror.inference.backend import (
    CorticalPredictorBackend,
    PreparedStimulus,
    RawPrediction,
    SegmentRecord,
)
from blackmirror.inference.mock_backend import MockBackend
from blackmirror.inference.service import InferenceService
from blackmirror.schemas.prediction import PredictionResult


def test_mock_backend_satisfies_the_protocol() -> None:
    assert isinstance(MockBackend(), CorticalPredictorBackend)


def test_full_pipeline_writes_every_artifact(tmp_settings: Settings, video_file) -> None:
    service = InferenceService(tmp_settings)
    result = service.predict(video_file)

    run_dir = result.artifacts.run_dir
    for name in (
        "manifest.json",
        "stimulus.json",
        "predictions.npy",
        "temporal.npz",
        "prediction_metadata.json",
        "validation.json",
        "performance.json",
        "provenance.json",
        "events_summary.json",
    ):
        assert (run_dir / name).is_file(), f"missing artifact: {name}"


def test_pipeline_runs_without_importing_tribe(tmp_settings: Settings, video_file) -> None:
    """The central architectural claim, asserted mechanically."""
    for name in [m for m in sys.modules if m.startswith(("tribev2", "neuralset"))]:
        del sys.modules[name]

    InferenceService(tmp_settings).predict(video_file)

    assert not any(m.startswith("tribev2") for m in sys.modules)


def test_manifest_reloads_into_the_contract(tmp_settings: Settings, video_file) -> None:
    service = InferenceService(tmp_settings)
    result = service.predict(video_file)

    manifest = result.artifacts.run_dir / "manifest.json"
    reloaded = PredictionResult.model_validate_json(manifest.read_text(encoding="utf-8"))

    assert reloaded.run_id == result.run_id
    assert reloaded.prediction.shape == result.prediction.shape
    assert reloaded.cortical.vertex_count == 20484


def test_persisted_matrix_matches_the_declared_metadata(
    tmp_settings: Settings, video_file
) -> None:
    result = InferenceService(tmp_settings).predict(video_file)
    array = np.load(result.artifacts.run_dir / result.prediction.artifact_path)

    assert array.shape == result.prediction.shape
    assert str(array.dtype) == result.prediction.dtype
    assert array.shape[1] == result.cortical.vertex_count


def test_temporal_arrays_align_with_the_matrix(tmp_settings: Settings, video_file) -> None:
    result = InferenceService(tmp_settings).predict(video_file)
    run_dir = result.artifacts.run_dir

    array = np.load(run_dir / result.prediction.artifact_path)
    with np.load(run_dir / result.temporal.arrays_artifact_path) as temporal:
        for key in (
            "segment_start_seconds",
            "segment_duration_seconds",
            "segment_n_events",
            "stimulus_time_seconds",
        ):
            assert temporal[key].shape[0] == array.shape[0]


def test_synthetic_runs_are_labelled_everywhere(tmp_settings: Settings, video_file) -> None:
    """Mock output must never be presentable as a brain prediction."""
    result = InferenceService(tmp_settings).predict(video_file)

    assert result.model.is_synthetic
    assert "SYNTHETIC" in result.interpretation_notice
    assert "SYNTHETIC" in result.prediction.semantics.upper()

    manifest = (result.artifacts.run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "SYNTHETIC" in manifest


def test_same_content_produces_a_stable_cache_key(tmp_settings: Settings, video_file) -> None:
    service = InferenceService(tmp_settings)
    first = service.predict(video_file)
    second = service.predict(video_file)

    assert first.run_id != second.run_id
    assert first.provenance.cache_key == second.provenance.cache_key


def test_cache_reuse_returns_the_earlier_run(tmp_settings: Settings, video_file) -> None:
    first = InferenceService(tmp_settings).predict(video_file)

    reuse = tmp_settings.model_copy(update={"reuse_cached_runs": True})
    second = InferenceService(reuse).predict(video_file)

    assert second.run_id == first.run_id


def test_model_loads_once_across_a_batch(tmp_settings: Settings, tmp_path) -> None:
    """Phase 5 depends on one loaded model scoring many variants."""
    variants = []
    for i in range(3):
        path = tmp_path / f"variant_{i}.mp4"
        path.write_bytes(bytes([i]) * 4096)
        variants.append(path)

    backend = MockBackend()
    service = InferenceService(tmp_settings, backend=backend)

    load_calls = 0
    original_load = backend.load

    def counting_load() -> None:
        nonlocal load_calls
        if not backend.is_loaded():
            load_calls += 1
        original_load()

    backend.load = counting_load  # type: ignore[method-assign]

    results = service.predict_many(variants)

    assert len(results) == 3
    assert load_calls == 1
    # Identical model provenance is what makes the comparison fair.
    assert len({r.provenance.model_fingerprint for r in results}) == 1
    assert len({r.run_id for r in results}) == 3


def test_different_variants_produce_different_cache_keys(
    tmp_settings: Settings, tmp_path
) -> None:
    paths = []
    for i in range(2):
        path = tmp_path / f"v{i}.mp4"
        path.write_bytes(bytes([i]) * 4096)
        paths.append(path)

    service = InferenceService(tmp_settings)
    keys = {service.predict(p).provenance.cache_key for p in paths}
    assert len(keys) == 2


# --- Failure propagation --------------------------------------------------


class _BrokenBackend(MockBackend):
    """Fails at a chosen stage, to check error context is preserved."""

    def __init__(self, stage: str) -> None:
        super().__init__()
        self.stage = stage

    def preprocess(self, stimulus):  # type: ignore[no-untyped-def]
        if self.stage == "preprocess":
            raise ValueError("upstream decoder exploded")
        return super().preprocess(stimulus)

    def infer(self, prepared):  # type: ignore[no-untyped-def]
        if self.stage == "infer":
            raise RuntimeError("out of memory")
        if self.stage == "misaligned":
            raw = super().infer(prepared)
            return RawPrediction(
                array=raw.array,
                segments=raw.segments[:-1],
                tr_seconds=raw.tr_seconds,
                surface_space=raw.surface_space,
                vertices_per_hemisphere=raw.vertices_per_hemisphere,
            )
        return super().infer(prepared)


def test_preprocessing_failure_names_the_stimulus(tmp_settings: Settings, video_file) -> None:
    service = InferenceService(tmp_settings, backend=_BrokenBackend("preprocess"))
    with pytest.raises(PreprocessingError) as info:
        service.predict(video_file)

    assert "clip.mp4" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_inference_failure_names_the_device(tmp_settings: Settings, video_file) -> None:
    service = InferenceService(tmp_settings, backend=_BrokenBackend("infer"))
    with pytest.raises(InferenceError) as info:
        service.predict(video_file)

    assert "cpu" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_row_segment_mismatch_is_rejected(tmp_settings: Settings, video_file) -> None:
    """Better to fail than to persist a silently misaligned timeline."""
    service = InferenceService(tmp_settings, backend=_BrokenBackend("misaligned"))
    with pytest.raises(PostprocessingError, match="Temporal alignment"):
        service.predict(video_file)


def test_no_run_directory_is_left_behind_on_failure(
    tmp_settings: Settings, video_file
) -> None:
    service = InferenceService(tmp_settings, backend=_BrokenBackend("infer"))
    with pytest.raises(InferenceError):
        service.predict(video_file)

    assert service.store.list_runs() == []


def test_mock_backend_is_deterministic(tmp_settings: Settings, video_file) -> None:
    """Reproducibility: identical bytes and config must give identical numbers."""
    a = InferenceService(tmp_settings).predict(video_file)
    b = InferenceService(tmp_settings).predict(video_file)

    left = np.load(a.artifacts.run_dir / "predictions.npy")
    right = np.load(b.artifacts.run_dir / "predictions.npy")
    np.testing.assert_array_equal(left, right)


def test_segment_record_defaults() -> None:
    record = SegmentRecord(index=0, start_seconds=3.0, duration_seconds=1.0)
    assert record.n_events == 0
    assert record.event_types == ()


def test_prepared_stimulus_carries_an_event_summary(stimulus) -> None:
    prepared: PreparedStimulus = MockBackend().preprocess(stimulus)
    assert prepared.event_summary["synthetic"] is True
    assert prepared.event_summary["n_events"] > 0
