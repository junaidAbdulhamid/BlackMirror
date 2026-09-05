"""The contract Phase 2+ consumes must serialize and round-trip losslessly."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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


def _result() -> PredictionResult:
    stimulus = StimulusInput(
        path=Path("/tmp/clip.mp4"),
        filename="clip.mp4",
        media_type=MediaType.VIDEO,
        file_size_bytes=1234,
        duration_seconds=8.0,
        sha256="a" * 64,
    )
    model = ModelMetadata(
        name="TRIBE v2",
        backend="tribe_v2",
        model_id="facebook/tribev2",
        checkpoint="best.ckpt",
        version="0.1.0",
        license="CC-BY-NC-4.0",
        modalities=("video", "audio", "text"),
        feature_extractors={"text": "meta-llama/Llama-3.2-3B"},
        loaded_device="cpu",
        dtype="float32",
        subject_conditioning="average_subjects",
    )
    return PredictionResult(
        run_id="20260903T000000Z-abcdef12",
        stimulus=stimulus,
        model=model,
        prediction=PredictionArrayMetadata(
            shape=(8, 20484),
            dtype="float32",
            axis_names=("time", "vertex"),
            temporal_samples=8,
            cortical_features=20484,
            semantics="Predicted cortical response.",
            artifact_path=Path("predictions.npy"),
        ),
        temporal=TemporalMetadata(
            n_time_points=8,
            tr_seconds=1.0,
            sampling_rate_hz=1.0,
            timeline_is_contiguous=True,
            segments_were_filtered=True,
            hemodynamic_offset_seconds=5.0,
        ),
        cortical=CorticalMetadata(
            surface_space="fsaverage5",
            vertex_count=20484,
            vertices_per_hemisphere=10242,
            hemisphere_order=("left", "right"),
            hemisphere_index_ranges={"left": (0, 10242), "right": (10242, 20484)},
        ),
        validation=PredictionValidation(
            valid=True,
            shape=(8, 20484),
            dtype="float32",
            nan_count=0,
            inf_count=0,
            finite_count=8 * 20484,
            mean=0.01,
        ),
        performance=PerformanceMetrics(
            preprocessing_seconds=1.0,
            model_load_seconds=2.0,
            inference_seconds=3.0,
            postprocessing_seconds=0.1,
            validation_seconds=0.1,
            persistence_seconds=0.1,
            total_seconds=6.3,
        ),
        provenance=RunProvenance(
            run_id="20260903T000000Z-abcdef12",
            blackmirror_version="0.1.0",
            python_version="3.12.14",
            platform="macOS",
            device="cpu",
            dtype="float32",
            random_seed=1234,
            stimulus_sha256="a" * 64,
            model_fingerprint="tribe_v2|facebook/tribev2|best.ckpt|0.1.0|float32",
            preprocessing_config={"remove_empty_segments": True},
            cache_key="c" * 64,
        ),
        artifacts=ArtifactPaths(
            run_dir=Path("/tmp/runs/x"),
            manifest=Path("manifest.json"),
            stimulus=Path("stimulus.json"),
            predictions=Path("predictions.npy"),
            prediction_metadata=Path("prediction_metadata.json"),
            validation=Path("validation.json"),
            performance=Path("performance.json"),
            provenance=Path("provenance.json"),
        ),
    )


def test_result_round_trips_through_json() -> None:
    original = _result()
    restored = PredictionResult.model_validate_json(original.model_dump_json())

    assert restored.run_id == original.run_id
    assert restored.prediction.shape == (8, 20484)
    assert restored.cortical.hemisphere_index_ranges["right"] == (10242, 20484)
    assert restored.temporal.hemodynamic_offset_seconds == 5.0
    assert restored.stimulus.sha256 == original.stimulus.sha256
    assert isinstance(restored.created_at, dt.datetime)


def test_manifest_contains_no_arrays() -> None:
    """Rule: arrays live in .npy/.npz; JSON stays small and human-readable."""
    payload = json.loads(_result().model_dump_json())

    def walk(node: object) -> None:
        if isinstance(node, list):
            assert len(node) < 100, "a large array leaked into the manifest"
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(payload)
    assert len(json.dumps(payload)) < 20_000


def test_every_result_carries_an_interpretation_notice() -> None:
    notice = _result().interpretation_notice
    assert "not measurements" in notice.lower()
    assert "medical" in notice.lower()


def test_result_is_immutable() -> None:
    result = _result()
    with pytest.raises(ValidationError):
        result.run_id = "tampered"  # type: ignore[misc]


def test_offset_cannot_be_marked_as_applied_to_raw() -> None:
    """The type system forbids claiming the raw matrix was shifted."""
    with pytest.raises(ValidationError):
        TemporalMetadata(
            n_time_points=1,
            tr_seconds=1.0,
            sampling_rate_hz=1.0,
            timeline_is_contiguous=True,
            segments_were_filtered=False,
            hemodynamic_offset_applied_to_raw=True,  # type: ignore[arg-type]
        )


def test_stimulus_requires_a_full_sha256() -> None:
    with pytest.raises(ValidationError):
        StimulusInput(
            path=Path("/tmp/x.mp4"),
            filename="x.mp4",
            media_type=MediaType.VIDEO,
            file_size_bytes=1,
            sha256="tooshort",
        )


def test_artifact_paths_resolve_against_the_run_dir() -> None:
    paths = _result().artifacts
    assert paths.resolve(Path("predictions.npy")) == Path("/tmp/runs/x/predictions.npy")
    assert paths.resolve(Path("/abs/elsewhere.npy")) == Path("/abs/elsewhere.npy")


def test_model_fingerprint_distinguishes_configurations() -> None:
    base = _result().model
    assert base.fingerprint() != base.model_copy(update={"dtype": "bfloat16"}).fingerprint()
    assert base.fingerprint() != base.model_copy(update={"version": "0.2.0"}).fingerprint()


def test_summary_lines_flag_a_gappy_timeline() -> None:
    result = _result()
    gappy = result.model_copy(
        update={"temporal": result.temporal.model_copy(update={"timeline_is_contiguous": False})}
    )
    assert any("GAPPY" in line for line in gappy.summary_lines())


def test_fingerprint_distinguishes_feature_extractors() -> None:
    """Swapping the text encoder changes what the head sees.

    Two such runs are not comparable and must not share a cache entry, so the
    encoder identity has to participate in the model fingerprint.
    """
    base = _result().model
    swapped = base.model_copy(
        update={"feature_extractors": {"text": "unsloth/Llama-3.2-3B"}}
    )
    assert base.fingerprint() != swapped.fingerprint()

    # Ordering of the extractor dict must not change the fingerprint.
    a = base.model_copy(update={"feature_extractors": {"text": "x", "audio": "y"}})
    b = base.model_copy(update={"feature_extractors": {"audio": "y", "text": "x"}})
    assert a.fingerprint() == b.fingerprint()
