"""Real TRIBE v2 inference. Opt-in.

    pytest -m tribe_integration

Deselected by default (see `addopts` in pyproject.toml) because it needs the
709 MB checkpoint, >10 GB of frozen feature extractors, possibly gated
HuggingFace access, and — on CPU — tens of minutes.

Set BLACKMIRROR_TEST_STIMULUS to point at a short clip; otherwise these skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from blackmirror.config.settings import BackendName, Settings
from blackmirror.inference.service import InferenceService
from blackmirror.schemas.prediction import PredictionResult

pytestmark = pytest.mark.tribe_integration

FSAVERAGE5_PER_HEMISPHERE = 10242
FSAVERAGE5_TOTAL = FSAVERAGE5_PER_HEMISPHERE * 2


@pytest.fixture(scope="module")
def stimulus_path() -> Path:
    raw = os.environ.get("BLACKMIRROR_TEST_STIMULUS")
    if not raw:
        pytest.skip("Set BLACKMIRROR_TEST_STIMULUS to a short media file to run these.")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        pytest.skip(f"BLACKMIRROR_TEST_STIMULUS does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        backend=BackendName.TRIBE_V2,
        artifact_dir=tmp_path_factory.mktemp("artifacts"),
        # Reuse the developer's real feature cache; re-extracting is very slow.
        model_cache_dir=Path(os.environ.get("BLACKMIRROR_MODEL_CACHE_DIR", "./cache")).resolve(),
    )


@pytest.fixture(scope="module")
def result(settings: Settings, stimulus_path: Path) -> PredictionResult:
    """One real inference, shared by every assertion below."""
    pytest.importorskip("tribev2")
    return InferenceService(settings).predict(stimulus_path)


def test_model_metadata_is_read_from_the_checkpoint(settings: Settings) -> None:
    pytest.importorskip("tribev2")
    metadata = InferenceService(settings).get_model_metadata()

    assert metadata.backend == "tribe_v2"
    assert metadata.license == "CC-BY-NC-4.0"
    assert not metadata.is_synthetic
    # Predictions are for an average subject, and that must be stated.
    assert "average_subjects" in (metadata.subject_conditioning or "")
    assert metadata.extra["surface_mesh"] == "fsaverage5"
    assert metadata.feature_extractors["text"] == "meta-llama/Llama-3.2-3B"


def test_prediction_has_the_expected_cortical_shape(result: PredictionResult) -> None:
    assert len(result.prediction.shape) == 2
    assert result.prediction.cortical_features == FSAVERAGE5_TOTAL
    assert result.prediction.temporal_samples > 0
    assert result.cortical.surface_space == "fsaverage5"
    assert result.cortical.vertices_per_hemisphere == FSAVERAGE5_PER_HEMISPHERE
    assert result.cortical.hemisphere_index_ranges == {
        "left": (0, FSAVERAGE5_PER_HEMISPHERE),
        "right": (FSAVERAGE5_PER_HEMISPHERE, FSAVERAGE5_TOTAL),
    }


def test_tr_is_read_from_the_model_not_assumed(result: PredictionResult) -> None:
    assert result.temporal.tr_seconds > 0
    assert result.temporal.sampling_rate_hz == pytest.approx(1.0 / result.temporal.tr_seconds)


def test_predictions_are_numerically_usable(result: PredictionResult) -> None:
    assert result.validation.valid
    assert result.validation.nan_count == 0
    assert result.validation.inf_count == 0
    assert result.validation.std is not None and result.validation.std > 0
    # If the output does not change over time, A/B comparison is meaningless.
    assert result.validation.temporal_variance_mean is not None
    assert result.validation.temporal_variance_mean > 0


def test_persisted_matrix_matches_the_manifest(result: PredictionResult) -> None:
    array = np.load(result.artifacts.run_dir / result.prediction.artifact_path)
    assert array.shape == result.prediction.shape
    assert str(array.dtype) == result.prediction.dtype


def test_temporal_alignment_is_recoverable(result: PredictionResult) -> None:
    """The alignment Phase 2 needs must survive segment filtering."""
    run_dir = result.artifacts.run_dir
    with np.load(run_dir / result.temporal.arrays_artifact_path) as temporal:
        starts = temporal["segment_start_seconds"]
        derived = temporal["stimulus_time_seconds"]

    assert starts.shape[0] == result.prediction.temporal_samples
    assert np.all(np.diff(starts) > 0), "segment start times must increase monotonically"
    np.testing.assert_allclose(
        derived, starts - result.temporal.hemodynamic_offset_seconds
    )

    if not result.temporal.timeline_is_contiguous:
        assert result.temporal.segments_were_filtered


def test_provenance_supports_a_controlled_comparison(result: PredictionResult) -> None:
    provenance = result.provenance
    assert provenance.stimulus_sha256 == result.stimulus.sha256
    assert provenance.torch_version is not None
    assert provenance.backend_package_version is not None
    assert provenance.cache_key
    # Patches must be recorded, so two runs can be checked for comparability.
    assert isinstance(provenance.applied_compat_patches, tuple)


def test_result_is_consumable_without_tribe(result: PredictionResult) -> None:
    """The Phase 2 contract: reload a run knowing nothing about TRIBE."""
    manifest = result.artifacts.run_dir / "manifest.json"
    reloaded = PredictionResult.model_validate_json(manifest.read_text(encoding="utf-8"))
    assert reloaded.run_id == result.run_id
    assert reloaded.cortical.vertex_count == FSAVERAGE5_TOTAL
