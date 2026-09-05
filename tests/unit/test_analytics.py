from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blackmirror.analytics.atlas import AtlasMapping, validate_atlas_mapping
from blackmirror.analytics.engine import NeuralAnalyticsEngine
from blackmirror.analytics.events import NeuralEventDetector
from blackmirror.analytics.metrics import (
    aggregate_hemispheres,
    aggregate_regions,
    analyze_window,
    change_magnitude,
    cosine_similarity,
    global_metrics,
    roi_correlation_matrix,
    spatial_concentration,
    top_vertices,
)
from blackmirror.analytics.schemas import AtlasMetadata, CorticalRegion, Hemisphere
from blackmirror.analytics.storage import AnalyticsStore


def tiny_atlas() -> AtlasMapping:
    regions = (
        CorticalRegion(
            region_id=0,
            atlas_label=10,
            name="ROI A",
            hemisphere=Hemisphere.LEFT,
            atlas="test",
            atlas_version="1",
            vertex_count=2,
        ),
        CorticalRegion(
            region_id=1,
            atlas_label=20,
            name="ROI B",
            hemisphere=Hemisphere.RIGHT,
            atlas="test",
            atlas_version="1",
            vertex_count=2,
        ),
    )
    return AtlasMapping(
        AtlasMetadata(
            name="test",
            version="1",
            surface_space="tiny",
            source="unit fixture",
            license="test-only",
            region_count=2,
            regions=regions,
        ),
        np.array([0, 0, 1, 1], dtype=np.int32),
        np.zeros(4, dtype=bool),
    )


def mesh_dir(tmp_path: Path) -> Path:
    path = tmp_path / "mesh"
    path.mkdir()
    (path / "mapping.json").write_text(
        json.dumps(
            {
                "total_vertices": 4,
                "hemisphere_index_ranges": {"left": [0, 2], "right": [2, 4]},
            }
        )
    )
    return path


def test_roi_aggregation_exact_example() -> None:
    responses = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float32)
    actual = aggregate_regions(responses, np.array([0, 0, 1, 1]), 2)
    np.testing.assert_array_equal(actual, [[1.5, 3.5], [3.0, 7.0]])


def test_roi_aggregation_omits_unmapped_and_nonfinite_values() -> None:
    responses = np.array([[1, np.nan, 100, 4]], dtype=np.float32)
    actual = aggregate_regions(responses, np.array([0, 0, -1, 1]), 2)
    np.testing.assert_array_equal(actual, [[1.0, 4.0]])


def test_invalid_atlas_mapping_is_rejected(tmp_path: Path) -> None:
    atlas = tiny_atlas()
    broken = AtlasMapping(atlas.metadata, np.array([0, 0, 1, 9]), atlas.medial_wall_mask)
    report = validate_atlas_mapping(broken, prediction_vertices=4, mesh_dir=mesh_dir(tmp_path))
    assert report.valid is False
    assert report.invalid_vertices == 1


def test_mapping_count_mismatch_returns_bounded_invalid_report(tmp_path: Path) -> None:
    report = validate_atlas_mapping(
        tiny_atlas(), prediction_vertices=2, mesh_dir=mesh_dir(tmp_path)
    )
    assert not report.valid
    assert report.mapped_vertices == 2
    assert report.mapped_fraction == 1.0


def test_mapping_validation_exact_counts(tmp_path: Path) -> None:
    report = validate_atlas_mapping(
        tiny_atlas(), prediction_vertices=4, mesh_dir=mesh_dir(tmp_path)
    )
    assert report.valid
    assert report.mapped_vertices == 4
    assert report.mapped_fraction == 1.0
    assert report.hemisphere_counts == {"left": 2, "right": 2}


def test_hemisphere_and_global_metrics_are_exact() -> None:
    responses = np.array([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=np.float32)
    hemis = aggregate_hemispheres(responses, {"left": (0, 2), "right": (2, 4)})
    np.testing.assert_array_equal(hemis["left"], [2, 3])
    np.testing.assert_array_equal(hemis["right"], [6, 7])
    np.testing.assert_array_equal(hemis["left_minus_right"], [-4, -4])
    metrics = global_metrics(responses)
    np.testing.assert_array_equal(metrics["mean"], [4, 5])
    np.testing.assert_allclose(metrics["l2_magnitude"], [np.sqrt(84), np.sqrt(120)])


def test_global_and_hemisphere_metrics_exclude_medial_wall() -> None:
    responses = np.array([[1.0, 1000.0, 3.0, 2000.0]])
    wall = np.array([False, True, False, True])
    metrics = global_metrics(responses, wall)
    assert metrics["mean"][0] == 2.0
    assert metrics["l2_magnitude"][0] == pytest.approx(np.sqrt(10))
    hemis = aggregate_hemispheres(
        responses, {"left": (0, 2), "right": (2, 4)}, wall
    )
    np.testing.assert_array_equal(hemis["left"], [1.0])
    np.testing.assert_array_equal(hemis["right"], [3.0])
    changes = change_magnitude(np.vstack([responses, responses * 2]), wall)
    np.testing.assert_array_equal(changes, [0, np.sqrt(10)])


def test_all_nonfinite_global_row_is_nan_not_uninitialized_memory() -> None:
    metrics = global_metrics(np.array([[np.nan, np.inf], [3.0, 4.0]]))
    assert np.isnan(metrics["l2_magnitude"][0])
    assert np.isnan(metrics["rms_magnitude"][0])
    assert metrics["rms_magnitude"][1] == pytest.approx(np.sqrt(12.5))


def test_roi_correlation_is_pairwise_finite_and_always_square() -> None:
    values = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 6.0]])
    correlation = roi_correlation_matrix(values)
    assert correlation.shape == (2, 2)
    np.testing.assert_allclose(correlation, np.ones((2, 2)))
    single = roi_correlation_matrix(np.array([[1.0], [2.0], [3.0]]))
    assert single.shape == (1, 1)
    assert single[0, 0] == pytest.approx(1.0)


def test_change_similarity_concentration_and_top_vertices() -> None:
    responses = np.array([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(change_magnitude(responses), [0, 1])
    assert cosine_similarity(responses[0], responses[1]) == pytest.approx(1 / np.sqrt(2))
    np.testing.assert_allclose(spatial_concentration(responses), [1, 1 / 3])
    assert top_vertices(responses, 1, 2) == ((1, 1.0), (0, 1.0))


def test_event_detector_finds_prominent_interior_peak() -> None:
    detector = NeuralEventDetector(z_threshold=0.5, min_distance=1)
    values = np.array([0, 0, 10, 0, 0], dtype=float)
    events = detector.detect(values, values, np.zeros(5), np.ones((5, 2)), np.arange(5.0))
    assert len(events) == 1
    assert events[0].timestep == 2
    assert events[0].event_type.value == "response_peak"


def test_event_detector_never_ranks_nonfinite_regions() -> None:
    detector = NeuralEventDetector(z_threshold=0.5, min_distance=1)
    values = np.array([0, 0, 10, 0, 0], dtype=float)
    roi = np.ones((5, 4))
    roi[2] = [np.nan, 3.0, np.inf, 2.0]
    event = detector.detect(values, values, np.zeros(5), roi, np.arange(5.0))[0]
    assert event.affected_region_ids == (1, 3)


def test_window_uses_actual_sparse_timestamps() -> None:
    responses = np.array([[0, 0], [1, 1], [10, 10]], dtype=float)
    roi = responses.copy()
    result = analyze_window(responses, roi, np.array([0.0, 4.0, 20.0]), 3.0, 5.0)
    assert result.start_time == 4.0
    assert result.end_time == 4.0
    assert result.peak_timestep == 1


def test_engine_serialization_and_persistence(tmp_path: Path) -> None:
    atlas = tiny_atlas()
    mesh = mesh_dir(tmp_path)
    validation = validate_atlas_mapping(atlas, prediction_vertices=4, mesh_dir=mesh)
    responses = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 1, 1, 1]], dtype=float)
    result, arrays = NeuralAnalyticsEngine().analyze(
        run_id="run-1",
        responses=responses,
        times=np.array([0.0, 2.0, 9.0]),
        atlas=atlas,
        validation=validation,
        hemisphere_ranges={"left": (0, 2), "right": (2, 4)},
    )
    assert result.roi_responses[0].peak_timestep == 1
    assert "psychological state" in result.metadata.interpretation_notice
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    np.save(run_dir / "predictions.npy", responses)
    store = AnalyticsStore(tmp_path)
    store.write(result, arrays, atlas)
    loaded = store.read("run-1")
    assert loaded == result
    with np.load(run_dir / "analytics" / "timeseries.npz") as saved:
        np.testing.assert_array_equal(saved["roi_timeseries"], arrays["roi_timeseries"])
