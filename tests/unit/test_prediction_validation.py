"""Prediction validation must DESCRIBE anomalies, never silently repair them."""

from __future__ import annotations

import numpy as np

from blackmirror.inference.validation import validate_predictions


def test_clean_prediction_passes() -> None:
    rng = np.random.default_rng(0)
    array = rng.normal(size=(10, 16)).astype(np.float32)
    result = validate_predictions(array, expected_vertex_count=16)

    assert result.valid
    assert result.errors == ()
    assert result.nan_count == 0
    assert result.inf_count == 0
    assert result.finite_count == array.size
    assert result.shape == (10, 16)
    assert result.dtype == "float32"
    assert result.min is not None and result.max is not None
    assert result.p01 <= result.p50 <= result.p99


def test_nan_and_inf_are_counted_not_removed() -> None:
    array = np.ones((4, 4), dtype=np.float32)
    array[0, 0] = np.nan
    array[1, 1] = np.inf
    array[2, 2] = -np.inf

    result = validate_predictions(array, expected_vertex_count=4)

    assert result.nan_count == 1
    assert result.inf_count == 2
    assert result.finite_count == 13
    # Structurally usable: the run still completes and the raw data is kept.
    assert result.valid
    assert any("NaN" in w for w in result.warnings)
    assert any("infinite" in w for w in result.warnings)
    # The input array must be untouched.
    assert np.isnan(array[0, 0])
    assert np.isposinf(array[1, 1])


def test_statistics_ignore_non_finite_values() -> None:
    array = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    result = validate_predictions(array, expected_vertex_count=2)
    assert result.min == 1.0
    assert result.max == 3.0
    assert result.mean == 2.0


def test_wrong_rank_is_a_structural_error() -> None:
    result = validate_predictions(np.zeros((5,), dtype=np.float32))
    assert not result.valid
    assert any("2-D" in e for e in result.errors)


def test_vertex_count_mismatch_is_a_structural_error() -> None:
    """A wrong vertex count means the cortical mapping is wrong — never a warning."""
    result = validate_predictions(np.zeros((3, 10), dtype=np.float32), expected_vertex_count=16)
    assert not result.valid
    assert any("does not match" in e for e in result.errors)


def test_zero_length_time_axis_is_a_structural_error() -> None:
    result = validate_predictions(np.zeros((0, 16), dtype=np.float32), expected_vertex_count=16)
    assert not result.valid
    assert any("Temporal dimension is zero" in e for e in result.errors)


def test_non_numeric_dtype_is_a_structural_error() -> None:
    result = validate_predictions(np.array([["a", "b"], ["c", "d"]]))
    assert not result.valid
    assert any("not numeric" in e for e in result.errors)


def test_constant_output_is_warned_about() -> None:
    """A model that ignores its input makes A/B testing meaningless."""
    result = validate_predictions(np.ones((10, 16), dtype=np.float32), expected_vertex_count=16)
    assert result.valid
    assert result.constant_vertices == 16
    assert result.temporal_variance_mean == 0.0
    assert any("does not vary over time" in w for w in result.warnings)


def test_single_timepoint_is_warned_about() -> None:
    result = validate_predictions(np.zeros((1, 16), dtype=np.float32), expected_vertex_count=16)
    assert result.valid
    assert any("one time point" in w for w in result.warnings)
