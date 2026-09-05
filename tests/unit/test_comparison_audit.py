"""Structural guards for Phase 5, and regressions for bugs found auditing it.

The comparability gate and the content availability gate are both hand-written
lists that must track schemas defined elsewhere. Both had silently gone stale:
`prediction.normalization` and the whole hemodynamic-alignment convention were
unchecked, and three Phase 4 features were being compared with no availability
gate at all. These tests fail when that happens again.
"""

from __future__ import annotations

import numpy as np
import pytest

from blackmirror.comparison.content import (
    AVAILABILITY_COLUMNS,
    AVAILABILITY_FOR_PREFIX,
    ContentVariant,
    compare_content_features,
)
from blackmirror.content.features import FEATURE_NAMES
from blackmirror.schemas.prediction import (
    CorticalMetadata,
    PredictionArrayMetadata,
    TemporalMetadata,
)

#: Fields that genuinely do not affect whether two runs can be subtracted.
#: Anything not listed here must be checked by validate_comparability.
_NOT_COMPARABILITY_RELEVANT = {
    # Descriptive or per-run bookkeeping, not a condition of the comparison.
    "artifact_path", "arrays_artifact_path", "array_keys", "event_summary", "notes",
    "mesh_artifact_dir", "mapping_notes", "atlas",
    # Shape/count facts already enforced by vertex_count and array validation.
    "shape", "dtype", "axis_names", "temporal_samples", "cortical_features",
    "is_raw_model_output", "is_surface_based", "vertices_per_hemisphere",
    "vertex_count", "n_time_points", "sampling_rate_hz",
    # Properties of one run's own timeline; alignment handles mismatches
    # explicitly and reports coverage.
    "n_segments_kept", "n_segments_total", "first_segment_start_seconds",
    "last_segment_end_seconds", "covered_seconds", "timeline_is_contiguous",
    "segments_were_filtered",
    # Provenance of the offset value, not the convention itself, which IS checked.
    "hemodynamic_offset_verified", "hemodynamic_offset_source",
}


class TestComparabilityGateCoversTheSchema:
    """Every field that changes what a subtraction means must be gated."""

    @pytest.mark.parametrize(
        "model",
        [PredictionArrayMetadata, TemporalMetadata, CorticalMetadata],
        ids=lambda m: m.__name__,
    )
    def test_no_relevant_field_is_left_unchecked(self, model: type) -> None:
        from pathlib import Path

        source = Path("src/blackmirror/comparison/validation.py").read_text()
        unchecked = [
            name
            for name in model.model_fields
            if name not in source and name not in _NOT_COMPARABILITY_RELEVANT
        ]
        assert not unchecked, (
            f"{model.__name__} field(s) {unchecked} are neither checked by the "
            f"comparability gate nor listed as irrelevant. A run pair differing in "
            f"one of these would be compared as though it were controlled."
        )

    def test_the_critical_conventions_are_named(self) -> None:
        """These four decide whether a subtraction means anything at all."""
        from pathlib import Path

        source = Path("src/blackmirror/comparison/validation.py").read_text()
        for field in (
            "normalization",          # normalised minus unnormalised is meaningless
            "tr_seconds",             # rows would span different durations
            "output_is_stimulus_aligned",   # a timestamp would mean different things
            "hemodynamic_offset_seconds",
        ):
            assert field in source, f"{field} must be a comparability check"


class TestContentAvailabilityGate:
    """An unanalysed stream is zeros, not NaN, so the gate is load-bearing."""

    def test_every_current_feature_is_gated(self) -> None:
        ungated = [
            name
            for name in FEATURE_NAMES
            if name not in AVAILABILITY_FOR_PREFIX and name not in AVAILABILITY_COLUMNS
        ]
        assert not ungated, (
            f"feature(s) {ungated} would be compared without an availability gate; "
            f"an absent stream reports 0.0, which is finite, so the difference would "
            f"read as a measured zero rather than as missing data."
        )

    def test_an_ungated_feature_is_refused_rather_than_compared(self) -> None:
        names = ("motion", "visual_available", "a_new_ungated_feature")
        times = np.array([0.0, 1.0])
        matrix = np.ones((2, 3), dtype=np.float64)
        variant = ContentVariant(
            schema_version="1", analysis_version="1.2", models={}, configuration={},
            feature_names=names, times=times, matrix=matrix,
        )
        with pytest.raises(ValueError, match="no availability gate"):
            compare_content_features(variant, variant)

    def test_an_unavailable_stream_is_nan_not_zero(self) -> None:
        """The bug this gate prevents, stated as a test."""
        names = ("motion", "motion_structural", "visual_available")
        times = np.array([0.0, 1.0])
        # Visual stream absent: features are finite zeros, mask is 0.
        matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        variant = ContentVariant(
            schema_version="1", analysis_version="1.2", models={}, configuration={},
            feature_names=names, times=times, matrix=matrix,
        )
        difference = compare_content_features(variant, variant)
        motion = difference.signed_delta[:, 0]
        structural = difference.signed_delta[:, 1]
        assert np.isnan(motion).all(), "an unanalysed stream must not report a difference"
        assert np.isnan(structural).all(), "the same must hold for every gated feature"
