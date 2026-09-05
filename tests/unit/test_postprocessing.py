"""Temporal alignment and cortical mapping.

These are the two facts Phase 2 cannot be built without, and the gappy-timeline
case is the one most likely to be silently mishandled.
"""

from __future__ import annotations

import numpy as np

from blackmirror.inference.backend import RawPrediction, SegmentRecord
from blackmirror.inference.postprocessing import (
    build_cortical_metadata,
    build_prediction_array_metadata,
    build_temporal_arrays,
    build_temporal_metadata,
)


def _temporal(raw: RawPrediction):
    arrays = build_temporal_arrays(raw)
    return arrays, build_temporal_metadata(
        raw, arrays=arrays, event_summary={}, arrays_artifact_path=None
    )


def test_contiguous_timeline_is_detected(raw_prediction: RawPrediction) -> None:
    _, meta = _temporal(raw_prediction)
    assert meta.timeline_is_contiguous
    assert meta.n_time_points == 6
    assert meta.tr_seconds == 1.0
    assert meta.sampling_rate_hz == 1.0
    assert meta.first_segment_start_seconds == 0.0
    assert meta.last_segment_end_seconds == 6.0


def test_gappy_timeline_is_detected(gappy_prediction: RawPrediction) -> None:
    """Dropped segments must never be reported as a contiguous timeline."""
    _, meta = _temporal(gappy_prediction)
    assert not meta.timeline_is_contiguous
    assert meta.segments_were_filtered
    assert meta.n_segments_kept == 5
    # Covered time is the sum of kept segments, not the span they straddle.
    assert meta.covered_seconds == 5.0
    assert meta.last_segment_end_seconds == 9.0


def test_row_index_is_not_time_when_segments_are_dropped(
    gappy_prediction: RawPrediction,
) -> None:
    """The regression this whole design exists to prevent."""
    arrays, _ = _temporal(gappy_prediction)
    starts = arrays["segment_start_seconds"]
    naive = np.arange(len(starts)) * gappy_prediction.tr_seconds
    assert not np.allclose(starts, naive)
    assert starts[3] == 7.0 and naive[3] == 3.0


def test_stimulus_aligned_output_is_not_shifted_again(
    raw_prediction: RawPrediction,
) -> None:
    """TRIBE already compensated the lag, so segment start IS stimulus time.

    Training applies the offset as `start = ta.start - offset` to the fMRI
    recording, pairing stimulus at t with BOLD measured at t+offset. Subtracting
    the offset again here would misattribute every response to content 5 s earlier.
    """
    arrays, meta = _temporal(raw_prediction)
    assert meta.hemodynamic_offset_seconds == 5.0
    assert meta.output_is_stimulus_aligned is True

    np.testing.assert_allclose(
        arrays["stimulus_time_seconds"], arrays["segment_start_seconds"]
    )
    np.testing.assert_allclose(
        arrays["bold_acquisition_time_seconds"], arrays["segment_start_seconds"] + 5.0
    )
    # The offset is metadata plus derived columns; the raw matrix is untouched.
    assert meta.hemodynamic_offset_applied_to_raw is False


def test_non_aligned_backend_shifts_content_time_backwards() -> None:
    """A backend that does NOT compensate the lag needs the opposite convention."""
    raw = RawPrediction(
        array=np.zeros((2, 16), dtype=np.float32),
        segments=[
            SegmentRecord(index=0, start_seconds=10.0, duration_seconds=1.0),
            SegmentRecord(index=1, start_seconds=11.0, duration_seconds=1.0),
        ],
        tr_seconds=1.0,
        surface_space="fsaverage5",
        vertices_per_hemisphere=8,
        hemodynamic_offset_seconds=5.0,
        output_is_stimulus_aligned=False,
    )
    arrays, _ = _temporal(raw)
    np.testing.assert_allclose(arrays["stimulus_time_seconds"], [5.0, 6.0])
    np.testing.assert_allclose(arrays["bold_acquisition_time_seconds"], [10.0, 11.0])


def test_content_time_never_precedes_the_stimulus(raw_prediction: RawPrediction) -> None:
    """Regression: an earlier build subtracted the offset and produced t = -5 s."""
    arrays, meta = _temporal(raw_prediction)
    assert float(arrays["stimulus_time_seconds"].min()) >= 0.0
    assert meta.notes == ()


def test_temporal_arrays_align_with_prediction_rows(raw_prediction: RawPrediction) -> None:
    arrays, _ = _temporal(raw_prediction)
    for values in arrays.values():
        assert values.shape[0] == raw_prediction.array.shape[0]


def test_cortical_ranges_split_hemispheres(raw_prediction: RawPrediction) -> None:
    meta = build_cortical_metadata(raw_prediction)
    assert meta.surface_space == "fsaverage5"
    assert meta.vertex_count == 16
    assert meta.vertices_per_hemisphere == 8
    assert meta.hemisphere_index_ranges == {"left": (0, 8), "right": (8, 16)}
    assert meta.hemisphere_order == ("left", "right")


def test_cortical_ranges_flag_unmapped_columns() -> None:
    """A width that the hemispheres do not cover must be surfaced, not hidden."""
    raw = RawPrediction(
        array=np.zeros((2, 20), dtype=np.float32),
        segments=[SegmentRecord(index=0, start_seconds=0.0, duration_seconds=1.0)] * 2,
        tr_seconds=1.0,
        surface_space="fsaverage5",
        vertices_per_hemisphere=8,
    )
    meta = build_cortical_metadata(raw)
    assert "WARNING" in (meta.mapping_notes or "")


def test_prediction_array_metadata_describes_the_axes(raw_prediction: RawPrediction) -> None:
    from pathlib import Path

    meta = build_prediction_array_metadata(raw_prediction, artifact_path=Path("predictions.npy"))
    assert meta.axis_names == ("time", "vertex")
    assert meta.shape == (6, 16)
    assert meta.temporal_samples == 6
    assert meta.cortical_features == 16
    assert meta.is_raw_model_output
    # No physical unit is published upstream, so none is claimed.
    assert meta.units is None


def test_non_physical_content_time_is_flagged() -> None:
    """Safety net: if a convention ever yields pre-stimulus times, say so loudly."""
    raw = RawPrediction(
        array=np.zeros((2, 16), dtype=np.float32),
        segments=[
            SegmentRecord(index=0, start_seconds=0.0, duration_seconds=1.0),
            SegmentRecord(index=1, start_seconds=1.0, duration_seconds=1.0),
        ],
        tr_seconds=1.0,
        surface_space="fsaverage5",
        vertices_per_hemisphere=8,
        hemodynamic_offset_seconds=5.0,
        output_is_stimulus_aligned=False,  # wrong convention for this backend
    )
    _, meta = _temporal(raw)
    assert meta.notes
    assert any("non-physical" in note for note in meta.notes)


class TestFilteredButCompleteTimeline:
    """A low kept/total ratio does not by itself mean a gappy timeline.

    Measured on a real 10 s run: TRIBE reported 10 of 100 segments kept while
    the 10 kept rows tiled 0-10 s with no gap at all. The ratio counts the
    backend's internal TR sub-segment enumeration across batches, not distinct
    stimulus time points.
    """

    @staticmethod
    def _raw(total: int, starts: list[float]) -> RawPrediction:
        return RawPrediction(
            array=np.zeros((len(starts), 4), dtype=np.float32),
            segments=[
                SegmentRecord(index=i, start_seconds=s, duration_seconds=1.0, n_events=3)
                for i, s in enumerate(starts)
            ],
            tr_seconds=1.0,
            surface_space="fsaverage5",
            vertices_per_hemisphere=2,
            segments_were_filtered=True,
            n_segments_total=total,
        )

    def test_a_complete_tiling_is_called_out(self) -> None:
        raw = self._raw(100, [float(i) for i in range(10)])
        meta = build_temporal_metadata(
            raw,
            arrays=build_temporal_arrays(raw),
            event_summary={},
            arrays_artifact_path=None,
        )
        assert meta.timeline_is_contiguous
        assert any("no gap" in note for note in meta.notes)
        assert any("not stimulus time" in note for note in meta.notes)

    def test_a_genuinely_gappy_timeline_gets_no_such_note(self) -> None:
        raw = self._raw(100, [0.0, 1.0, 7.0, 8.0])
        meta = build_temporal_metadata(
            raw,
            arrays=build_temporal_arrays(raw),
            event_summary={},
            arrays_artifact_path=None,
        )
        assert not meta.timeline_is_contiguous
        assert not any("no gap" in note for note in meta.notes)
