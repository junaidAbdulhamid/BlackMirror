"""Turn a backend's :class:`RawPrediction` into the standardized contract pieces.

This is where the two facts Phase 2 depends on are established:

1. **Temporal alignment.** Which stimulus moment does prediction row *i*
   describe? TRIBE v2 can drop rows, so this cannot be `i * TR`.
2. **Cortical mapping.** Which vertex of which hemisphere does column *j* colour?

No numerical transformation of the prediction matrix happens here. Derived
*time* arrays are produced (they are metadata, not response data) and are stored
alongside the raw matrix, never merged into it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blackmirror.inference.backend import RawPrediction
from blackmirror.schemas.prediction import (
    CorticalMetadata,
    PredictionArrayMetadata,
    TemporalMetadata,
)

#: Tolerance (seconds) for deciding whether segments tile the timeline evenly.
_CONTIGUITY_TOLERANCE = 1e-3


def build_temporal_arrays(raw: RawPrediction) -> dict[str, np.ndarray]:
    """Build the per-row timing arrays persisted as ``temporal.npz``.

    Two derived time columns are provided. Both are metadata, stored separately
    from the raw prediction, which is never shifted.

    ``stimulus_time_seconds``
        The content time a row describes. When the model already performed the
        hemodynamic alignment during training (TRIBE v2 does), this equals
        ``segment_start_seconds``. This is the column to sync playback on.

    ``bold_acquisition_time_seconds``
        When the corresponding BOLD signal would physically be measured, i.e.
        ``stimulus_time + hemodynamic_offset``. Useful only if comparing against
        real scanner timings; it is NOT a playback clock.
    """
    starts = np.array([s.start_seconds for s in raw.segments], dtype=np.float64)
    durations = np.array([s.duration_seconds for s in raw.segments], dtype=np.float64)
    n_events = np.array([s.n_events for s in raw.segments], dtype=np.int64)
    offset = float(raw.hemodynamic_offset_seconds)

    if raw.output_is_stimulus_aligned:
        stimulus_time = starts.copy()
        acquisition_time = starts + offset
    else:
        # The model did NOT compensate the lag: a row's segment time is when the
        # BOLD would be measured, so the driving content is `offset` earlier.
        stimulus_time = starts - offset
        acquisition_time = starts.copy()

    return {
        "segment_start_seconds": starts,
        "segment_duration_seconds": durations,
        "segment_n_events": n_events,
        "stimulus_time_seconds": stimulus_time,
        "bold_acquisition_time_seconds": acquisition_time,
    }


TEMPORAL_ARRAY_KEYS: dict[str, str] = {
    "segment_start_seconds": (
        "Raw start time of each prediction row's segment on the stimulus timeline, "
        "as reported by the model. Row i covers "
        "[segment_start_seconds[i], +segment_duration_seconds[i])."
    ),
    "segment_duration_seconds": "Duration each prediction row covers (normally the TR).",
    "segment_n_events": (
        "Number of stimulus events inside each segment. Zero only when the backend "
        "was configured to keep event-free segments."
    ),
    "stimulus_time_seconds": (
        "DERIVED: the stimulus content time each predicted response describes. When "
        "output_is_stimulus_aligned is true (TRIBE v2), this equals "
        "segment_start_seconds, because the model already compensated the hemodynamic "
        "lag during training. THIS is the column to synchronise content playback on."
    ),
    "bold_acquisition_time_seconds": (
        "DERIVED: stimulus_time_seconds + hemodynamic_offset_seconds — roughly when the "
        "corresponding BOLD signal would physically be measured in a scanner. For "
        "comparison against real acquisition timings only; NOT a playback clock."
    ),
}


def _timeline_is_contiguous(starts: np.ndarray, tr_seconds: float) -> bool:
    """True when rows tile the timeline at exactly TR spacing with no gaps."""
    if starts.size <= 1:
        return True
    gaps = np.diff(starts)
    return bool(np.all(np.abs(gaps - tr_seconds) <= _CONTIGUITY_TOLERANCE))


def build_temporal_metadata(
    raw: RawPrediction,
    *,
    arrays: dict[str, np.ndarray],
    event_summary: dict[str, object],
    arrays_artifact_path: Path | None,
) -> TemporalMetadata:
    """Summarize temporal alignment for the manifest."""
    starts = arrays["segment_start_seconds"]
    durations = arrays["segment_duration_seconds"]

    contiguous = _timeline_is_contiguous(starts, raw.tr_seconds)
    first_start = float(starts.min()) if starts.size else None
    last_end = float((starts + durations).max()) if starts.size else None
    covered = float(durations.sum()) if durations.size else None

    notes: list[str] = []
    derived = arrays["stimulus_time_seconds"]
    if derived.size and float(derived.min()) < 0.0:
        # A stimulus starts at t=0, so a negative content time is non-physical and
        # means the alignment convention is wrong for this backend. Surfaced rather
        # than silently corrected: guessing would misattribute every predicted
        # response to the wrong content event.
        notes.append(
            f"stimulus_time_seconds reaches {float(derived.min()):.2f}s, before the "
            f"stimulus begins, which is non-physical. The hemodynamic alignment "
            f"convention for this backend is wrong. Use segment_start_seconds and "
            f"treat content-to-response attribution as unresolved."
        )

    if (
        raw.segments_were_filtered
        and raw.n_segments_total
        and raw.n_segments_total > len(raw.segments)
        and contiguous
        and covered is not None
        and first_start is not None
        and abs(covered - ((last_end or 0.0) - first_start)) <= _CONTIGUITY_TOLERANCE
    ):
        # A kept/total ratio well below 1 looks like a badly gappy timeline. When
        # the kept rows in fact tile their span without a gap, say so, because
        # the ratio counts enumerated sub-segments and not distinct time points.
        notes.append(
            f"{len(raw.segments)} of {raw.n_segments_total} enumerated segments were "
            f"kept, but the kept rows tile "
            f"[{first_start:.2f}s, {last_end:.2f}s) with no gap. The ratio counts the "
            f"backend's internal sub-segment enumeration, not stimulus time; no part "
            f"of this span is missing."
        )

    return TemporalMetadata(
        n_time_points=int(raw.array.shape[0]) if raw.array.ndim >= 1 else 0,
        tr_seconds=float(raw.tr_seconds),
        sampling_rate_hz=1.0 / float(raw.tr_seconds),
        timeline_is_contiguous=contiguous,
        segments_were_filtered=raw.segments_were_filtered,
        n_segments_total=raw.n_segments_total,
        n_segments_kept=len(raw.segments),
        first_segment_start_seconds=first_start,
        last_segment_end_seconds=last_end,
        covered_seconds=covered,
        hemodynamic_offset_seconds=float(raw.hemodynamic_offset_seconds),
        hemodynamic_offset_verified=raw.hemodynamic_offset_verified,
        hemodynamic_offset_source=raw.hemodynamic_offset_source,
        output_is_stimulus_aligned=raw.output_is_stimulus_aligned,
        arrays_artifact_path=arrays_artifact_path,
        array_keys=dict(TEMPORAL_ARRAY_KEYS),
        event_summary=dict(event_summary),
        notes=tuple(notes),
    )


def build_cortical_metadata(
    raw: RawPrediction,
    *,
    mesh_artifact_dir: Path | None = None,
) -> CorticalMetadata:
    """Describe how prediction columns map onto the cortical surface."""
    per_hemi = int(raw.vertices_per_hemisphere)
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for hemisphere in raw.hemisphere_order:
        ranges[hemisphere] = (cursor, cursor + per_hemi)
        cursor += per_hemi

    total_columns = int(raw.array.shape[1]) if raw.array.ndim >= 2 else 0

    notes = (
        f"Prediction column j maps to vertex (j - start) of the hemisphere whose "
        f"half-open range [start, end) contains j. Vertex indices are positions in the "
        f"{raw.surface_space} surface mesh; export it with `blackmirror export-mesh` "
        f"to obtain matching coordinates and faces."
    )
    if cursor != total_columns:
        notes += (
            f" WARNING: hemisphere ranges cover {cursor} columns but the matrix has "
            f"{total_columns}. Trailing columns are unmapped."
        )

    return CorticalMetadata(
        surface_space=raw.surface_space,
        vertex_count=total_columns,
        vertices_per_hemisphere=per_hemi,
        hemisphere_order=tuple(raw.hemisphere_order),
        hemisphere_index_ranges=ranges,
        is_surface_based=True,
        includes_subcortex=raw.includes_subcortex,
        mesh_artifact_dir=mesh_artifact_dir,
        medial_wall_handling=raw.medial_wall_handling,
        atlas=raw.atlas,
        mapping_notes=notes,
    )


def build_prediction_array_metadata(
    raw: RawPrediction,
    *,
    artifact_path: Path,
) -> PredictionArrayMetadata:
    """Describe the raw prediction matrix as stored."""
    shape = tuple(int(dim) for dim in raw.array.shape)
    return PredictionArrayMetadata(
        shape=shape,
        dtype=str(raw.array.dtype),
        axis_names=("time", "vertex"),
        temporal_samples=shape[0] if len(shape) > 0 else 0,
        cortical_features=shape[1] if len(shape) > 1 else 0,
        units=raw.units,
        normalization=raw.normalization,
        semantics=raw.semantics,
        is_raw_model_output=True,
        artifact_path=artifact_path,
    )
