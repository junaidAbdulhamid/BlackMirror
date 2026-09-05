"""Strict, availability-aware comparison of versioned Phase 4 feature artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blackmirror.comparison.alignment import align_observed_times

#: Which availability mask governs each feature.
#:
#: This gate is load-bearing, not decorative. An unanalysed stream does not
#: produce NaN — `build_feature_matrix` starts from zeros, so a run with no
#: video reports `motion = 0.0`, which is finite. Without a gate the comparison
#: would subtract 0.0 from 0.0 and report a real, measured difference of zero
#: for content that was never looked at.
#:
#: Every feature must appear here; see `_require_complete_gates`.
AVAILABILITY_FOR_PREFIX = {
    "motion": "visual_available",
    "motion_structural": "visual_available",
    "luminance_shift": "visual_available",
    "brightness": "visual_available",
    "brightness_p90": "visual_available",
    "contrast": "visual_available",
    "saturation": "visual_available",
    "entropy": "visual_available",
    "audio_energy": "audio_available",
    "spectral_centroid": "audio_available",
    "spectral_flatness": "audio_available",
    "speech_activity": "audio_available",
    "speech_present": "content_event_available",
    "music_present": "content_event_available",
    "text_present": "content_event_available",
    "cta_present": "content_event_available",
    "object_count": "content_event_available",
    "shot_change": "visual_available",
}

#: Columns that ARE the availability masks, and so gate themselves.
AVAILABILITY_COLUMNS = frozenset(AVAILABILITY_FOR_PREFIX.values())


def _require_complete_gates(names: tuple[str, ...]) -> None:
    """Refuse to compare features whose availability is unknown.

    A hand-maintained mapping silently goes stale the moment a feature is added
    upstream, and the failure is invisible: the new column is simply compared
    without a gate and an unanalysed stream reads as a measured zero. Measured
    on this codebase, three features (`motion_structural`, `luminance_shift`,
    `brightness_p90`) were added to Phase 4 and were ungated here. Failing loudly
    is the only way this stays correct as features are added.
    """
    ungated = [
        name
        for name in names
        if name not in AVAILABILITY_FOR_PREFIX and name not in AVAILABILITY_COLUMNS
    ]
    if ungated:
        raise ValueError(
            "content feature(s) have no availability gate and cannot be compared "
            f"safely: {', '.join(sorted(ungated))}. Add them to "
            "AVAILABILITY_FOR_PREFIX in blackmirror.comparison.content."
        )


@dataclass(frozen=True)
class ContentVariant:
    schema_version: str
    analysis_version: str
    models: dict[str, str]
    configuration: dict[str, float | int | str | bool | None]
    feature_names: tuple[str, ...]
    times: NDArray[np.floating]
    matrix: NDArray[np.floating]


@dataclass(frozen=True)
class ContentDifference:
    times: NDArray[np.float64]
    signed_delta: NDArray[np.float64]
    jointly_available: NDArray[np.bool_]
    feature_names: tuple[str, ...]
    coverage_fraction: NDArray[np.float64]


def compare_content_features(
    reference: ContentVariant,
    candidate: ContentVariant,
    *,
    tolerance_seconds: float = 1e-6,
) -> ContentDifference:
    """Compare only jointly available, identically versioned feature columns."""
    gates = (
        ("schema_version", reference.schema_version, candidate.schema_version),
        ("analysis_version", reference.analysis_version, candidate.analysis_version),
        ("models", reference.models, candidate.models),
        ("configuration", reference.configuration, candidate.configuration),
        ("feature_names", reference.feature_names, candidate.feature_names),
    )
    mismatches = [name for name, left, right in gates if left != right]
    if mismatches:
        raise ValueError("content artifacts are not comparable: " + ", ".join(mismatches))
    names = reference.feature_names
    if not names or len(set(names)) != len(names):
        raise ValueError("content feature names must be nonempty and unique")
    _require_complete_gates(names)
    for label, variant in (("reference", reference), ("candidate", candidate)):
        if np.asarray(variant.matrix).shape != (len(variant.times), len(names)):
            raise ValueError(f"{label} content matrix shape does not match its contract")
    aligned = align_observed_times(
        reference.times, candidate.times, tolerance_seconds=tolerance_seconds
    )
    left = np.asarray(reference.matrix, dtype=np.float64)[aligned.reference_indices]
    right = np.asarray(candidate.matrix, dtype=np.float64)[aligned.candidate_indices]
    available = np.isfinite(left) & np.isfinite(right)
    index = {name: position for position, name in enumerate(names)}
    for feature, availability in AVAILABILITY_FOR_PREFIX.items():
        if feature in index and availability in index:
            available[:, index[feature]] &= (left[:, index[availability]] > 0.5) & (
                right[:, index[availability]] > 0.5
            )
    delta = np.where(available, right - left, np.nan)
    return ContentDifference(
        times=aligned.times,
        signed_delta=delta,
        jointly_available=available,
        feature_names=names,
        coverage_fraction=available.mean(axis=0),
    )
