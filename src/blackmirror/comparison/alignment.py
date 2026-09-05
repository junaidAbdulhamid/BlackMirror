"""Deterministic exact and explicitly opt-in timeline alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AlignedTimeline:
    times: NDArray[np.float64]
    reference_indices: NDArray[np.int64]
    candidate_indices: NDArray[np.int64]


@dataclass(frozen=True)
class InterpolatedTimeline:
    """Reference observations and candidate interpolation brackets.

    Target timestamps are observed reference samples inside the candidate's
    closed time range. Thus interpolation never extrapolates or invents a
    reference timestamp.
    """

    times: NDArray[np.float64]
    reference_indices: NDArray[np.int64]
    candidate_left_indices: NDArray[np.int64]
    candidate_right_indices: NDArray[np.int64]
    candidate_right_weights: NDArray[np.float64]
    candidate_observed_mask: NDArray[np.bool_]


def align_observed_times(
    reference: NDArray[np.floating],
    candidate: NDArray[np.floating],
    *,
    tolerance_seconds: float = 1e-6,
) -> AlignedTimeline:
    """Return unique observed timestamp pairs within tolerance.

    Any timestamp with multiple possible partners is rejected rather than resolved
    by an arbitrary tie-break. Inputs must be finite and strictly increasing.
    """
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("timeline tolerance must be nonnegative")
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    for name, values in (("reference", left), ("candidate", right)):
        if values.ndim != 1 or not len(values):
            raise ValueError(f"{name} timeline must be a non-empty vector")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} timeline contains nonfinite timestamps")
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} timeline must be strictly increasing with no duplicates")
    reference_indices_list: list[int] = []
    candidate_indices_list: list[int] = []
    low = high = 0
    previous_candidate = -1
    for reference_index, timestamp in enumerate(left):
        while low < len(right) and right[low] < timestamp - tolerance_seconds:
            low += 1
        high = max(high, low)
        while high < len(right) and right[high] <= timestamp + tolerance_seconds:
            high += 1
        if high - low > 1:
            raise ValueError("timeline alignment is ambiguous within the requested tolerance")
        if high - low == 1:
            if low == previous_candidate:
                raise ValueError("timeline alignment is ambiguous within the requested tolerance")
            reference_indices_list.append(reference_index)
            candidate_indices_list.append(low)
            previous_candidate = low
    reference_indices = np.asarray(reference_indices_list, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices_list, dtype=np.int64)
    if not len(reference_indices):
        raise ValueError("timelines have no observed samples in common")
    paired_times = (left[reference_indices] + right[candidate_indices]) / 2
    return AlignedTimeline(
        times=paired_times,
        reference_indices=reference_indices,
        candidate_indices=candidate_indices,
    )


def align_linear_candidate_to_reference(
    reference: NDArray[np.floating],
    candidate: NDArray[np.floating],
    *,
    tolerance_seconds: float = 1e-6,
    max_gap_seconds: float,
) -> InterpolatedTimeline:
    """Align to observed reference timestamps with linear candidate interpolation.

    Matching within tolerance is treated as observed. Other targets are
    bracketed by two candidate observations. Duplicate/unsorted/nonfinite inputs
    and extrapolation-only timelines are rejected.
    """
    # Reuse exact alignment validation while allowing timelines with no exact match.
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("timeline tolerance must be nonnegative")
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    for name, values in (("reference", left), ("candidate", right)):
        if values.ndim != 1 or not len(values):
            raise ValueError(f"{name} timeline must be a non-empty vector")
        if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} timeline must be finite and strictly increasing")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("maximum interpolation gap must be finite and positive")
    ref_indices = np.flatnonzero(
        (left >= right[0] - tolerance_seconds) & (left <= right[-1] + tolerance_seconds)
    ).astype(np.int64)
    if not len(ref_indices):
        raise ValueError("timelines have no overlapping interpolation domain")
    targets = left[ref_indices]
    insertion = np.searchsorted(right, targets, side="left")
    left_indices = np.clip(insertion - 1, 0, len(right) - 1).astype(np.int64)
    right_indices = np.clip(insertion, 0, len(right) - 1).astype(np.int64)
    observed = np.zeros(len(targets), dtype=bool)
    for index, (target, position) in enumerate(zip(targets, insertion, strict=True)):
        neighbors = {max(0, position - 1), min(len(right) - 1, position)}
        matches = [item for item in neighbors if abs(right[item] - target) <= tolerance_seconds]
        if len(matches) > 1:
            raise ValueError("timeline alignment is ambiguous within the requested tolerance")
        if len(matches) == 1:
            left_indices[index] = right_indices[index] = matches[0]
            observed[index] = True
        elif position == 0 or position == len(right):
            raise ValueError("timeline interpolation would require extrapolation")
        elif right[position] - right[position - 1] > max_gap_seconds:
            raise ValueError("candidate interpolation gap exceeds configured maximum")
    denominator = right[right_indices] - right[left_indices]
    weights = np.divide(
        targets - right[left_indices],
        denominator,
        out=np.zeros_like(targets),
        where=denominator != 0,
    )
    return InterpolatedTimeline(
        times=targets,
        reference_indices=ref_indices,
        candidate_left_indices=left_indices,
        candidate_right_indices=right_indices,
        candidate_right_weights=weights,
        candidate_observed_mask=observed,
    )


def interpolate_rows(
    values: NDArray[np.floating], alignment: InterpolatedTimeline
) -> NDArray[np.float64]:
    """Interpolate rows while propagating unavailable/nonfinite endpoints."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("interpolated values must be a matrix")
    low = array[alignment.candidate_left_indices]
    high = array[alignment.candidate_right_indices]
    weight = alignment.candidate_right_weights[:, None]
    return low * (1.0 - weight) + high * weight
