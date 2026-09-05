"""Turning a temporal scope into concrete sample indices, per variant.

WHY THIS IS ITS OWN STEP
    Two variants can place the same content event at different timestamps. An
    objective that says "during the CTA" must therefore resolve to a *different*
    absolute interval in each variant, and comparing them on a single shared
    window would measure a CTA in one variant against whatever the other happened
    to be showing. Resolution is per variant, and the interval that was actually
    used is persisted with the score.

HALF-OPEN INTERVALS
    A sample at time t belongs to window [start, end) when start <= t < end.
    This matches Phase 2 and Phase 4, so an event boundary never assigns one
    sample to two adjacent windows.

WHAT IT REFUSES TO DO
    Invent a window. A content event that a variant does not contain is an
    error, not an empty interval: silently scoring zero samples would let a
    variant "win" a CTA objective by having no CTA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blackmirror.scoring.schemas import ResolvedWindow, TemporalScope, TemporalScopeType


class WindowResolutionError(ValueError):
    """The scope cannot be resolved against this variant."""


@dataclass(frozen=True)
class ContentEventInterval:
    """The minimum a resolver needs from a Phase 4 event."""

    event_type: str
    start_time: float
    end_time: float


def resolve_window(
    scope: TemporalScope,
    *,
    times: NDArray[np.floating],
    duration_seconds: float,
    events: list[ContentEventInterval] | None = None,
) -> ResolvedWindow:
    """Resolve `scope` against one variant's own timeline."""
    axis = np.asarray(times, dtype=np.float64)
    if axis.ndim != 1 or not axis.size:
        raise WindowResolutionError("variant has no analytics time axis")
    if not np.isfinite(axis).all():
        raise WindowResolutionError("variant time axis contains non-finite timestamps")

    start, end, source = _bounds(scope, axis, duration_seconds, events)
    if end <= start:
        raise WindowResolutionError(
            f"resolved window [{start:.3f}, {end:.3f}) is empty for scope {scope.type.value}"
        )
    indices = np.flatnonzero((axis >= start) & (axis < end))
    if not indices.size:
        period = float(np.median(np.diff(axis))) if axis.size > 1 else float("nan")
        hint = ""
        if scope.type in (
            TemporalScopeType.CONTENT_EVENT,
            TemporalScopeType.CONTENT_EVENT_RELATIVE_WINDOW,
        ):
            hint = (
                f" This window is {end - start:.3f}s long against a sampling period of "
                f"{period:.3f}s, so it can fall between samples. Content events are detected "
                f"at sub-second resolution while predictions are sampled at the TR. Set "
                f"skip_events_without_samples=True to consider only occurrences that contain "
                f"a sample, or widen the scope with a relative window."
            )
        raise WindowResolutionError(
            f"window [{start:.3f}, {end:.3f}) contains no prediction samples; the variant's "
            f"samples nearest that interval are at "
            f"{axis[max(0, int(np.searchsorted(axis, start)) - 1)]:.3f}s and "
            f"{axis[min(len(axis) - 1, int(np.searchsorted(axis, start)))]:.3f}s.{hint}"
        )
    return ResolvedWindow(
        start_seconds=round(float(start), 6),
        end_seconds=round(float(end), 6),
        sample_indices=tuple(int(value) for value in indices),
        sample_count=int(indices.size),
        source=source,
    )


def _bounds(
    scope: TemporalScope,
    axis: NDArray[np.float64],
    duration_seconds: float,
    events: list[ContentEventInterval] | None,
) -> tuple[float, float, str]:
    if scope.type is TemporalScopeType.FULL_STIMULUS:
        # Open the end just past the last sample so it is included by [start, end).
        return float(axis[0]), float(axis[-1]) + 1e-9, "full_stimulus"

    if scope.type is TemporalScopeType.ABSOLUTE_TIME_WINDOW:
        assert scope.start_seconds is not None and scope.end_seconds is not None
        return scope.start_seconds, scope.end_seconds, "absolute_time_window"

    if scope.type is TemporalScopeType.NORMALIZED_TIME_WINDOW:
        assert scope.start_fraction is not None and scope.end_fraction is not None
        if not np.isfinite(duration_seconds) or duration_seconds <= 0:
            raise WindowResolutionError("normalized window needs a positive stimulus duration")
        return (
            scope.start_fraction * duration_seconds,
            scope.end_fraction * duration_seconds,
            f"normalized_time_window[{scope.start_fraction:g}-{scope.end_fraction:g}]",
        )

    match = _select_event(scope, events, axis)
    if scope.type is TemporalScopeType.CONTENT_EVENT:
        return (
            match.start_time,
            match.end_time,
            f"content_event:{scope.event_type}[{scope.event_selection}]",
        )

    assert scope.offset_start_seconds is not None and scope.offset_end_seconds is not None
    # Offsets are anchored to the event's START, so "2 s before to 3 s after"
    # means the same thing regardless of how long the event itself runs.
    return (
        match.start_time + scope.offset_start_seconds,
        match.start_time + scope.offset_end_seconds,
        (
            f"content_event_relative:{scope.event_type}[{scope.event_selection}]"
            f"{scope.offset_start_seconds:+g}..{scope.offset_end_seconds:+g}s"
        ),
    )


def _contains_sample(
    event: ContentEventInterval, axis: NDArray[np.float64], scope: TemporalScope
) -> bool:
    """Whether the interval this event resolves to holds a prediction sample."""
    if scope.type is TemporalScopeType.CONTENT_EVENT_RELATIVE_WINDOW:
        assert scope.offset_start_seconds is not None and scope.offset_end_seconds is not None
        start = event.start_time + scope.offset_start_seconds
        end = event.start_time + scope.offset_end_seconds
    else:
        start, end = event.start_time, event.end_time
    return bool(np.any((axis >= start) & (axis < end)))


def _select_event(
    scope: TemporalScope,
    events: list[ContentEventInterval] | None,
    axis: NDArray[np.float64],
) -> ContentEventInterval:
    if events is None:
        raise WindowResolutionError(
            f"scope needs content event {scope.event_type!r} but this variant has no "
            f"content analysis; run `blackmirror analyze-content` for it first"
        )
    matches = [event for event in events if event.event_type == scope.event_type]
    if not matches:
        present = sorted({event.event_type for event in events})
        raise WindowResolutionError(
            f"this variant contains no {scope.event_type!r} event, so the objective cannot "
            f"be measured on it. Present event types: {present or 'none'}. Scoring it as an "
            f"empty window would let a variant rank on an event it does not have."
        )
    if scope.skip_events_without_samples:
        resolvable = [event for event in matches if _contains_sample(event, axis, scope)]
        if not resolvable:
            shortest = min(e.end_time - e.start_time for e in matches)
            raise WindowResolutionError(
                f"none of the {len(matches)} {scope.event_type!r} event(s) in this variant "
                f"contain a prediction sample; the shortest is {shortest:.3f}s against a "
                f"sampling period of about "
                f"{float(np.median(np.diff(axis))) if axis.size > 1 else float('nan'):.3f}s"
            )
        matches = resolvable
    if scope.event_selection == "first":
        return min(matches, key=lambda event: (event.start_time, event.end_time))
    if scope.event_selection == "last":
        return max(matches, key=lambda event: (event.start_time, event.end_time))
    # "longest", with earliest start breaking ties so the result is deterministic.
    return min(
        matches, key=lambda event: (-(event.end_time - event.start_time), event.start_time)
    )
