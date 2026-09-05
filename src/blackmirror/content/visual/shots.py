"""Shot boundary detection.

WHAT IT DOES
    Finds the cuts — the points where the camera changes — and returns the
    continuous takes between them.

WHY IT MATTERS
    A shot is the natural atomic unit of video. Editing pace, keyframe
    selection, and the fusion interval grid are all built on it, and "cuts per
    minute" is one of the most diffable properties between two variants.

HOW IT WORKS
    Three PySceneDetect detectors are run and their boundaries unioned, because
    each is blind to what the others see:

    - ContentDetector measures frame-to-frame HSV change. Excellent on hard
      cuts. Blind to fades: two consecutive near-black frames barely differ, so
      the change signal it keys on never appears.
    - AdaptiveDetector compares each frame's change against a rolling window,
      which catches progressive changes a fixed threshold rides over.
    - ThresholdDetector tracks absolute luminance and fires on fades to/from
      black, which is exactly ContentDetector's blind spot.

    Measured on the Sintel trailer: ContentDetector alone found 11 boundaries
    and missed fades at 0.58 s, 10.83 s and 15.67 s. Frame luminance at those
    points is 0.0-7.1 out of 255 against a 46.9 clip mean, confirming a real
    fade-to-black rather than detector noise.

    Boundaries within MERGE_TOLERANCE of each other are the same event seen by
    two detectors and are merged, keeping the earliest time — a fade's true
    start precedes the frame at which content change becomes measurable.

IN:  a video file
OUT: ShotSegment[] with times and frame indices
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from blackmirror.content.schemas import ShotSegment, TransitionKind
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: ContentDetector's HSV-change threshold. 27 is the upstream default, tuned for
#: broadcast-style content; lower is more sensitive and over-segments on motion.
DEFAULT_THRESHOLD = 27.0

#: Shots shorter than this are almost always detector noise on a dissolve.
MIN_SHOT_SECONDS = 0.25

#: Two detectors firing within this many seconds describe the same transition.
#: A fade takes ~0.5 s at 24 fps, so its luminance floor and the point content
#: change becomes measurable can sit a few frames apart.
MERGE_TOLERANCE = 0.40


def detect_shots(
    video_path: Path,
    *,
    duration: float,
    threshold: float = DEFAULT_THRESHOLD,
    min_shot_seconds: float = MIN_SHOT_SECONDS,
) -> tuple[list[ShotSegment], list[str]]:
    """Detect shots. Returns (shots, warnings).

    Always returns at least one shot spanning the whole stimulus: a single
    continuous take is a legitimate result, and downstream code should never
    have to special-case an empty list.
    """
    warnings: list[str] = []
    try:
        from scenedetect import (
            AdaptiveDetector,
            ContentDetector,
            SceneManager,
            ThresholdDetector,
            open_video,
        )
    except ImportError:
        warnings.append("PySceneDetect not installed; treating the stimulus as one shot.")
        return [_whole(duration)], warnings

    # Each detector is run over its own pass of the video. SceneManager holds
    # per-detector state, so they cannot share one.
    passes: dict[str, tuple[TransitionKind, object]] = {
        "content": (TransitionKind.CUT, lambda: ContentDetector(threshold=threshold)),
        "adaptive": (TransitionKind.GRADUAL, lambda: AdaptiveDetector()),
        "threshold": (TransitionKind.FADE, lambda: ThresholdDetector()),
    }

    found: list[_Boundary] = []
    failures: list[str] = []
    for name, (kind, factory) in passes.items():
        try:
            video = open_video(str(video_path))
            manager = SceneManager()
            manager.add_detector(factory())  # type: ignore[operator]
            manager.detect_scenes(video, show_progress=False)
            scenes = manager.get_scene_list()
        except Exception as exc:
            failures.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        for start, _end in scenes:
            found.append(_Boundary(start.get_seconds(), start.get_frames(), kind, name))

    if failures:
        warnings.append(f"Shot detector(s) failed and were skipped: {'; '.join(failures)}.")
    if len(failures) == len(passes):
        warnings.append("All shot detectors failed; using one shot.")
        return [_whole(duration)], warnings

    boundaries = _unify(found, duration)
    if not boundaries:
        # No cuts found is a real answer for short or static content.
        return [_whole(duration)], warnings

    fades = sum(1 for b in boundaries if b.kind is TransitionKind.FADE)
    content_only = {
        round(b.seconds, 2) for b in found if b.detector == "content"
    }
    recovered = [
        round(b.seconds, 2)
        for b in boundaries
        if b.kind is not TransitionKind.START
        and not any(abs(b.seconds - c) <= MERGE_TOLERANCE for c in content_only)
    ]
    if recovered:
        warnings.append(
            f"{len(recovered)} gradual transition(s) at {recovered} were found only by "
            f"the fade/adaptive detectors; frame-difference detection alone misses these."
        )

    raw = _spans(boundaries, duration)
    merged = _merge_short(raw, min_shot_seconds)
    if len(merged) < len(raw):
        warnings.append(
            f"Merged {len(raw) - len(merged)} shot(s) shorter than {min_shot_seconds}s "
            f"into their neighbours."
        )

    shots = []
    for i, (start, end, frame_start, frame_end, kind, detectors) in enumerate(merged):
        # Duration is derived from the ROUNDED boundaries, not rounded
        # independently: rounding each separately can leave them inconsistent
        # by a millisecond, which the schema validator rejects.
        rounded_start = round(start, 3)
        rounded_end = round(end, 3)
        shots.append(
            ShotSegment(
                index=i,
                start_time=rounded_start,
                end_time=rounded_end,
                duration=round(max(0.0, rounded_end - rounded_start), 3),
                frame_start=frame_start,
                frame_end=frame_end,
                transition_in=kind,
                detected_by=detectors,
            )
        )
    logger.info("Detected %d shot(s), %d beginning with a fade", len(shots), fades)
    return shots, warnings


@dataclass(frozen=True)
class _Boundary:
    """One detector's opinion that a shot starts here."""

    seconds: float
    frame: int
    kind: TransitionKind
    detector: str


def _unify(found: list[_Boundary], duration: float) -> list[_Boundary]:
    """Collapse near-coincident boundaries from different detectors into one.

    The transition type is decided by *which* detectors fired, not by a ranking
    of their opinions, because each detector's evidence means something
    different:

    - ThresholdDetector fires on absolute luminance crossing a floor. That is
      positive physical evidence of a fade, so it wins outright.
    - ContentDetector fires on a single-frame content spike, which is what a
      hard cut is.
    - AdaptiveDetector fires on both, so it corroborates rather than
      discriminates. Only when it fires *alone* — the others seeing nothing — is
      the transition progressive enough to call GRADUAL.

    Ranking the kinds instead would mislabel every hard cut: AdaptiveDetector
    also fires on cuts, so a "gradual beats cut" rule turned the measured
    124.6-luminance hard cut at 16.50 s in the Sintel trailer into "gradual".
    """
    usable = sorted(
        (b for b in found if 0.0 <= b.seconds < duration), key=lambda b: b.seconds
    )
    if not usable:
        return []

    groups: list[list[_Boundary]] = [[usable[0]]]
    for boundary in usable[1:]:
        if boundary.seconds - groups[-1][0].seconds <= MERGE_TOLERANCE:
            groups[-1].append(boundary)
        else:
            groups.append([boundary])

    unified: list[_Boundary] = []
    for group in groups:
        # Earliest time in the group: a fade begins before the frame at which
        # content change becomes measurable.
        first = group[0]
        detectors = tuple(sorted({b.detector for b in group}))
        kind = _kind_from_detectors(detectors)
        unified.append(_Boundary(first.seconds, first.frame, kind, ",".join(detectors)))

    # The stimulus always starts at 0, and that opening is not a transition.
    if unified[0].seconds <= MERGE_TOLERANCE:
        head = unified[0]
        unified[0] = _Boundary(0.0, 0, TransitionKind.START, head.detector)
    else:
        unified.insert(0, _Boundary(0.0, 0, TransitionKind.START, "implicit"))
    return unified


def _kind_from_detectors(detectors: tuple[str, ...]) -> TransitionKind:
    """Name the transition from the evidence that found it. See :func:`_unify`."""
    if "threshold" in detectors:
        return TransitionKind.FADE
    if "content" in detectors:
        return TransitionKind.CUT
    if "adaptive" in detectors:
        return TransitionKind.GRADUAL
    return TransitionKind.UNKNOWN


def _spans(
    boundaries: list[_Boundary], duration: float
) -> list[tuple[float, float, int, int | None, TransitionKind, tuple[str, ...]]]:
    """Turn boundary instants into the half-open spans between them."""
    spans = []
    for i, boundary in enumerate(boundaries):
        end = boundaries[i + 1].seconds if i + 1 < len(boundaries) else duration
        frame_end = boundaries[i + 1].frame if i + 1 < len(boundaries) else None
        spans.append(
            (
                boundary.seconds,
                min(end, duration),
                boundary.frame,
                frame_end,
                boundary.kind,
                tuple(boundary.detector.split(",")),
            )
        )
    return spans


_Span = tuple[float, float, int, "int | None", TransitionKind, tuple[str, ...]]


def _merge_short(raw: list[_Span], minimum: float) -> list[_Span]:
    """Absorb sub-threshold shots into the previous one.

    Dropping them would leave gaps in the timeline; merging keeps the shot list
    a complete, contiguous partition of the stimulus. The surviving shot keeps
    the *earlier* shot's transition, since that is how it began.
    """
    if not raw:
        return raw
    merged = [raw[0]]
    for span in raw[1:]:
        start, end, _frame_start, frame_end, _kind, _detectors = span
        previous = merged[-1]
        if end - start < minimum:
            merged[-1] = (previous[0], end, previous[2], frame_end, previous[4], previous[5])
        else:
            merged.append(span)
    # A short final shot has no successor to absorb it, so fold it backwards.
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < minimum:
        last = merged.pop()
        previous = merged[-1]
        merged[-1] = (previous[0], last[1], previous[2], last[3], previous[4], previous[5])
    return merged


def _whole(duration: float) -> ShotSegment:
    end = round(duration, 3)
    return ShotSegment(
        index=0,
        start_time=0.0,
        end_time=end,
        duration=end,
        frame_start=0,
        frame_end=None,
        transition_in=TransitionKind.START,
        detected_by=("none",),
    )


def editing_pace(shots: list[ShotSegment], duration: float) -> dict[str, float]:
    """Editing-rhythm metrics derived from the shot list.

    Mean shot duration is the classic measure; the median is reported too
    because one long establishing shot skews the mean badly on short content.
    """
    if not shots or duration <= 0:
        return {}
    lengths = sorted(shot.duration for shot in shots)
    middle = len(lengths) // 2
    median = (
        lengths[middle]
        if len(lengths) % 2
        else (lengths[middle - 1] + lengths[middle]) / 2.0
    )
    return {
        "shot_count": float(len(shots)),
        "mean_shot_duration": sum(lengths) / len(lengths),
        "median_shot_duration": median,
        # Cuts, not shots: N shots means N-1 transitions.
        "cuts_per_minute": max(0.0, len(shots) - 1) / (duration / 60.0),
    }
