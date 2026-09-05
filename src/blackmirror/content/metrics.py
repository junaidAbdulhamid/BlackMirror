"""Quantitative content metrics.

WHAT THESE ARE FOR
    Single numbers that describe a whole stimulus. Phase 5 will diff two of
    these side by side — "Variant B cuts 2.4x faster in the opening" is a claim
    about `cuts_per_minute`, not about anyone's brain.

RULE
    Every metric is computed from persisted evidence or reported as None. A
    metric that cannot be derived is absent, never estimated.
"""

from __future__ import annotations

import numpy as np

from blackmirror.content.audio.signals import AudioSignals
from blackmirror.content.schemas import (
    AudioSegment,
    AudioSegmentType,
    CallToAction,
    ContentEvent,
    ContentMetrics,
    SceneSegment,
    ShotSegment,
    TextOverlay,
    TranscriptSegment,
)
from blackmirror.content.visual.motion import VisualSignals


def compute_metrics(
    *,
    duration: float,
    shots: list[ShotSegment],
    scenes: list[SceneSegment],
    transcript: list[TranscriptSegment],
    audio_segments: list[AudioSegment],
    overlays: list[TextOverlay],
    calls_to_action: list[CallToAction],
    events: list[ContentEvent],
    visual_signals: VisualSignals,
    audio_signals: AudioSignals,
) -> ContentMetrics:
    """Derive the content summary."""
    shot_durations = [shot.duration for shot in shots if shot.duration > 0]
    mean_shot = sum(shot_durations) / len(shot_durations) if shot_durations else None
    median_shot = float(np.median(shot_durations)) if shot_durations else None
    # Cuts, not shots: N shots means N-1 transitions.
    cuts_per_minute = (
        max(0.0, len(shots) - 1) / (duration / 60.0) if duration > 0 and shots else None
    )

    speech_seconds = _covered(
        [(segment.start_time, segment.end_time) for segment in transcript], duration
    )
    silence_seconds = _covered(
        _spans(audio_segments, AudioSegmentType.SILENCE), duration
    )
    music_seconds = _covered(_spans(audio_segments, AudioSegmentType.MUSIC), duration)

    words = [word for segment in transcript for word in segment.words]
    word_count = len(words) or sum(len(segment.text.split()) for segment in transcript)

    return ContentMetrics(
        duration_seconds=round(duration, 3),
        shot_count=len(shots),
        mean_shot_duration=_round(mean_shot),
        median_shot_duration=_round(median_shot),
        cuts_per_minute=_round(cuts_per_minute),
        scene_count=len(scenes),
        speech_fraction=_fraction(speech_seconds, duration),
        silence_fraction=_fraction(silence_seconds, duration),
        music_fraction=_fraction(music_seconds, duration),
        word_count=word_count,
        words_per_minute=(
            round(word_count / (duration / 60.0), 2) if duration > 0 and word_count else None
        ),
        text_overlay_count=len(overlays),
        cta_count=len(calls_to_action),
        first_cta_time=(
            round(min(cta.start_time for cta in calls_to_action), 3)
            if calls_to_action
            else None
        ),
        mean_motion=_signal_mean(visual_signals.motion),
        mean_audio_energy=_signal_mean(audio_signals.energy),
        mean_brightness=_signal_mean(visual_signals.brightness),
        content_event_count=len(events),
    )


def _spans(
    segments: list[AudioSegment], kind: AudioSegmentType
) -> list[tuple[float, float]]:
    return [
        (segment.start_time, segment.end_time)
        for segment in segments
        if segment.segment_type is kind
    ]


def _covered(spans: list[tuple[float, float]], duration: float) -> float:
    """Total time covered by a set of possibly-overlapping intervals.

    Overlaps are merged rather than summed: two overlapping speech segments do
    not mean the stimulus contains more speech than it lasts.
    """
    if not spans or duration <= 0:
        return 0.0
    ordered = sorted((max(0.0, s), min(duration, e)) for s, e in spans if e > s)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def _fraction(seconds: float, duration: float) -> float | None:
    if duration <= 0:
        return None
    return round(min(1.0, max(0.0, seconds / duration)), 4)


def _signal_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    return round(float(finite.mean()), 5) if finite.size else None


def _round(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None
