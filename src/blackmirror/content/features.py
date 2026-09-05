"""Content feature matrix.

WHAT IT DOES
    Resamples every content signal onto one uniform grid, producing X in R^(T x F).

WHY A MATRIX WHEN WE ALREADY HAVE EVENTS
    Events are uneven in length, which is right for reading but wrong for maths.
    Correlation, lag analysis and any future statistical model need a regular
    grid where column f at row t is comparable to column f at row t+1.

    Both representations are kept. Events are the human-facing truth; the matrix
    is the machine-facing projection of it, and the resampling method is stated
    rather than hidden.

IN:  dense signals + fused events
OUT: (matrix, feature_names, times)
"""

from __future__ import annotations

import numpy as np

from blackmirror.content.audio.signals import AudioSignals
from blackmirror.content.schemas import ContentEvent
from blackmirror.content.visual.motion import VisualSignals, resample_to
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Uniform grid rate for the feature matrix. 4 Hz matches the visual sampler and
#: is comfortably finer than the 1 s TR of the neural predictions, so alignment
#: never has to upsample the content side.
FEATURE_RATE_HZ = 4.0

#: Column order is fixed and versioned: Phase 5 will diff two matrices, and that
#: is only meaningful if column i means the same thing in both.
FEATURE_NAMES: tuple[str, ...] = (
    "motion",
    # Motion decomposed: a fade changes every pixel without anything moving, so
    # the two halves answer different questions and correlate differently.
    "motion_structural",
    "luminance_shift",
    "brightness",
    "brightness_p90",
    "contrast",
    "saturation",
    "entropy",
    "audio_energy",
    "spectral_centroid",
    "spectral_flatness",
    "speech_activity",
    "speech_present",
    "music_present",
    "text_present",
    "shot_change",
    "cta_present",
    "object_count",
    "visual_available",
    "audio_available",
    "content_event_available",
)


def build_feature_matrix(
    *,
    duration: float,
    visual_signals: VisualSignals,
    audio_signals: AudioSignals,
    events: list[ContentEvent],
    shot_times: list[float],
    rate_hz: float = FEATURE_RATE_HZ,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Resample everything onto a uniform grid."""
    if duration <= 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), FEATURE_NAMES, np.zeros(0)

    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and greater than zero")
    count = max(1, int(np.ceil(duration * rate_hz)))
    times = (np.arange(count, dtype=np.float32) + 0.5) / rate_hz
    times = np.minimum(times, np.float32(duration))
    matrix = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float32)
    column = {name: index for index, name in enumerate(FEATURE_NAMES)}

    # Continuous signals: linear interpolation onto the grid.
    if len(visual_signals):
        matrix[:, column["visual_available"]] = (
            (times >= visual_signals.times[0]) & (times <= visual_signals.times[-1])
        )
        for name, series in (
            ("motion", visual_signals.motion),
            ("motion_structural", visual_signals.motion_structural),
            ("luminance_shift", visual_signals.luminance_shift),
            ("brightness", visual_signals.brightness),
            ("brightness_p90", visual_signals.brightness_p90),
            ("contrast", visual_signals.contrast),
            ("saturation", visual_signals.saturation),
            ("entropy", visual_signals.entropy),
        ):
            matrix[:, column[name]] = resample_to(visual_signals.times, series, times)

    if len(audio_signals):
        matrix[:, column["audio_available"]] = (
            (times >= audio_signals.times[0]) & (times <= audio_signals.times[-1])
        )
        for name, series in (
            ("audio_energy", audio_signals.energy),
            ("spectral_centroid", audio_signals.spectral_centroid),
            ("spectral_flatness", audio_signals.spectral_flatness),
            ("speech_activity", audio_signals.speech_activity),
        ):
            matrix[:, column[name]] = resample_to(audio_signals.times, series, times)

    # Categorical signals: nearest-containing event, never interpolated — a
    # half-present CTA is not a meaningful quantity.
    for row, time in enumerate(times):
        event = _event_at(events, float(time))
        if event is None:
            continue
        matrix[row, column["content_event_available"]] = 1.0
        matrix[row, column["speech_present"]] = 1.0 if event.speech_text else 0.0
        matrix[row, column["music_present"]] = (
            1.0 if event.audio_description == "music" else 0.0
        )
        matrix[row, column["text_present"]] = 1.0 if event.on_screen_text else 0.0
        matrix[row, column["cta_present"]] = (
            1.0
            if event.event_type.value == "cta"
            or event.language_features.get("is_call_to_action")
            else 0.0
        )
        matrix[row, column["object_count"]] = float(len(event.objects))

    # Shot changes are impulses at cut times, not a continuous quantity.
    for cut in shot_times:
        index = int(np.clip(round(cut * rate_hz - 0.5), 0, count - 1))
        matrix[index, column["shot_change"]] = 1.0

    logger.info("Built content feature matrix %s", matrix.shape)
    return matrix, FEATURE_NAMES, times


def _event_at(events: list[ContentEvent], time: float) -> ContentEvent | None:
    for event in events:
        if event.start_time <= time < event.end_time:
            return event
    return None


def summarise_columns(
    matrix: np.ndarray, names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Per-column statistics, for the API and for sanity checks."""
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        if index >= matrix.shape[1]:
            break
        column = matrix[:, index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        summary[name] = {
            "mean": round(float(finite.mean()), 5),
            "std": round(float(finite.std()), 5),
            "min": round(float(finite.min()), 5),
            "max": round(float(finite.max()), 5),
        }
    return summary
