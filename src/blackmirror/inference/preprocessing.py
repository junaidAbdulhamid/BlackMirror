"""Stimulus -> TRIBE event representation.

TRIBE ships its own preprocessing pipeline (audio extraction, chunking, WhisperX
word-level transcription, sentence/context annotation). We **wrap** it rather
than reimplement it: duplicating a research model's preprocessing is the fastest
way to silently produce inputs the model was never trained on.

What we add is the part TRIBE does not persist — a modality-neutral summary of
the resulting events, so Phase 3+ can correlate content events with predicted
neural changes without re-running an expensive pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from blackmirror.errors import PreprocessingError
from blackmirror.schemas.stimulus import MediaType, StimulusInput
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Maps our media type onto the keyword argument TRIBE's loader expects.
_MEDIA_TYPE_TO_KWARG: dict[MediaType, str] = {
    MediaType.VIDEO: "video_path",
    MediaType.AUDIO: "audio_path",
    MediaType.TEXT: "text_path",
}


def build_events_dataframe(model: Any, stimulus: StimulusInput, *, transcribe: bool = True) -> Any:
    """Produce TRIBE's events DataFrame for a stimulus.

    Args:
        model: A loaded ``TribeModel``.
        stimulus: The validated stimulus.
        transcribe: When False, skip WhisperX word-level transcription and build
            audio/video events only. This is a **reduced-modality** path: TRIBE
            is trimodal, so dropping the text stream changes what the model sees.
            Provided because WhisperX is the least portable part of the pipeline
            (CUDA-oriented, undeclared dependency). Recorded in the event summary.

    Raises:
        PreprocessingError: TRIBE's pipeline failed.
    """
    kwarg = _MEDIA_TYPE_TO_KWARG[stimulus.media_type]

    if transcribe:
        try:
            return model.get_events_dataframe(**{kwarg: str(stimulus.path)})
        except Exception as exc:
            raise PreprocessingError(
                f"TRIBE event extraction failed for {stimulus.filename} "
                f"({stimulus.media_type.value}): {type(exc).__name__}: {exc}"
            ) from exc

    if stimulus.media_type is MediaType.TEXT:
        raise PreprocessingError(
            "A text stimulus cannot be processed with transcription disabled: TRIBE "
            "synthesises speech from the text and transcribes it back to obtain word "
            "timings, so the transcription step is the pipeline."
        )

    logger.warning(
        "Transcription disabled: building audio/video events only. TRIBE is trimodal, "
        "so the text stream is absent and predictions are NOT comparable with a "
        "full-modality run."
    )
    try:
        import pandas as pd
        from tribev2.demo_utils import get_audio_and_text_events

        event = {
            "type": "Audio" if stimulus.media_type is MediaType.AUDIO else "Video",
            "filepath": str(stimulus.path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    except Exception as exc:
        raise PreprocessingError(
            f"TRIBE audio-only event extraction failed for {stimulus.filename}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def summarize_events(events: Any, *, stimulus: StimulusInput, transcribed: bool) -> dict[str, Any]:
    """Describe the event structure in a form later phases can consume.

    Deliberately does not embed the events themselves — the full table is
    persisted separately as Parquet.
    """
    summary: dict[str, Any] = {
        "backend": "tribe_v2",
        "source_media_type": stimulus.media_type.value,
        "transcription_enabled": transcribed,
        "n_events": 0,
        "event_types": {},
        "columns": [],
    }
    if events is None:
        return summary

    try:
        summary["n_events"] = len(events)
        summary["columns"] = [str(c) for c in events.columns]

        if "type" in events.columns:
            counts = Counter(str(value) for value in events["type"].tolist())
            summary["event_types"] = dict(sorted(counts.items()))
            summary["modalities_present"] = sorted(counts)

        if "start" in events.columns:
            starts = events["start"].astype(float)
            summary["first_event_start_seconds"] = float(starts.min())
            summary["last_event_start_seconds"] = float(starts.max())
        if "duration" in events.columns:
            durations = events["duration"].astype(float)
            summary["total_event_duration_seconds"] = float(durations.sum())
            summary["mean_event_duration_seconds"] = float(durations.mean())
        if "text" in events.columns:
            words = events["text"].dropna()
            summary["n_text_events"] = len(words)
    except Exception as exc:
        logger.warning("Could not fully summarize events (%s: %s)", type(exc).__name__, exc)
        summary["summary_error"] = f"{type(exc).__name__}: {exc}"

    return summary
