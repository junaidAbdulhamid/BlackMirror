"""Transcript acquisition.

WHY PHASE 1'S EVENTS ARE THE PRIMARY SOURCE
    TRIBE's preprocessing already transcribed the stimulus with WhisperX and
    persisted word-level timings to `events.parquet`. Re-transcribing here would
    risk analysing text the *model never saw* — a slightly different transcript
    means the content analysis and the neural prediction are describing subtly
    different stimuli, which would quietly undermine every alignment claim in
    Phase 4 and every comparison in Phase 5.

    So: reuse Phase 1's events when present, and fall back to running WhisperX
    ourselves only when they are absent (reduced-modality runs, or a stimulus
    analysed outside an inference run). The source is always recorded.

IN:  a run's events.parquet, or a media file
OUT: TranscriptSegment[] with word timings
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from blackmirror.content.schemas import (
    AnalysisSource,
    Provenance,
    TranscriptSegment,
    WordTiming,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Phase 1 writes these event types; `Sentence` carries utterance text and
#: `Word` carries the per-word timing we want to preserve.
SENTENCE_TYPES = {"Sentence"}
WORD_TYPE = "Word"


def transcript_from_run_events(
    events_path: Path, *, duration: float
) -> tuple[list[TranscriptSegment], list[str]]:
    """Build transcript segments from Phase 1's events table.

    Returns an empty list (not an error) when the run has no speech — a silent
    stimulus is a legitimate outcome, and TRIBE drops the text extractor there.
    """
    warnings: list[str] = []
    if not events_path.exists():
        return [], []

    try:
        import pandas as pd

        frame = pd.read_parquet(events_path)
    except Exception as exc:
        return [], [f"Could not read Phase 1 events ({type(exc).__name__}: {exc})."]

    if "type" not in frame.columns:
        return [], ["Phase 1 events table has no 'type' column."]

    sentences = frame[frame["type"].isin(SENTENCE_TYPES)]
    words = frame[frame["type"] == WORD_TYPE] if WORD_TYPE in set(frame["type"]) else None

    if sentences.empty:
        # Some runs carry words without sentence grouping.
        if words is None or words.empty:
            return [], []
        warnings.append("No Sentence events; grouping words into a single segment.")
        return _segments_from_words_only(words, duration), warnings

    provenance = Provenance(
        source=AnalysisSource.TRIBE_EVENTS,
        model_id="whisperx-large-v3 (via TRIBE preprocessing)",
        notes="The exact transcript the prediction model consumed.",
    )

    segments: list[TranscriptSegment] = []
    for index, row in enumerate(sentences.sort_values("start").itertuples(index=False)):
        start = _float(getattr(row, "start", None))
        length = _float(getattr(row, "duration", None)) or 0.0
        text = _text(getattr(row, "text", None))
        if start is None or not text:
            continue
        end = min(duration, start + length)
        segments.append(
            TranscriptSegment(
                index=index,
                start_time=round(start, 3),
                end_time=round(end, 3),
                text=text,
                words=_words_within(words, start, end),
                provenance=provenance,
            )
        )

    if segments:
        logger.info("Loaded %d transcript segment(s) from Phase 1 events", len(segments))
    return segments, warnings


def _segments_from_words_only(words: Any, duration: float) -> list[TranscriptSegment]:
    timings = _word_timings(words)
    if not timings:
        return []
    text = " ".join(word.text for word in timings)
    return [
        TranscriptSegment(
            index=0,
            start_time=timings[0].start_time,
            end_time=min(duration, timings[-1].end_time),
            text=text,
            words=tuple(timings),
            provenance=Provenance(
                source=AnalysisSource.TRIBE_EVENTS,
                model_id="whisperx-large-v3 (via TRIBE preprocessing)",
            ),
        )
    ]


def _words_within(words: Any, start: float, end: float) -> tuple[WordTiming, ...]:
    if words is None or len(words) == 0:
        return ()
    return tuple(
        timing
        for timing in _word_timings(words)
        # Half-open on the left, inclusive slack on the right so a word ending
        # exactly on the sentence boundary is not dropped.
        if timing.start_time >= start - 1e-6 and timing.start_time < end + 1e-6
    )


def _word_timings(words: Any) -> list[WordTiming]:
    timings: list[WordTiming] = []
    for row in words.sort_values("start").itertuples(index=False):
        start = _float(getattr(row, "start", None))
        length = _float(getattr(row, "duration", None)) or 0.0
        text = _text(getattr(row, "text", None))
        if start is None or not text:
            continue
        timings.append(
            WordTiming(text=text, start_time=round(start, 3), end_time=round(start + length, 3))
        )
    return timings


def transcribe_with_whisperx(
    audio_path: Path, *, duration: float, language: str = "en"
) -> tuple[list[TranscriptSegment], list[str]]:
    """Fallback transcription via `uvx whisperx`.

    Mirrors how Phase 1 invokes it — a subprocess in uv's own environment, not
    an in-process import — including the Apple Silicon compute-type and
    `torch.load` workarounds that Phase 1 documented.
    """
    warnings: list[str] = []
    uvx = shutil.which("uvx")
    if uvx is None:
        return [], ["`uvx` not on PATH; cannot run the WhisperX fallback."]

    import sys

    # float16 is unsupported by CTranslate2 on Apple Silicon CPU (Phase 1 finding).
    compute_type = "int8" if sys.platform == "darwin" else "float16"
    environment = dict(os.environ)
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    with tempfile.TemporaryDirectory() as output_dir:
        command = [
            uvx, "whisperx", str(audio_path),
            "--model", "large-v3",
            "--language", language,
            "--device", "cpu",
            "--compute_type", compute_type,
            "--batch_size", "16",
            "--output_dir", output_dir,
            "--output_format", "json",
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, env=environment, timeout=3600
            )
        except subprocess.SubprocessError as exc:
            return [], [f"WhisperX fallback failed to start: {exc}"]

        if result.returncode != 0:
            return [], [f"WhisperX fallback failed: {(result.stderr or '').strip()[:400]}"]

        produced = list(Path(output_dir).glob("*.json"))
        if not produced:
            return [], ["WhisperX produced no output."]
        payload = json.loads(produced[0].read_text(encoding="utf-8"))

    provenance = Provenance(
        source=AnalysisSource.WHISPERX,
        model_id=f"whisperx large-v3 ({compute_type})",
        notes="Fallback transcript; the prediction model may have seen a different one.",
    )
    warnings.append(
        "Transcript came from the WhisperX fallback rather than Phase 1 events, so it "
        "is not guaranteed identical to the text the prediction model consumed."
    )

    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(payload.get("segments", [])):
        start = _float(segment.get("start"))
        end = _float(segment.get("end"))
        text = _text(segment.get("text"))
        if start is None or end is None or not text:
            continue
        words = tuple(
            WordTiming(
                text=_text(word.get("word")) or "",
                start_time=round(float(word["start"]), 3),
                end_time=round(float(word.get("end", word["start"])), 3),
            )
            for word in segment.get("words", [])
            if "start" in word and _text(word.get("word"))
        )
        segments.append(
            TranscriptSegment(
                index=index,
                start_time=round(start, 3),
                end_time=round(min(end, duration), 3),
                text=text,
                words=words,
                provenance=provenance,
            )
        )
    return segments, warnings


def speech_intervals(segments: list[TranscriptSegment]) -> list[tuple[float, float]]:
    """Speech spans, for the audio segmenter to treat as authoritative."""
    return [(segment.start_time, segment.end_time) for segment in segments]


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
