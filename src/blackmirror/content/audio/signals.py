"""Audio signal analysis.

WHAT IT DOES
    Turns a waveform into interpretable time-series — how loud, how tonal, how
    noisy — and segments it into speech / music / silence.

WHY NOT librosa
    Everything here is a short-time statistic over framed audio: RMS, zero-
    crossing rate, spectral centroid, spectral flatness. numpy + scipy compute
    them in a few lines with no numba dependency, and writing them out makes the
    definitions auditable rather than hidden behind a library default.

WHAT THESE SIGNALS ARE NOT
    `energy` is signal power. It is not "intensity", "excitement", or any
    affective quantity. The music/speech split prefers AudioSet tagging and
    falls back to a DSP heuristic
    with stated confidence, not a trained classifier.

IN:  mono 16 kHz PCM WAV
OUT: aligned time-series + AudioSegment[]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blackmirror.content.schemas import (
    AnalysisSource,
    AudioSegment,
    AudioSegmentType,
    Provenance,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: 25 ms windows every 10 ms — the standard speech-processing framing. Long
#: enough to be spectrally meaningful, short enough to resolve phonemes.
FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010

#: Silence is defined relative to the clip's own loudness, not an absolute dB
#: floor, so quiet and loud mixes are treated alike.
SILENCE_RELATIVE_THRESHOLD = 0.08
MIN_SEGMENT_SECONDS = 0.30


@dataclass
class AudioSignals:
    """Dense per-frame audio features on a shared time base."""

    times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    zero_crossing_rate: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    spectral_centroid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    spectral_flatness: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    speech_activity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "audio_times": self.times,
            "audio_energy": self.energy,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_centroid": self.spectral_centroid,
            "spectral_flatness": self.spectral_flatness,
            "speech_activity": self.speech_activity,
        }

    def __len__(self) -> int:
        return int(self.times.size)


def compute_audio_signals(wav_path: Path) -> tuple[AudioSignals, list[str]]:
    """Frame the waveform and compute short-time features."""
    warnings: list[str] = []
    try:
        import soundfile as sf
    except ImportError:
        return AudioSignals(), ["soundfile not installed; audio analysis unavailable."]

    try:
        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as exc:
        return AudioSignals(), [f"Could not read audio ({type(exc).__name__}: {exc})."]

    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        return AudioSignals(), ["Audio track is empty."]

    frame_length = max(1, round(FRAME_SECONDS * sample_rate))
    hop_length = max(1, round(HOP_SECONDS * sample_rate))
    if samples.size < frame_length:
        return AudioSignals(), ["Audio is shorter than one analysis frame."]

    frames = _frame(samples, frame_length, hop_length)
    times = (np.arange(frames.shape[0]) * hop_length + frame_length / 2.0) / sample_rate

    energy = np.sqrt(np.maximum((frames**2).mean(axis=1), 0.0))
    zcr = (np.abs(np.diff(np.sign(frames), axis=1)) > 0).mean(axis=1)

    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)

    centroid = (spectrum * freqs).sum(axis=1) / spectrum.sum(axis=1)
    # Flatness = geometric/arithmetic mean of the spectrum. Near 1 for noise,
    # near 0 for a strongly tonal (harmonic) signal.
    log_spectrum = np.log(spectrum)
    flatness = np.exp(log_spectrum.mean(axis=1)) / spectrum.mean(axis=1)

    speech = _speech_activity(energy, zcr, flatness)

    signals = AudioSignals(
        times=times.astype(np.float32),
        energy=energy.astype(np.float32),
        zero_crossing_rate=zcr.astype(np.float32),
        spectral_centroid=centroid.astype(np.float32),
        spectral_flatness=flatness.astype(np.float32),
        speech_activity=speech.astype(np.float32),
    )
    logger.info("Computed audio signals over %d frames", len(signals))
    return signals, warnings


def _frame(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Strided framing — a view, not a copy."""
    count = 1 + (samples.size - frame_length) // hop_length
    stride = samples.strides[0]
    return np.lib.stride_tricks.as_strided(
        samples, shape=(count, frame_length), strides=(stride * hop_length, stride)
    )


def _speech_activity(
    energy: np.ndarray, zcr: np.ndarray, flatness: np.ndarray
) -> np.ndarray:
    """Voice activity as a soft [0,1] score.

    Speech is loud, spectrally structured (low flatness), and has moderate
    zero-crossing rates. This is a classical heuristic, not a trained VAD: it
    will confuse sung vocals with speech and may miss whispering.
    """
    if energy.size == 0:
        return energy
    loud = energy / (np.percentile(energy, 95) + 1e-9)
    loud = np.clip(loud, 0.0, 1.0)
    tonal = np.clip(1.0 - flatness / (np.percentile(flatness, 90) + 1e-9), 0.0, 1.0)
    moderate_zcr = np.clip(1.0 - np.abs(zcr - 0.12) / 0.25, 0.0, 1.0)
    return np.clip(0.5 * loud + 0.3 * tonal + 0.2 * moderate_zcr, 0.0, 1.0)


def segment_audio(
    signals: AudioSignals,
    duration: float,
    *,
    speech_intervals: list[tuple[float, float]] | None = None,
) -> tuple[list[AudioSegment], list[str]]:
    """Partition audio into speech / music / silence.

    This is the **fallback** path, used only when audio tagging is disabled or
    produced nothing; :func:`~blackmirror.content.audio.tagging.segments_from_tags`
    is preferred and is the one that can tell singing from speech.

    A transcript is authoritative for speech where one exists — a real ASR beats
    any heuristic, and using it keeps these segments consistent with the words
    shown elsewhere in the UI. Otherwise speech comes from a DSP heuristic that
    keys on loudness, tonality and zero-crossing rate, on all three of which
    sung vocals score exactly like speech. That confusion is reported as a
    warning rather than left for the reader to discover.

    The remaining audio is split by loudness: quiet stretches become SILENCE,
    audible non-speech becomes MUSIC, otherwise OTHER.
    """
    warnings: list[str] = []
    if len(signals) == 0 or duration <= 0:
        return [], ["No audio signal to segment."]

    times = signals.times
    energy = signals.energy
    threshold = float(np.percentile(energy, 95)) * SILENCE_RELATIVE_THRESHOLD

    labels = np.where(energy > threshold, 1, 0)  # 1 = audible
    speech_mask: np.ndarray = np.zeros(times.size, dtype=bool)
    if speech_intervals:
        for start, end in speech_intervals:
            speech_mask |= (times >= start) & (times < end)
    else:
        speech_mask = np.asarray(signals.speech_activity > 0.55, dtype=bool)
        warnings.append(
            "No transcript and no audio tagging available; speech regions came from the "
            "DSP heuristic, which cannot distinguish sung vocals from speech."
        )

    # Tonal, audible, non-speech audio is treated as music. This is a proxy for
    # music, not a detection of it — speech is tonal too.
    music = signals.spectral_flatness < float(np.percentile(signals.spectral_flatness, 60))
    warnings.append(
        "Music regions came from a spectral-flatness heuristic, not audio tagging."
    )

    kinds = np.full(times.size, AudioSegmentType.OTHER.value, dtype=object)
    kinds[labels == 0] = AudioSegmentType.SILENCE.value
    kinds[(labels == 1) & music] = AudioSegmentType.MUSIC.value
    kinds[speech_mask] = AudioSegmentType.SPEECH.value

    segments = _merge_short_segments(_runs_to_segments(times, kinds, energy, duration))
    logger.info("Segmented audio into %d segment(s)", len(segments))
    return segments, warnings


def _runs_to_segments(
    times: np.ndarray, kinds: np.ndarray, energy: np.ndarray, duration: float
) -> list[AudioSegment]:
    """Collapse a per-frame label array into contiguous runs."""
    if times.size == 0:
        return []
    segments: list[AudioSegment] = []
    start_index = 0
    for index in range(1, times.size + 1):
        ended = index == times.size or kinds[index] != kinds[start_index]
        if not ended:
            continue
        # Frame timestamps are centres. The first classified run covers the
        # stimulus from t=0; otherwise every audio analysis has an artificial
        # uncovered prefix of half a frame.
        start = 0.0 if start_index == 0 else float(times[start_index])
        end = float(times[index]) if index < times.size else duration
        window = energy[start_index:index]
        segments.append(
            AudioSegment(
                segment_type=AudioSegmentType(kinds[start_index]),
                start_time=round(max(0.0, start), 3),
                end_time=round(min(duration, end), 3),
                mean_energy=float(window.mean()) if window.size else 0.0,
                confidence=None,  # a heuristic reports no calibrated confidence
                provenance=Provenance(
                    source=AnalysisSource.SIGNAL_DSP,
                    model_id="rms+zcr+spectral-flatness",
                    notes="DSP heuristic; speech regions prefer ASR intervals when available.",
                ),
            )
        )
        start_index = index
    return segments


def _merge_short_segments(segments: list[AudioSegment]) -> list[AudioSegment]:
    """Absorb brief classifier flicker into a neighbour without creating gaps."""
    merged = list(segments)
    while len(merged) > 1:
        index = next(
            (
                position
                for position, segment in enumerate(merged)
                if segment.end_time - segment.start_time < MIN_SEGMENT_SECONDS
            ),
            None,
        )
        if index is None:
            break
        if index == 0:
            neighbour = 1
        elif index == len(merged) - 1:
            neighbour = index - 1
        else:
            left_duration = merged[index - 1].end_time - merged[index - 1].start_time
            right_duration = merged[index + 1].end_time - merged[index + 1].start_time
            neighbour = index - 1 if left_duration >= right_duration else index + 1

        absorbed = merged[index]
        keeper = merged[neighbour]
        absorbed_duration = absorbed.end_time - absorbed.start_time
        keeper_duration = keeper.end_time - keeper.start_time
        total_duration = absorbed_duration + keeper_duration
        replacement = AudioSegment(
            segment_type=keeper.segment_type,
            start_time=min(absorbed.start_time, keeper.start_time),
            end_time=max(absorbed.end_time, keeper.end_time),
            mean_energy=(
                absorbed.mean_energy * absorbed_duration
                + keeper.mean_energy * keeper_duration
            )
            / total_duration,
            confidence=keeper.confidence,
            provenance=keeper.provenance,
        )
        low, high = sorted((index, neighbour))
        merged[low : high + 1] = [replacement]
    return merged
