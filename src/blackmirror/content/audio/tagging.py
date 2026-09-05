"""Audio tagging with a trained model.

WHAT THIS REPLACES
    Music detection was a DSP heuristic: audible, non-speech, low spectral
    flatness. That is a reasonable proxy but it is not a classifier — it has no
    calibrated confidence, it confuses tonal sound design with music, and it
    cannot name anything else that happens in the audio.

THE MODEL
    Audio Spectrogram Transformer fine-tuned on AudioSet
    (`MIT/ast-finetuned-audioset-10-10-0.4593`): 527 labels covering Music,
    Speech, Applause, Laughter, Vehicle, Explosion, Silence and much else.

    AudioSet is **multi-label** — music and speech genuinely co-occur — so the
    head is sigmoid, not softmax, and each probability is independent. Reading
    these as competing classes would be a category error.

WHY WINDOWED
    AST consumes ~10.24 s of audio at a time. Long stimuli are tagged with
    overlapping windows and the per-label probability is tracked over time,
    which is what turns a clip-level tagger into a temporal signal.

IN:  mono 16 kHz waveform
OUT: per-window label probabilities -> music segments + audio events
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blackmirror.content.schemas import (
    AnalysisSource,
    AudioSegment,
    AudioSegmentType,
    Provenance,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

#: AST is trained at this rate and its feature extractor rejects any other.
MODEL_SAMPLE_RATE = 16_000

#: AST's native receptive field. Windows shorter than this are zero-padded by
#: the feature extractor, which is fine for short stimuli.
WINDOW_SECONDS = 10.24

#: How far each window advances. The 10.24 s receptive field is fixed by the
#: model, but *localization* is not: a sound is placed within one hop, so a 5 s
#: hop could only say "somewhere in this 5 s". At 1 s the window overlap is
#: 90%, which costs ~5x the forward passes (measured: 1.5 s -> 7.6 s on a 10 s
#: clip) and buys 5x finer onset placement. Short sounds are still diluted
#: inside the receptive field — that part is not fixable by hopping.
HOP_SECONDS = 1.0

#: Probability above which a label is reported. AudioSet sigmoid outputs are
#: reasonably calibrated; 0.35 keeps confident tags without flooding the list.
LABEL_THRESHOLD = 0.35

#: Labels that describe the *medium* rather than a discrete event. Reported as
#: segments, not as events, so "Music" does not appear as a thing that happened.
MEDIUM_LABELS = frozenset({"Music", "Speech", "Silence"})

#: Broad AudioSet parents that add nothing over their children.
UNINFORMATIVE = frozenset(
    {"Sound effect", "Field recording", "Inside, small room", "Inside, large room or hall",
     "Outside, rural or natural", "Outside, urban or manmade", "Noise", "Environmental noise"}
)


@dataclass(frozen=True)
class AudioTagWindow:
    """Label probabilities for one analysis window."""

    start: float
    end: float
    probabilities: dict[str, float]

    def top(self, count: int = 5) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)[:count]


class AudioTagger:
    """Lazily-loaded AST wrapper, reused across every window of a run."""

    name = MODEL_ID

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model: Any = None
        self._extractor: Any = None
        self._torch: Any = None
        self._unavailable: str | None = None

    def available(self) -> bool:
        return self._load() is not None

    def _load(self) -> Any:
        if self._model is not None or self._unavailable is not None:
            return self._model
        try:
            import torch
            from transformers import ASTFeatureExtractor, ASTForAudioClassification

            self._extractor = ASTFeatureExtractor.from_pretrained(self.model_id)
            self._model = ASTForAudioClassification.from_pretrained(self.model_id).eval()
            self._torch = torch
        except Exception as exc:
            self._unavailable = f"{type(exc).__name__}: {exc}"
            logger.warning("Audio tagger unavailable (%s)", self._unavailable)
        return self._model

    def tag(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
    ) -> list[AudioTagWindow]:
        """Tag overlapping windows across the waveform."""
        if self._load() is None or samples.size == 0:
            return []

        torch = self._torch
        window = int(window_seconds * sample_rate)
        hop = max(1, int(hop_seconds * sample_rate))
        duration = samples.size / sample_rate
        starts = list(range(0, max(1, samples.size - window // 2), hop)) or [0]

        if sample_rate != MODEL_SAMPLE_RATE:
            # Guaranteed to fail in the extractor. Say so once, loudly, rather
            # than per window at debug level where nobody sees it.
            logger.warning(
                "Audio tagging needs %d Hz but received %d Hz; resample before tagging.",
                MODEL_SAMPLE_RATE,
                sample_rate,
            )
            return []

        results: list[AudioTagWindow] = []
        failures: list[str] = []
        for start in starts:
            chunk = samples[start : start + window]
            # A window shorter than ~1 s carries too little spectral evidence.
            if chunk.size < sample_rate // 2:
                continue
            try:
                inputs = self._extractor(
                    chunk, sampling_rate=sample_rate, return_tensors="pt"
                )
                with torch.inference_mode():
                    logits = self._model(**inputs).logits
                # Multi-label: sigmoid per label, never softmax across labels.
                probabilities = torch.sigmoid(logits).squeeze(0)
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                continue

            labels = {
                self._model.config.id2label[index]: round(float(value), 4)
                for index, value in enumerate(probabilities)
                if float(value) >= LABEL_THRESHOLD * 0.5  # keep a margin for smoothing
            }
            results.append(
                AudioTagWindow(
                    start=round(start / sample_rate, 3),
                    end=round(min(duration, (start + window) / sample_rate), 3),
                    probabilities=labels,
                )
            )
        if failures and not results:
            # Silently returning [] here is what hid a sample-rate mismatch:
            # the run reported no audio events and gave no reason.
            logger.warning(
                "Audio tagging produced nothing; every one of %d window(s) failed. First: %s",
                len(failures),
                failures[0],
            )
        elif failures:
            logger.warning("Audio tagging skipped %d window(s): %s", len(failures), failures[0])
        logger.info("Tagged %d audio window(s)", len(results))
        return results


def probability_series(
    windows: list[AudioTagWindow], label: str, times: np.ndarray
) -> np.ndarray:
    """Sample one label's probability onto a time base.

    Overlapping windows are combined by taking the maximum, not the mean: a
    sound present in half a window is present, and averaging would dilute it
    toward the quieter neighbour.
    """
    series = np.zeros(times.size, dtype=np.float32)
    if not windows:
        return series
    for index, time in enumerate(times):
        covering = [w for w in windows if w.start <= time < w.end]
        if not covering:
            covering = [min(windows, key=lambda w: abs((w.start + w.end) / 2 - time))]
        series[index] = max(w.probabilities.get(label, 0.0) for w in covering)
    return series


#: Resolution of the label grid segments are built on. Finer than this buys
#: nothing: the model's own hop is 1 s.
FRAME_SECONDS = 0.25

#: How much more probable Speech must be than Singing before an untranscribed
#: region is called speech rather than music. Sung vocals are voiced speech-like
#: audio, and AudioSet scores both labels on them.
SPEECH_OVER_SINGING = 1.2


def segments_from_tags(
    windows: list[AudioTagWindow],
    *,
    duration: float,
    speech_intervals: list[tuple[float, float]] | None = None,
    threshold: float = LABEL_THRESHOLD,
) -> tuple[list[AudioSegment], list[str]]:
    """Derive speech / music / silence segments from model probabilities.

    Segments are built by labelling a dense time grid and collapsing equal
    neighbours into runs, **not** by emitting one segment per window. Windows
    overlap by 90% at the default hop, so one-segment-per-window returned a pile
    of near-duplicate 10 s spans rather than a partition of the timeline.

    ``Singing`` is scored against ``Speech`` rather than ignored. A sung vocal
    is voiced, loud and tonal, so it lights up ``Speech`` too; treating it as
    speech would put song lyrics in the same bucket as dialogue.

    A transcript still wins where one exists: ASR gives exact word boundaries,
    while AST gives a probability over a 10.24 s receptive field.
    """
    warnings: list[str] = []
    if not windows:
        return [], ["Audio tagger produced no windows; falling back to DSP segmentation."]

    provenance = Provenance(
        source=AnalysisSource.AUDIO_TAGGING,
        model_id=MODEL_ID,
        notes="AudioSet multi-label probabilities (sigmoid), windowed.",
    )

    count = max(1, round(duration / FRAME_SECONDS))
    times = np.arange(count, dtype=np.float64) * FRAME_SECONDS
    music = probability_series(windows, "Music", times)
    speech = probability_series(windows, "Speech", times)
    singing = probability_series(windows, "Singing", times)
    silence = probability_series(windows, "Silence", times)

    kinds: list[AudioSegmentType] = []
    scores: list[float] = []
    sung_frames = 0
    for i in range(count):
        m, sp, si, sl = (
            float(music[i]),
            float(speech[i]),
            float(singing[i]),
            float(silence[i]),
        )
        is_speech = sp >= threshold and sp >= SPEECH_OVER_SINGING * si
        if si >= threshold and not is_speech:
            sung_frames += 1
        if sl >= threshold and m < threshold and not is_speech:
            kinds.append(AudioSegmentType.SILENCE)
            scores.append(sl)
        elif is_speech and sp >= m:
            kinds.append(AudioSegmentType.SPEECH)
            scores.append(sp)
        elif m >= threshold or si >= threshold:
            kinds.append(AudioSegmentType.MUSIC)
            scores.append(max(m, si))
        elif is_speech:
            kinds.append(AudioSegmentType.SPEECH)
            scores.append(sp)
        else:
            kinds.append(AudioSegmentType.OTHER)
            scores.append(max(m, sp, sl))

    if sung_frames:
        warnings.append(
            f"{sung_frames} frame(s) scored as sung vocals rather than speech and are "
            f"segmented as music. Speech-versus-singing is a model judgement."
        )

    segments = _collapse_runs(kinds, scores, times, duration, provenance)

    # Transcript intervals override for speech: exact beats windowed.
    if speech_intervals:
        for start, end in speech_intervals:
            segments.append(
                AudioSegment(
                    segment_type=AudioSegmentType.SPEECH,
                    start_time=round(start, 3),
                    end_time=round(min(duration, end), 3),
                    mean_energy=0.0,
                    confidence=None,
                    provenance=Provenance(
                        source=AnalysisSource.TRIBE_EVENTS,
                        model_id="transcript intervals",
                        notes="ASR word boundaries take precedence over windowed tagging.",
                    ),
                )
            )

    return sorted(segments, key=lambda s: s.start_time), warnings


def _collapse_runs(
    kinds: list[AudioSegmentType],
    scores: list[float],
    times: np.ndarray,
    duration: float,
    provenance: Provenance,
) -> list[AudioSegment]:
    """Collapse a per-frame label array into contiguous segments."""
    segments: list[AudioSegment] = []
    if not kinds:
        return segments
    run_start = 0
    for i in range(1, len(kinds) + 1):
        if i < len(kinds) and kinds[i] is kinds[run_start]:
            continue
        start = float(times[run_start])
        end = float(times[i]) if i < len(times) else duration
        end = min(end, duration)
        if end > start:
            segments.append(
                AudioSegment(
                    segment_type=kinds[run_start],
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    mean_energy=0.0,  # filled by the caller from the DSP signal
                    confidence=round(
                        float(sum(scores[run_start:i]) / max(1, i - run_start)), 4
                    ),
                    provenance=provenance,
                )
            )
        run_start = i
    return segments


def events_from_tags(
    windows: list[AudioTagWindow], *, duration: float, threshold: float = LABEL_THRESHOLD
) -> list[dict[str, float | str]]:
    """Discrete audio events — applause, impacts, vehicles and so on.

    Medium labels (Music/Speech/Silence) are excluded: they describe what the
    audio *is*, not something that *happened*, and are reported as segments.
    Broad AudioSet parents are dropped as uninformative.

    Consecutive windows in which a label stays above threshold are merged into
    **one** event. Emitting one event per window multiplied every sound by the
    window overlap: at the 1 s hop a single applause spanning one 10.24 s
    receptive field produced six identical events, and a minute of audio would
    have produced fifty-five.

    The reported time is the *peak* of the label's probability across the merged
    windows, not the first window's start. The 10.24 s receptive field is fixed
    by the model and cannot be narrowed, so a window start says only "somewhere
    in the next ten seconds"; the peak is the best localization the model's own
    output supports. `start_time`/`end_time` still bound the merged span, so the
    uncertainty stays visible rather than being hidden behind a point estimate.
    """
    ordered = sorted(windows, key=lambda w: w.start)
    by_label: dict[str, list[AudioTagWindow]] = {}
    for window in ordered:
        for label, probability in window.probabilities.items():
            if probability < threshold:
                continue
            if label in MEDIUM_LABELS or label in UNINFORMATIVE:
                continue
            by_label.setdefault(label, []).append(window)

    events: list[dict[str, float | str]] = []
    for label, hits in by_label.items():
        run: list[AudioTagWindow] = []
        for window in hits:
            # A gap means the label dropped below threshold in between, which is
            # a second occurrence rather than a continuation.
            if run and window.start > run[-1].end:
                events.append(_event(label, run, duration))
                run = []
            run.append(window)
        if run:
            events.append(_event(label, run, duration))

    return sorted(events, key=lambda item: (item["start_time"], -float(item["confidence"])))


def _event(
    label: str, run: list[AudioTagWindow], duration: float
) -> dict[str, float | str]:
    """One merged event, timed at the peak of its probability."""
    peak = max(run, key=lambda w: w.probabilities.get(label, 0.0))
    probability = peak.probabilities.get(label, 0.0)
    return {
        "label": label,
        "start_time": round(run[0].start, 3),
        "end_time": round(min(duration, max(w.end for w in run)), 3),
        # Midpoint of the highest-scoring window: the model localises no finer.
        "peak_time": round(min(duration, (peak.start + peak.end) / 2.0), 3),
        "confidence": round(float(probability), 4),
    }


def load_waveform(
    wav_path: Path, *, target_rate: int = MODEL_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """Read a mono waveform resampled to the rate the model requires.

    AST is trained on 16 kHz and its feature extractor *rejects* any other rate
    outright. The pipeline's own ffmpeg extraction already writes 16 kHz, but a
    stimulus supplied directly as ``.wav``/``.flac`` (both valid TRIBE inputs)
    is commonly 44.1 kHz. Without this resample the extractor raised, every
    window was skipped, and tagging returned an empty list with no warning —
    the analysis simply had no audio events and did not say why.
    """
    try:
        import soundfile as sf

        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as exc:
        logger.warning("Could not read %s: %s", wav_path.name, exc)
        return np.zeros(0, dtype=np.float32), target_rate
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)
    sample_rate = int(sample_rate)
    if sample_rate != target_rate and samples.size:
        samples, sample_rate = _resample(samples, sample_rate, target_rate)
    return samples, sample_rate


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> tuple[np.ndarray, int]:
    """Resample, preferring torchaudio's polyphase filter over linear interpolation."""
    try:
        import torch
        import torchaudio

        resampled = torchaudio.functional.resample(
            torch.from_numpy(samples), source_rate, target_rate
        )
        return resampled.numpy().astype(np.float32), target_rate
    except Exception as exc:
        logger.debug("torchaudio resample unavailable (%s); interpolating", exc)

    # Linear interpolation is inferior — it does not band-limit, so energy above
    # the new Nyquist aliases down — but it is far better than not tagging at all.
    count = round(samples.size * target_rate / source_rate)
    if count <= 0:
        return np.zeros(0, dtype=np.float32), target_rate
    source_t = np.linspace(0.0, 1.0, samples.size, dtype=np.float64)
    target_t = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return np.interp(target_t, source_t, samples).astype(np.float32), target_rate
