"""Content analysis orchestration.

WHAT IT DOES
    Runs every modality pipeline over one stimulus, fuses the results, aligns
    them to the run's neural events, and persists the whole thing.

THREE PROPERTIES THAT MATTER MORE THAN THE ANALYSIS ITSELF

    Caching. Content analysis is the most expensive thing in Phase 4 — CLIP over
    keyframes, OCR, embeddings. A page load must never trigger it. The cache key
    covers the stimulus bytes, every model id, the analysis version and the
    configuration, so a stale result is recomputed and an identical one is not.

    Failure isolation. OCR failing must not destroy a run that has perfectly
    good transcript and shot data. Each stage is wrapped; a failure appends a
    warning and leaves that modality empty rather than aborting.

    Parallelism. Visual and audio work are independent and both are I/O- and
    CPU-bound, so they run concurrently in threads. Deliberately threads and not
    processes: the payloads are large arrays, and the heavy libraries release
    the GIL.

IN:  a completed Phase 1 run
OUT: a persisted ContentAnalysisResult
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from blackmirror.content import metrics as metrics_module
from blackmirror.content.alignment import (
    DEFAULT_CONTEXT_WINDOW_SECONDS,
    associate_neural_events,
    explore_correlations,
)
from blackmirror.content.audio.diarization import (
    MODEL_ID as SPEAKER_MODEL_ID,
)
from blackmirror.content.audio.diarization import (
    SpeakerEmbedder,
    apply_speakers,
    diarize,
)
from blackmirror.content.audio.signals import (
    AudioSignals,
    compute_audio_signals,
    segment_audio,
)
from blackmirror.content.audio.tagging import (
    MODEL_ID as AUDIO_TAG_MODEL_ID,
)
from blackmirror.content.audio.tagging import (
    AudioTagger,
    events_from_tags,
    load_waveform,
    segments_from_tags,
)
from blackmirror.content.features import build_feature_matrix
from blackmirror.content.fusion import fuse
from blackmirror.content.language.semantics import (
    EMBEDDING_MODEL_ID,
    EmbeddingModel,
    analyse_language,
    detect_calls_to_action,
)
from blackmirror.content.language.transcript import (
    speech_intervals,
    transcribe_with_whisperx,
    transcript_from_run_events,
)
from blackmirror.content.media import extract_audio, extract_frame, probe_media, sample_times
from blackmirror.content.schemas import (
    ContentAnalysisMetadata,
    ContentAnalysisResult,
    ContentArrayArtifact,
    FrameKind,
)
from blackmirror.content.storage import ContentStore
from blackmirror.content.timeline import ContentTimeline
from blackmirror.content.visual.motion import (
    VisualSignals,
    compute_visual_signals,
    interval_mean,
)
from blackmirror.content.visual.objects import (
    BASE_QUERIES,
    ObjectDetector,
    detect_objects,
    queries_from_transcript,
)
from blackmirror.content.visual.objects import (
    MODEL_ID as DETECTOR_MODEL_ID,
)
from blackmirror.content.visual.ocr import OcrReader, detect_text_overlays
from blackmirror.content.visual.scenes import segment_scenes
from blackmirror.content.visual.semantics import (
    MODEL_ID as CLIP_MODEL_ID,
)
from blackmirror.content.visual.semantics import (
    ClipSemanticsBackend,
    classify_frame_kind,
    describe_frames,
)
from blackmirror.content.visual.shots import detect_shots
from blackmirror.content.workspace import ContentWorkspace
from blackmirror.errors import BlackMirrorError
from blackmirror.utils.hashing import sha256_file, stable_hash
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Bumped whenever a change would alter results. Participates in the cache key.
ANALYSIS_VERSION = "1.2"

#: Keyframes per shot. Three samples catch the start, middle and end of a take
#: without making CLIP cost scale with video length.
KEYFRAMES_PER_SHOT = 3

#: Extra frames sampled purely for OCR, which needs denser coverage than
#: semantics: a text overlay can appear and vanish inside a single shot.
OCR_SAMPLE_HZ = 1.0


@dataclass
class ContentAnalysisConfig:
    """Everything that changes the answer. Hashed into the cache key."""

    keyframes_per_shot: int = KEYFRAMES_PER_SHOT
    ocr_sample_hz: float = OCR_SAMPLE_HZ
    context_window_seconds: float = DEFAULT_CONTEXT_WINDOW_SECONDS
    enable_visual_semantics: bool = True
    enable_ocr: bool = True
    #: Which OCR recognition model to use. "default" covers Chinese, Latin and
    #: digits; "japanese" is the only other verified option. Set explicitly —
    #: auto-detection is not offered because model confidence was measured to be
    #: an unreliable selector. See blackmirror.content.visual.ocr.OCR_SCRIPTS.
    ocr_script: str = "default"
    enable_embeddings: bool = True
    enable_correlations: bool = True
    enable_audio_tagging: bool = True
    enable_object_detection: bool = True
    enable_diarization: bool = True
    #: Score utterances for simultaneous speech. Off by default: it needs a
    #: separate virtual environment (pyannote cannot share TRIBE's torch/numpy
    #: pins) and an accepted pyannote licence. See content/audio/overlap.py.
    enable_overlap_scoring: bool = False

    def as_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "keyframes_per_shot": self.keyframes_per_shot,
            "ocr_sample_hz": self.ocr_sample_hz,
            "context_window_seconds": self.context_window_seconds,
            "enable_visual_semantics": self.enable_visual_semantics,
            "enable_ocr": self.enable_ocr,
            "ocr_script": self.ocr_script,
            "enable_embeddings": self.enable_embeddings,
            "enable_correlations": self.enable_correlations,
            "enable_audio_tagging": self.enable_audio_tagging,
            "enable_object_detection": self.enable_object_detection,
            "enable_diarization": self.enable_diarization,
            "enable_overlap_scoring": self.enable_overlap_scoring,
        }


@dataclass
class _Stage:
    """Timing and warning collector, so one stage cannot silently swallow another."""

    seconds: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def run(self, name: str, call: Callable[[], Any], default: Any) -> Any:
        """Execute a stage, isolating its failure.

        A stage that raises contributes a warning and its default value. The
        analysis continues with whatever succeeded, which is the difference
        between a partial result and no result at all.
        """
        started = time.perf_counter()
        try:
            return call()
        except Exception as exc:
            self.failures.append(name)
            self.warnings.append(f"{name} failed ({type(exc).__name__}: {exc}).")
            logger.warning("Content stage '%s' failed: %s", name, exc)
            return default
        finally:
            self.seconds[name] = round(time.perf_counter() - started, 3)


def compute_cache_key(
    *,
    stimulus_sha256: str,
    models: dict[str, str],
    config: ContentAnalysisConfig,
    upstream_fingerprints: dict[str, str] | None = None,
) -> str:
    """Identity of an analysis: same bytes, same models, same settings."""
    return stable_hash(
        {
            "version": ANALYSIS_VERSION,
            "stimulus": stimulus_sha256,
            "models": dict(sorted(models.items())),
            "config": config.as_dict(),
            "upstream": dict(sorted((upstream_fingerprints or {}).items())),
        }
    )


def analyze_content(
    run_id: str,
    *,
    artifact_root: Path,
    config: ContentAnalysisConfig | None = None,
    force: bool = False,
) -> ContentAnalysisResult:
    """Analyse a completed run's stimulus and persist the result."""
    from blackmirror.storage.artifact_store import ArtifactStore

    config = config or ContentAnalysisConfig()
    store = ArtifactStore(artifact_root)
    content_store = ContentStore(artifact_root)

    result = store.read_manifest(run_id)
    stimulus_path = result.stimulus.path
    if not stimulus_path.exists():
        raise BlackMirrorError(
            f"Stimulus for run {run_id} is no longer at {stimulus_path}. Phase 1 "
            f"references media by path and hash rather than copying it."
        )

    models = {
        "visual_semantics": CLIP_MODEL_ID if config.enable_visual_semantics else "disabled",
        "ocr": (
            f"rapidocr-onnxruntime[{config.ocr_script}]" if config.enable_ocr else "disabled"
        ),
        "embeddings": EMBEDDING_MODEL_ID if config.enable_embeddings else "disabled",
        "shot_detection": "pyscenedetect-content",
        "audio": "dsp-rms-zcr-flatness",
        "transcript": "tribe-events-or-whisperx",
        "audio_tagging": AUDIO_TAG_MODEL_ID if config.enable_audio_tagging else "disabled",
        "object_detection": DETECTOR_MODEL_ID if config.enable_object_detection else "disabled",
        "diarization": SPEAKER_MODEL_ID if config.enable_diarization else "disabled",
    }
    run_dir = store.run_dir(run_id)
    upstream_fingerprints = _upstream_fingerprints(run_dir)
    cache_key = compute_cache_key(
        stimulus_sha256=result.stimulus.sha256,
        models=models,
        config=config,
        upstream_fingerprints=upstream_fingerprints,
    )

    if not force:
        cached = content_store.cached_result(run_id, cache_key)
        if cached is not None:
            logger.info("Reusing cached content analysis for %s", run_id)
            return cached

    stage = _Stage()
    started = time.perf_counter()

    media = probe_media(stimulus_path)
    duration = media.duration_seconds
    keyframe_dir = content_store.keyframe_directory(run_id)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    # Regenerable intermediates live in the model cache, keyed by stimulus hash —
    # never in the run directory, which must not duplicate media.
    workspace = ContentWorkspace.for_stimulus(
        Path(settings_cache_dir(artifact_root)), result.stimulus.sha256
    )

    # --- transcript first: it makes the audio segmenter far better ---
    transcript: list[Any] = stage.run(
        "transcript",
        lambda: _load_transcript(store.run_dir(run_id), stimulus_path, duration, stage, workspace),
        [],
    )

    # --- visual and audio pipelines run concurrently ---
    with ThreadPoolExecutor(max_workers=2) as pool:
        visual_future = pool.submit(
            stage.run,
            "visual_signals",
            lambda: _visual_signals(stimulus_path, duration),
            (VisualSignals(), []),
        )
        audio_future = pool.submit(
            stage.run,
            "audio_signals",
            lambda: _audio_signals(stimulus_path, workspace, media.has_audio),
            (AudioSignals(), []),
        )
        visual_signals, visual_warnings = visual_future.result()
        audio_signals, audio_warnings = audio_future.result()
    stage.warnings.extend(visual_warnings)
    stage.warnings.extend(audio_warnings)

    shots, shot_warnings = stage.run(
        "shot_detection", lambda: detect_shots(stimulus_path, duration=duration), ([], [])
    )
    stage.warnings.extend(shot_warnings)

    # A trained tagger supersedes the DSP heuristic where it is available; the
    # heuristic remains the fallback so the pipeline degrades rather than fails.
    audio_windows: list[Any] = []
    audio_events: list[Any] = []
    if config.enable_audio_tagging and media.has_audio:
        audio_windows = stage.run(
            "audio_tagging",
            lambda: _tag_audio(workspace),
            [],
        )
        audio_events = events_from_tags(audio_windows, duration=duration)

    if audio_windows:
        audio_segments, segment_warnings = stage.run(
            "audio_segmentation",
            lambda: segments_from_tags(
                audio_windows,
                duration=duration,
                speech_intervals=speech_intervals(transcript),
            ),
            ([], []),
        )
        audio_segments = _attach_energy(audio_segments, audio_signals)
    else:
        audio_segments, segment_warnings = stage.run(
            "audio_segmentation",
            lambda: segment_audio(
                audio_signals, duration, speech_intervals=speech_intervals(transcript)
            ),
            ([], []),
        )
        if config.enable_audio_tagging and media.has_audio:
            segment_warnings = [
                *segment_warnings,
                "Audio tagging unavailable; segments came from the DSP heuristic, "
                "which reports no calibrated confidence.",
            ]
    stage.warnings.extend(segment_warnings)

    # Diarization needs both the waveform and the transcript.
    if config.enable_diarization and transcript and media.has_audio:
        transcript = stage.run(
            "diarization",
            lambda: _diarize(workspace, transcript, stage),
            transcript,
        )

    # Diarization gives every utterance exactly one speaker. This scores how
    # likely each one is to contain two, which diarization cannot see.
    if config.enable_overlap_scoring and transcript and media.has_audio:
        transcript = stage.run(
            "overlap_scoring",
            lambda: _score_overlap(workspace, transcript, stage),
            transcript,
        )

    # --- keyframes: the input to both semantics and OCR ---
    keyframes = stage.run(
        "keyframes",
        lambda: _extract_keyframes(stimulus_path, shots, keyframe_dir, config, duration),
        [],
    )
    relative = {path: path.name for _, path in keyframes}

    overlays: list[Any] = []
    if config.enable_ocr:
        ocr_frames = stage.run(
            "ocr_frames",
            lambda: _extract_ocr_frames(
                stimulus_path, workspace.ocr_frame_dir, config, duration
            ),
            [],
        )
        detected, ocr_warnings = stage.run(
            "ocr",
            lambda: detect_text_overlays(
                OcrReader(script=config.ocr_script),
                ocr_frames,
                duration=duration,
                frame_span=1.0 / max(config.ocr_sample_hz, 0.01),
            ),
            ([], []),
        )
        overlays = detected
        stage.warnings.extend(ocr_warnings)

    # Frame kind is evidence-based (OCR text + entropy), which is why OCR runs
    # before semantics: asking CLIP "is this a title card?" was measured to
    # misfire badly on dark photographic frames.
    frame_kinds = _frame_kinds(keyframes, overlays, visual_signals)

    attributes: list[Any] = []
    if config.enable_visual_semantics and keyframes:
        described, semantic_warnings = stage.run(
            "visual_semantics",
            lambda: describe_frames(
                ClipSemanticsBackend(),
                keyframes,
                relative_paths=relative,
                frame_kinds=frame_kinds,
            ),
            ([], []),
        )
        attributes = described
        stage.warnings.extend(semantic_warnings)

    object_appearances: list[Any] = []
    detections: list[Any] = []
    if config.enable_object_detection and keyframes:
        # An open-vocabulary detector finds only what it is asked for, so ask it
        # about the words the content itself repeats.
        spoken = " ".join(segment.text for segment in transcript) if transcript else ""
        extra_queries = queries_from_transcript(spoken)
        if extra_queries:
            stage.warnings.append(
                f"Detector vocabulary extended with {len(extra_queries)} term(s) from the "
                f"transcript: {', '.join(extra_queries)}."
            )
        appearances, found, detect_warnings = stage.run(
            "object_detection",
            lambda: detect_objects(
                ObjectDetector(queries=BASE_QUERIES + extra_queries),
                keyframes,
                frame_span=duration / max(1, len(keyframes)),
                duration=duration,
            ),
            ([], [], []),
        )
        object_appearances, detections = appearances, found
        stage.warnings.extend(detect_warnings)

    scenes, scene_warnings = stage.run(
        "scenes",
        lambda: segment_scenes(shots, attributes, transcript, duration=duration),
        ([], []),
    )
    stage.warnings.extend(scene_warnings)

    embedder = EmbeddingModel() if config.enable_embeddings else None
    language, language_warnings = stage.run(
        "language",
        lambda: analyse_language(transcript, duration=duration, embedder=embedder),
        ([], []),
    )
    stage.warnings.extend(language_warnings)

    calls_to_action = stage.run(
        "cta", lambda: detect_calls_to_action(language, overlays, embedder=embedder), []
    )

    events, fusion_warnings = stage.run(
        "fusion",
        lambda: fuse(
            duration=duration,
            shots=shots,
            scenes=scenes,
            attributes=attributes,
            overlays=overlays,
            audio_segments=audio_segments,
            transcript=transcript,
            language=language,
            calls_to_action=calls_to_action,
            visual_signals=visual_signals,
            audio_signals=audio_signals,
            detections=detections,
        ),
        ([], []),
    )
    stage.warnings.extend(fusion_warnings)

    matrix, feature_names, feature_times = stage.run(
        "features",
        lambda: build_feature_matrix(
            duration=duration,
            visual_signals=visual_signals,
            audio_signals=audio_signals,
            events=events,
            shot_times=[shot.start_time for shot in shots[1:]],
        ),
        (np.zeros((0, 0), dtype=np.float32), (), np.zeros(0, dtype=np.float32)),
    )

    timeline = ContentTimeline(events)
    associations, correlations = stage.run(
        "neural_alignment",
        lambda: _align_to_neural(
            artifact_root=artifact_root,
            run_id=run_id,
            timeline=timeline,
            matrix=matrix,
            feature_names=feature_names,
            feature_times=feature_times,
            config=config,
            warnings=stage.warnings,
        ),
        ([], []),
    )

    computed = metrics_module.compute_metrics(
        duration=duration,
        shots=shots,
        scenes=scenes,
        transcript=transcript,
        audio_segments=audio_segments,
        overlays=overlays,
        calls_to_action=calls_to_action,
        events=events,
        visual_signals=visual_signals,
        audio_signals=audio_signals,
    )

    arrays: dict[str, np.ndarray] = {
        "features": matrix,
        "feature_times": feature_times,
        **visual_signals.as_dict(),
        **audio_signals.as_dict(),
    }
    stage.seconds["total"] = round(time.perf_counter() - started, 3)

    analysis = ContentAnalysisResult(
        run_id=run_id,
        stimulus_id=result.stimulus.stimulus_id,
        stimulus_filename=result.stimulus.filename,
        media=media,
        modalities_analysed=_modalities(attributes, overlays, audio_segments, transcript),
        shots=tuple(shots),
        scenes=tuple(scenes),
        visual_attributes=tuple(attributes),
        text_overlays=tuple(overlays),
        objects=tuple(object_appearances),
        audio_segments=tuple(audio_segments),
        audio_events=tuple(audio_events),
        transcript=tuple(transcript),
        language=tuple(language),
        calls_to_action=tuple(calls_to_action),
        events=tuple(events),
        associations=tuple(associations),
        correlations=tuple(correlations),
        metrics=computed,
        arrays=ContentArrayArtifact(
            path=f"features-{cache_key[:16]}.npz",
            keys={
                "features": "Content feature matrix [T, F] on a uniform grid.",
                "feature_times": "Centre time of each feature-matrix row, seconds.",
                "motion": "Frame-difference motion magnitude.",
                "audio_energy": "Short-time RMS energy.",
            },
            frame_count=int(matrix.shape[0]) if matrix.ndim == 2 else 0,
            feature_names=tuple(feature_names),
        ),
        metadata=ContentAnalysisMetadata(
            analysis_version=ANALYSIS_VERSION,
            stimulus_sha256=result.stimulus.sha256,
            cache_key=cache_key,
            models=models,
            configuration=config.as_dict(),
            stage_seconds=stage.seconds,
            warnings=tuple(dict.fromkeys(stage.warnings)),
            failed_stages=tuple(dict.fromkeys(stage.failures)),
            # Everything runs locally, so there is no external spend to report.
            api_cost={},
        ),
    )

    content_store.write(analysis, arrays)
    logger.info(
        "Content analysis complete for %s in %.1fs (%d events, %d associations)",
        run_id,
        stage.seconds["total"],
        len(events),
        len(associations),
    )
    return analysis


def _upstream_fingerprints(run_dir: Path) -> dict[str, str]:
    """Hash optional upstream artifacts whose contents alter Phase 4 output."""
    candidates = {
        "events": run_dir / "events.parquet",
        "analytics_metadata": run_dir / "analytics" / "metadata.json",
        "analytics_timeseries": run_dir / "analytics" / "timeseries.npz",
    }
    return {
        name: sha256_file(path) if path.is_file() else "absent"
        for name, path in candidates.items()
    }


def _load_transcript(
    run_dir: Path,
    stimulus: Path,
    duration: float,
    stage: _Stage,
    workspace: ContentWorkspace,
) -> list[Any]:
    """Prefer Phase 1's transcript; fall back to WhisperX only if absent."""
    segments, warnings = transcript_from_run_events(run_dir / "events.parquet", duration=duration)
    stage.warnings.extend(warnings)
    if segments:
        return segments

    audio_path = workspace.audio_path
    try:
        if not workspace.has_audio():
            extract_audio(stimulus, audio_path)
    except Exception as exc:
        stage.warnings.append(f"Could not extract audio for transcription: {exc}")
        return []
    segments, warnings = transcribe_with_whisperx(audio_path, duration=duration)
    stage.warnings.extend(warnings)
    return segments


def _tag_audio(workspace: ContentWorkspace) -> list[Any]:
    """Run the AudioSet tagger over the decoded waveform."""
    if not workspace.has_audio():
        return []
    samples, sample_rate = load_waveform(workspace.audio_path)
    if samples.size == 0:
        return []
    return AudioTagger().tag(samples, sample_rate)


def _attach_energy(segments: list[Any], signals: AudioSignals) -> list[Any]:
    """Fill each tagged segment's mean energy from the DSP signal.

    The tagger reports labels, not levels; energy still comes from the waveform.
    """
    if not len(signals):
        return segments
    filled = []
    for segment in segments:
        value = interval_mean(
            signals.times, signals.energy, segment.start_time, segment.end_time
        )
        filled.append(segment.model_copy(update={"mean_energy": float(value or 0.0)}))
    return filled


def _score_overlap(
    workspace: ContentWorkspace, transcript: list[Any], stage: Any
) -> list[Any]:
    """Attach an overlap probability to each utterance, scored in the isolated env."""
    from blackmirror.content.audio.overlap import (
        repository_root,
        score_overlap,
    )

    wav_path = workspace.audio_path
    if not wav_path.exists():
        return transcript
    spans = [(float(seg.start_time), float(seg.end_time)) for seg in transcript]
    scores, warnings = score_overlap(wav_path, spans, root=repository_root())
    stage.warnings.extend(warnings)
    if not scores:
        return transcript
    by_span = {(s.start, s.end): s.probability for s in scores}
    return [
        segment.model_copy(
            update={
                "overlap_probability": by_span.get(
                    (round(float(segment.start_time), 3), round(float(segment.end_time), 3))
                )
            }
        )
        for segment in transcript
    ]


def _diarize(workspace: ContentWorkspace, transcript: list[Any], stage: _Stage) -> list[Any]:
    """Attach speaker labels, or leave the transcript untouched and say why."""
    if not workspace.has_audio():
        return transcript
    samples, sample_rate = load_waveform(workspace.audio_path)
    assignments, warnings = diarize(
        SpeakerEmbedder(cache_dir=workspace.root / "speaker"),
        samples,
        sample_rate,
        transcript,
    )
    stage.warnings.extend(warnings)
    return apply_speakers(transcript, assignments) if assignments else transcript


def _frame_kinds(
    keyframes: list[tuple[float, Path]],
    overlays: list[Any],
    signals: VisualSignals,
) -> dict[float, FrameKind]:
    """Classify each keyframe from measured evidence, not from a CLIP prompt."""
    kinds: dict[float, FrameKind] = {}
    for timestamp, _ in keyframes:
        entropy = interval_mean(
            signals.times, signals.entropy, timestamp - 0.25, timestamp + 0.25
        )
        saturation = interval_mean(
            signals.times, signals.saturation, timestamp - 0.25, timestamp + 0.25
        )
        has_text = any(
            overlay.start_time <= timestamp < overlay.end_time for overlay in overlays
        )
        kinds[timestamp] = classify_frame_kind(
            entropy=entropy, saturation=saturation, has_text=has_text
        )
    return kinds


def _visual_signals(path: Path, duration: float) -> tuple[VisualSignals, list[str]]:
    return compute_visual_signals(path, duration=duration)


def _audio_signals(
    path: Path, workspace: ContentWorkspace, has_audio: bool
) -> tuple[AudioSignals, list[str]]:
    if not has_audio:
        return AudioSignals(), ["Stimulus has no audio track."]
    if not workspace.has_audio():
        extract_audio(path, workspace.audio_path)
    return compute_audio_signals(workspace.audio_path)


def settings_cache_dir(artifact_root: Path) -> Path:
    """Where regenerable content intermediates live.

    Derived from settings so it follows BLACKMIRROR_MODEL_CACHE_DIR, with a
    sibling of the artifact root as a fallback for direct callers.
    """
    try:
        from blackmirror.config.settings import get_settings

        return Path(get_settings().model_cache_dir)
    except Exception:
        return artifact_root.parent / "cache"


def _extract_keyframes(
    path: Path,
    shots: list[Any],
    directory: Path,
    config: ContentAnalysisConfig,
    duration: float,
) -> list[tuple[float, Path]]:
    """One to three representative frames per shot."""
    frames: list[tuple[float, Path]] = []
    for shot in shots or []:
        for timestamp in sample_times(shot.start_time, shot.end_time, config.keyframes_per_shot):
            target = directory / f"shot{shot.index:03d}_{timestamp:07.3f}.jpg"
            try:
                extract_frame(path, timestamp, target)
            except Exception:
                continue
            frames.append((timestamp, target))
    if not frames and duration > 0:
        timestamp = duration / 2.0
        target = directory / f"mid_{timestamp:07.3f}.jpg"
        extract_frame(path, timestamp, target)
        frames.append((timestamp, target))
    return frames


def _extract_ocr_frames(
    path: Path, directory: Path, config: ContentAnalysisConfig, duration: float
) -> list[tuple[float, Path]]:
    """Denser uniform sampling for text, which can flash between keyframes."""
    ocr_dir = directory
    step = 1.0 / max(config.ocr_sample_hz, 0.01)
    frames: list[tuple[float, Path]] = []
    timestamp = 0.0
    while timestamp < duration:
        target = ocr_dir / f"t{timestamp:07.3f}.jpg"
        try:
            extract_frame(path, timestamp, target, width=768)
        except Exception:
            timestamp += step
            continue
        frames.append((timestamp, target))
        timestamp += step
    return frames


def _align_to_neural(
    *,
    artifact_root: Path,
    run_id: str,
    timeline: ContentTimeline,
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
    feature_times: np.ndarray,
    config: ContentAnalysisConfig,
    warnings: list[str],
) -> tuple[list[Any], list[Any]]:
    """Attach content context to Phase 3 neural events, if they exist.

    Phase 3 is consumed through its artifacts, never imported — the same
    boundary Phase 2 keeps with Phase 1.
    """
    import json

    analytics_path = artifact_root / "runs" / run_id / "analytics" / "metadata.json"
    if not analytics_path.exists():
        warnings.append(
            "No Phase 3 analytics for this run; neural associations were not built. "
            "Run `blackmirror analyze <run_id>` first."
        )
        return [], []

    payload = json.loads(analytics_path.read_text(encoding="utf-8"))
    neural_events = payload.get("events", [])
    associations = associate_neural_events(
        neural_events, timeline, context_window_seconds=config.context_window_seconds
    )

    correlations: list[Any] = []
    if config.enable_correlations and matrix.size:
        timeseries_path = artifact_root / "runs" / run_id / "analytics" / "timeseries.npz"
        if timeseries_path.exists():
            with np.load(timeseries_path) as bundle:
                neural_times = np.asarray(bundle.get("times", np.zeros(0)), dtype=np.float64)
                series = {
                    key: np.asarray(bundle[key], dtype=np.float64)
                    for key in (
                        "global_mean",
                        "global_l2_magnitude",
                        "global_rms_magnitude",
                        "change_magnitude",
                        "spatial_concentration",
                    )
                    if key in bundle.files
                }
            if neural_times.size and not series:
                warnings.append(
                    "Phase 3 timeseries contained no recognised neural metric; "
                    "correlations skipped. Key names may have changed upstream."
                )
            if neural_times.size and series:
                correlations, correlation_warnings = explore_correlations(
                    matrix, feature_names, feature_times, series, neural_times
                )
                warnings.extend(correlation_warnings)
    return associations, correlations


def _modalities(
    attributes: list[Any], overlays: list[Any], audio_segments: list[Any], transcript: list[Any]
) -> tuple[str, ...]:
    present: list[str] = []
    if attributes:
        present.append("visual")
    if overlays:
        present.append("on_screen_text")
    if audio_segments:
        present.append("audio")
    if transcript:
        present.append("speech")
    return tuple(present)
