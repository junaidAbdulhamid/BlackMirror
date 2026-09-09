"""Multimodal event fusion — the core of Phase 4.

THE PROBLEM
    Each pipeline produces intervals on its own natural grid. Shots are bounded
    by cuts, utterances by speech, music by acoustics, motion by a 4 Hz sampler.
    None of them agree. To ask "what was happening at 13.2 seconds?" they must
    be reconciled onto one timeline.

EVENT-DRIVEN INTERVALS, NOT FIXED BINS
    The obvious approach is fixed 1-second bins. It is also wrong: a bin that
    straddles a cut mixes two shots into one description, and a 0.4 s text flash
    disappears entirely.

    Instead the fused grid is built from the *union of real boundaries* — every
    shot cut, every utterance start and end, every audio-segment edge. Between
    two consecutive boundaries nothing changes categorically, so each resulting
    interval is internally homogeneous by construction. Intervals are then
    populated by asking each modality what it held over that span.

    Cost: intervals are uneven in length, so anything averaging over them must
    weight by duration. That is handled in features.py and metrics.py.

IN:  every modality's events and signals
OUT: ContentEvent[] on one shared timeline
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np

from blackmirror.content.audio.signals import AudioSignals
from blackmirror.content.schemas import (
    AnalysisSource,
    AudioSegment,
    AudioSegmentType,
    CallToAction,
    ContentEvent,
    ContentEventType,
    LanguageAnalysis,
    ObjectAppearance,
    Provenance,
    SceneSegment,
    ShotSegment,
    TextOverlay,
    TranscriptSegment,
    VisualAttributes,
)
from blackmirror.content.visual.motion import VisualSignals, interval_mean
from blackmirror.content.visual.objects import Detection, objects_in_interval
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

def build_boundaries(
    *,
    duration: float,
    shots: list[ShotSegment],
    transcript: list[TranscriptSegment],
    audio_segments: list[AudioSegment],
    overlays: list[TextOverlay],
    calls_to_action: list[CallToAction] | None = None,
) -> list[float]:
    """Collect every real modality boundary into one sorted, deduplicated grid."""
    points: list[float] = [0.0, duration]
    for shot in shots:
        points.extend((shot.start_time, shot.end_time))
    for utterance in transcript:
        points.extend((utterance.start_time, utterance.end_time))
    for audio in audio_segments:
        points.extend((audio.start_time, audio.end_time))
    for overlay in overlays:
        points.extend((overlay.start_time, overlay.end_time))
    for call in calls_to_action or []:
        points.extend((call.start_time, call.end_time))

    inside = sorted(point for point in points if -1e-6 <= point <= duration + 1e-6)
    # Only coalesce numerical duplicates. Real boundaries even milliseconds
    # apart can represent a short title/CTA/cut and must remain observable.
    merged = list(dict.fromkeys(round(min(max(point, 0.0), duration), 6) for point in inside))
    if len(merged) < 2:
        return [0.0, duration]
    merged[-1] = duration
    return merged


def fuse(
    *,
    duration: float,
    shots: list[ShotSegment],
    scenes: list[SceneSegment],
    attributes: list[VisualAttributes],
    overlays: list[TextOverlay],
    audio_segments: list[AudioSegment],
    transcript: list[TranscriptSegment],
    language: list[LanguageAnalysis],
    calls_to_action: list[CallToAction],
    visual_signals: VisualSignals,
    audio_signals: AudioSignals,
    detections: list[Detection] | None = None,
) -> tuple[list[ContentEvent], list[str]]:
    """Fuse every modality into one ordered list of ContentEvents."""
    warnings: list[str] = []
    boundaries = build_boundaries(
        duration=duration,
        shots=shots,
        transcript=transcript,
        audio_segments=audio_segments,
        overlays=overlays,
        calls_to_action=calls_to_action,
    )
    intervals = list(itertools.pairwise(boundaries))
    if not intervals:
        return [], ["No fusable intervals could be constructed."]

    events: list[ContentEvent] = []
    for start, end in intervals:
        events.append(
            _build_event(
                start=start,
                end=end,
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
                detections=detections or [],
            )
        )

    logger.info(
        "Fused %d modality boundary point(s) into %d content event(s)",
        len(boundaries),
        len(events),
    )
    return events, warnings


def _overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    """Half-open overlap test, tolerant of float noise at shared boundaries."""
    return other_start < end - 1e-6 and other_end > start + 1e-6


def _build_event(
    *,
    start: float,
    end: float,
    shots: list[ShotSegment],
    scenes: list[SceneSegment],
    attributes: list[VisualAttributes],
    overlays: list[TextOverlay],
    audio_segments: list[AudioSegment],
    transcript: list[TranscriptSegment],
    language: list[LanguageAnalysis],
    calls_to_action: list[CallToAction],
    visual_signals: VisualSignals,
    audio_signals: AudioSignals,
    detections: list[Detection],
) -> ContentEvent:
    modalities: list[str] = []
    provenance: list[Provenance] = []

    shot_hits = [s for s in shots if _overlaps(start, end, s.start_time, s.end_time)]
    scene_hit = next(
        (s for s in scenes if _overlaps(start, end, s.start_time, s.end_time)), None
    )
    attribute_hits = [
        attribute
        for attribute in attributes
        if any(shot.start_time <= attribute.time < shot.end_time for shot in shot_hits)
    ]
    overlay_hits = [o for o in overlays if _overlaps(start, end, o.start_time, o.end_time)]
    audio_hits = [a for a in audio_segments if _overlaps(start, end, a.start_time, a.end_time)]
    speech_hits = [t for t in transcript if _overlaps(start, end, t.start_time, t.end_time)]
    language_hits = [
        item for item in language if _overlaps(start, end, item.start_time, item.end_time)
    ]
    cta_hits = [c for c in calls_to_action if _overlaps(start, end, c.start_time, c.end_time)]

    # --- visual ---
    visual_description: str | None = None
    objects: tuple[str, ...] = ()
    if attribute_hits:
        modalities.append("visual")
        primary = attribute_hits[0]
        action = primary.actions[0] if primary.actions else None
        pieces = [part for part in (primary.setting, action) if part]
        if primary.shot_scale and primary.shot_scale != "unknown":
            pieces.append(f"{primary.shot_scale.replace('_', ' ')} framing")
        visual_description = "; ".join(pieces) or None
        objects = tuple(dict.fromkeys(obj for a in attribute_hits for obj in a.objects))
        provenance.append(primary.provenance)

    detected = objects_in_interval(detections, start, end)
    if detected:
        # A detector localises objects; CLIP only says the frame resembles a
        # phrase. Detections therefore take precedence when both are present.
        objects = detected
        if "visual" not in modalities:
            modalities.append("visual")

    visual_features: dict[str, float] = {}
    if len(visual_signals):
        for name, series in (
            ("motion", visual_signals.motion),
            ("brightness", visual_signals.brightness),
            ("contrast", visual_signals.contrast),
            ("saturation", visual_signals.saturation),
            ("entropy", visual_signals.entropy),
        ):
            value = interval_mean(visual_signals.times, series, start, end)
            if value is not None:
                visual_features[name] = round(value, 5)
        if visual_features and "visual" not in modalities:
            modalities.append("visual")

    # --- audio ---
    audio_features: dict[str, float] = {}
    if len(audio_signals):
        for name, series in (
            ("energy", audio_signals.energy),
            ("spectral_centroid", audio_signals.spectral_centroid),
            ("spectral_flatness", audio_signals.spectral_flatness),
            ("speech_activity", audio_signals.speech_activity),
        ):
            value = interval_mean(audio_signals.times, series, start, end)
            if value is not None:
                audio_features[name] = round(value, 5)
        if audio_features:
            modalities.append("audio")

    audio_description: str | None = None
    if audio_hits:
        def _overlap_length(segment: AudioSegment) -> float:
            return min(end, segment.end_time) - max(start, segment.start_time)

        dominant = max(audio_hits, key=_overlap_length)
        audio_description = dominant.segment_type.value
        provenance.append(dominant.provenance)
        if "audio" not in modalities:
            modalities.append("audio")

    # --- speech / language ---
    speech_text: str | None = None
    if speech_hits:
        modalities.append("speech")
        speech_text = " ".join(segment.text for segment in speech_hits).strip()
        provenance.append(speech_hits[0].provenance)

    language_features: dict[str, float | bool | str] = {}
    semantic_description: str | None = None
    if language_hits:
        modalities.append("language")
        primary_language = language_hits[0]
        language_features = {
            "tone": primary_language.tone.value,
            "is_question": primary_language.is_question,
            "is_call_to_action": primary_language.is_call_to_action,
            "contains_numeric_claim": primary_language.contains_numeric_claim,
            "structural_role": primary_language.structural_role.value,
        }
        semantic_description = primary_language.tone.value
        provenance.append(primary_language.provenance)

    on_screen_text = tuple(overlay.text for overlay in overlay_hits)
    if on_screen_text:
        modalities.append("on_screen_text")
        provenance.append(overlay_hits[0].provenance)

    event_type = _classify(
        has_speech=bool(speech_hits),
        has_cta=bool(cta_hits),
        has_overlay=bool(overlay_hits),
        audio_description=audio_description,
    )

    identity = f"{start:.4f}:{end:.4f}:{event_type}"
    return ContentEvent(
        event_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
        event_type=event_type,
        start_time=round(start, 3),
        end_time=round(end, 3),
        modalities=tuple(dict.fromkeys(modalities)),
        visual_description=visual_description,
        speech_text=speech_text or None,
        on_screen_text=on_screen_text,
        audio_description=audio_description,
        semantic_description=semantic_description,
        objects=objects,
        visual_features=visual_features,
        audio_features=audio_features,
        language_features=language_features,
        shot_indices=tuple(shot.index for shot in shot_hits),
        scene_index=scene_hit.index if scene_hit else None,
        confidence=None,
        provenance=tuple(dict.fromkeys(provenance)),
    )


#: Object queries that count as a product for PRODUCT_REVEAL. Drawn from the
#: detector's own vocabulary, so a marker can only appear when one of these was
#: genuinely detected.
PRODUCT_QUERIES = frozenset(
    {
        "a product package",
        "a bottle",
        "a computer",
        "a mobile phone",
        "a car",
        "a logo",
        "a book",
        "food",
        "a drink",
    }
)

#: A product must hold the screen at least this long to count as a reveal
#: rather than a passing frame.
MIN_REVEAL_SECONDS = 0.5


def structural_events(
    *,
    duration: float,
    scenes: list[SceneSegment],
    objects: list[ObjectAppearance] | None = None,
) -> tuple[list[ContentEvent], list[str]]:
    """Derive HOOK and PRODUCT_REVEAL markers, when the evidence supports them.

    These are *additional* events, not relabelled intervals: an opening segment
    is usually also a speech or music segment, and forcing one label onto it
    would lose that.

    Neither marker is invented. HOOK is the first measured scene, labelled a
    positional convention because no detector for "a hook" exists.
    PRODUCT_REVEAL is emitted only when the open-vocabulary detector actually
    found a product-like object that persisted; a stimulus with no product
    produces no marker rather than a guess at one.
    """
    events: list[ContentEvent] = []
    warnings: list[str] = []

    if scenes:
        opening = min(scenes, key=lambda scene: scene.start_time)
        end = min(opening.end_time, duration)
        if end > opening.start_time:
            identity = f"hook|{opening.start_time}|{end}"
            events.append(
                ContentEvent(
                    event_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                    event_type=ContentEventType.HOOK,
                    start_time=round(opening.start_time, 3),
                    end_time=round(end, 3),
                    modalities=("visual",),
                    visual_description=opening.description,
                    provenance=(
                        Provenance(
                            source=AnalysisSource.SCENE_DETECT,
                            model_id="first-scene",
                            notes=(
                                "Positional convention: the opening scene. No detector for "
                                "a 'hook' exists; the boundary is a measured scene boundary "
                                "and the label asserts position only, never rhetorical "
                                "function."
                            ),
                        ),
                    ),
                )
            )
    else:
        warnings.append("no scenes were detected, so no hook marker could be placed")

    reveals = [
        item
        for item in (objects or [])
        if item.label in PRODUCT_QUERIES
        and (item.last_seen - item.first_seen) >= MIN_REVEAL_SECONDS
    ]
    if reveals:
        first = min(reveals, key=lambda item: item.first_seen)
        end = min(first.last_seen, duration)
        if end > first.first_seen:
            identity = f"reveal|{first.label}|{first.first_seen}"
            events.append(
                ContentEvent(
                    event_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                    event_type=ContentEventType.PRODUCT_REVEAL,
                    start_time=round(first.first_seen, 3),
                    end_time=round(end, 3),
                    modalities=("visual",),
                    objects=(first.label,),
                    visual_description=f"first sustained appearance of {first.label}",
                    provenance=(
                        Provenance(
                            source=AnalysisSource.OBJECT_DETECTION,
                            model_id=first.provenance.model_id,
                            notes=(
                                "First appearance of a product-vocabulary object persisting "
                                f"at least {MIN_REVEAL_SECONDS}s. The detector reports only "
                                "what it was asked about, so this marker is bounded by that "
                                "vocabulary."
                            ),
                        ),
                    ),
                )
            )
    elif objects:
        warnings.append(
            "no product-vocabulary object persisted long enough to mark a product reveal"
        )

    return events, warnings


def _classify(
    *,
    has_speech: bool,
    has_cta: bool,
    has_overlay: bool,
    audio_description: str | None,
) -> ContentEventType:
    """Label the interval by its most distinctive property.

    Order matters: a CTA that also has speech is more usefully a CTA. Every
    interval remains queryable by any modality regardless of this label.
    """
    if has_cta:
        return ContentEventType.CTA
    if has_speech:
        return ContentEventType.SPEECH_SEGMENT
    if has_overlay:
        return ContentEventType.TEXT_OVERLAY
    if audio_description == AudioSegmentType.SILENCE.value:
        return ContentEventType.SILENCE
    if audio_description == AudioSegmentType.MUSIC.value:
        return ContentEventType.MUSIC_SEGMENT
    return ContentEventType.FUSED_INTERVAL


def dominant_signal_windows(
    signals: VisualSignals | AudioSignals, series_name: str, top_k: int = 3
) -> list[float]:
    """Timestamps of the strongest moments in a dense signal.

    Used for content-side highlights; deliberately separate from Phase 3's
    neural event detection, which operates on entirely different data.
    """
    mapping = signals.as_dict()
    values = mapping.get(series_name)
    times = mapping.get("visual_times", mapping.get("audio_times"))
    if values is None or times is None or values.size == 0:
        return []
    order = np.argsort(values)[::-1][:top_k]
    return [round(float(times[index]), 3) for index in sorted(order)]
