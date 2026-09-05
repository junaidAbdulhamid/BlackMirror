"""Scene segmentation.

WHAT A SCENE IS, AND WHY IT IS NOT A SHOT
    A shot is a camera take, bounded by cuts. A scene is a *semantic* unit that
    usually spans several shots — a conversation cut back and forth between two
    people is many shots but one scene. Shots come from pixels; scenes need
    meaning.

ALGORITHM (deliberately simple, and documented because it is a judgement call)
    Greedy agglomeration over consecutive shots. A shot joins the current scene
    when it is *similar enough* to it, where similarity combines:

      * visual continuity — cosine similarity of the shots' CLIP label vectors
      * speech continuity — whether an utterance spans the shot boundary
      * duration pressure — a scene that has grown past `max_scene_seconds`
        is closed regardless, so one long scene cannot swallow the video

    Full film-grammar scene detection is a research problem. This is a pragmatic
    grouping that produces stable, explainable boundaries, and every scene
    records *why* it was grouped.

IN:  shots + their visual attributes + transcript
OUT: SceneSegment[]
"""

from __future__ import annotations

import numpy as np

from blackmirror.content.schemas import (
    SceneSegment,
    ShotSegment,
    TranscriptSegment,
    VisualAttributes,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Below this cosine similarity two shots are treated as different scenes.
VISUAL_SIMILARITY_THRESHOLD = 0.86

#: A scene is closed once it reaches this length, so pacing stays legible.
MAX_SCENE_SECONDS = 20.0


def segment_scenes(
    shots: list[ShotSegment],
    attributes: list[VisualAttributes],
    transcript: list[TranscriptSegment],
    *,
    duration: float,
    similarity_threshold: float = VISUAL_SIMILARITY_THRESHOLD,
    max_scene_seconds: float = MAX_SCENE_SECONDS,
) -> tuple[list[SceneSegment], list[str]]:
    """Group shots into scenes."""
    warnings: list[str] = []
    if not shots:
        return [], ["No shots available for scene segmentation."]
    if len(shots) == 1:
        return [
            SceneSegment(
                index=0,
                start_time=shots[0].start_time,
                end_time=shots[0].end_time,
                duration=shots[0].duration,
                shot_indices=(0,),
                description=_describe(attributes, shots[0].start_time, shots[0].end_time),
                grouping_reason="single shot",
            )
        ], warnings

    vectors = _shot_vectors(shots, attributes)
    if vectors is None:
        warnings.append(
            "No visual attributes available; scenes fall back to fixed-duration grouping."
        )

    groups: list[list[int]] = [[0]]
    reasons: list[str] = ["scene start"]

    for index in range(1, len(shots)):
        current = groups[-1]
        scene_start = shots[current[0]].start_time
        scene_end = shots[index].end_time

        too_long = scene_end - scene_start > max_scene_seconds
        similarity = (
            _similarity(vectors[current[-1]], vectors[index]) if vectors is not None else 0.0
        )
        speech_bridges = _speech_spans(transcript, shots[index].start_time)

        if too_long:
            groups.append([index])
            reasons.append(f"previous scene exceeded {max_scene_seconds:.0f}s")
        elif vectors is not None and similarity >= similarity_threshold:
            current.append(index)
        elif speech_bridges:
            # An utterance carrying across the cut is strong evidence of one
            # continuous scene even when the framing changes completely.
            current.append(index)
        elif vectors is None:
            # Without semantics, group consecutive shots up to the documented
            # duration cap instead of silently degrading to one scene per shot.
            current.append(index)
        else:
            groups.append([index])
            reasons.append(
                f"visual similarity {similarity:.2f} < {similarity_threshold:.2f}"
                if vectors is not None
                else "no visual continuity signal"
            )

    scenes: list[SceneSegment] = []
    for scene_index, (members, reason) in enumerate(zip(groups, reasons, strict=True)):
        start = shots[members[0]].start_time
        end = min(duration, shots[members[-1]].end_time)
        keyframes = [
            attribute.keyframe_path
            for attribute in attributes
            if attribute.keyframe_path and start <= attribute.time < end
        ]
        rounded_start = round(start, 3)
        rounded_end = round(end, 3)
        scenes.append(
            SceneSegment(
                index=scene_index,
                start_time=rounded_start,
                end_time=rounded_end,
                duration=round(max(0.0, rounded_end - rounded_start), 3),
                shot_indices=tuple(members),
                description=_describe(attributes, start, end),
                keyframe_path=keyframes[0] if keyframes else None,
                grouping_reason=reason,
            )
        )

    logger.info("Grouped %d shot(s) into %d scene(s)", len(shots), len(scenes))
    return scenes, warnings


def _shot_vectors(
    shots: list[ShotSegment], attributes: list[VisualAttributes]
) -> list[np.ndarray] | None:
    """One label vector per shot, averaged over its keyframes.

    Using the CLIP label scores rather than raw embeddings keeps this
    interpretable — two shots are "similar" because they scored alike on the
    same named labels.
    """
    if not attributes:
        return None
    keys = sorted({key for attribute in attributes for key in attribute.labels})
    if not keys:
        return None

    vectors: list[np.ndarray] = []
    for shot in shots:
        inside = [
            attribute
            for attribute in attributes
            if shot.start_time <= attribute.time < shot.end_time
        ]
        if inside:
            stacked = np.array(
                [[attribute.labels.get(key, 0.0) for key in keys] for attribute in inside],
                dtype=np.float32,
            )
            vectors.append(stacked.mean(axis=0))
        else:
            # Missing semantic evidence is not evidence that the shot resembles
            # whichever sampled frame happens to be temporally closest.
            vectors.append(np.zeros(len(keys), dtype=np.float32))
    return vectors


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _speech_spans(transcript: list[TranscriptSegment], boundary: float) -> bool:
    """True when an utterance straddles this shot boundary."""
    return any(
        segment.start_time < boundary - 1e-6 < segment.end_time for segment in transcript
    )


def _describe(
    attributes: list[VisualAttributes], start: float, end: float
) -> str | None:
    """A short, factual scene description assembled from structured fields.

    Composed from labels rather than generated as prose, so it is reproducible
    and traceable to the underlying scores.
    """
    inside = [a for a in attributes if start <= a.time < end]
    if not inside:
        return None
    settings = [a.setting for a in inside if a.setting]
    scales = [a.shot_scale for a in inside if a.shot_scale]
    actions = [a.actions[0] for a in inside if a.actions]
    parts: list[str] = []
    if settings:
        parts.append(max(set(settings), key=settings.count))
    if actions:
        parts.append(max(set(actions), key=actions.count))
    if scales:
        parts.append(f"{max(set(scales), key=scales.count).replace('_', ' ')} framing")
    return "; ".join(parts) if parts else None
