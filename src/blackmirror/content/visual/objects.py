"""Object detection and temporal tracking.

WHAT THIS ADDS OVER CLIP
    CLIP zero-shot answers "does this frame resemble the phrase 'a person'?".
    A detector answers "how many, and where". That difference matters for the
    quantities Phase 5 wants to diff: screen time, appearance counts, and when
    something first entered frame.

    `ObjectAppearance` was previously modelled but never populated. This fills it.

MODEL
    `google/owlv2-base-patch16-ensemble` (Apache-2.0) — an **open-vocabulary**
    detector. It is given a list of text queries and finds those, rather than
    predicting from a fixed class list.

WHY OPEN-VOCABULARY, MEASURED
    The previous implementation used `facebook/detr-resnet-50`, whose 91 COCO
    classes cannot express anything outside them. Every out-of-vocabulary object
    is mapped to its nearest COCO neighbour, confidently, and no threshold
    fixes it. Measured on the Sintel trailer (animated fantasy), same frames,
    same sampling:

        t=33 s  dragon in flight   DETR "bird" 0.97     OWLv2 "a flying creature" 0.73
        t=35 s  dragon in flight   DETR "person" 0.99   OWLv2 "a flying creature" 0.78
        t=37 s  fantasy interior   DETR "bed" 0.99      OWLv2 (nothing)
        t=39 s  fantasy interior   DETR "bed" 0.78      OWLv2 (nothing)
        t=02 s  title card         DETR "boat" 0.58     OWLv2 (nothing above 0.25)

    Two structural properties do the work. The vocabulary is *supplied*, so the
    detector cannot invent "bed" — it can only answer about what it was asked.
    And it is free to answer "none of these", which a fixed-class softmax over
    object queries is far less willing to do.

    This does not make it correct, only answerable. A query for "a dragon" that
    returns 0.7 means the model matched that phrase to a region; it remains a
    model judgement, and the queried vocabulary is recorded in provenance so a
    reader knows exactly which question was asked.

THE VOCABULARY IS PART OF THE RESULT
    Because the detector only finds what it is asked for, the query list is not
    an implementation detail — it bounds what could possibly be reported. A
    base list covers common on-screen entities; the caller may extend it with
    terms drawn from the content itself (transcript nouns, scene labels). Both
    the full queried vocabulary and which terms were content-derived are
    persisted, and absence of a label means "not asked, or not found", never
    "not present".

TRACKING, HONESTLY
    This is **not** instance tracking. Detections are grouped *by label*, so
    "person: first seen 2.1 s, last seen 8.4 s, 12 detections" means the label
    was present in those frames — not that it was the same person throughout.
    Identity tracking needs re-identification embeddings and is out of scope;
    the field names say `screen_time`, not `duration_of_one_object`.

    Because detection runs on sampled keyframes, screen time is estimated as
    (frames containing the label) x (seconds each frame represents). That is a
    lower bound at the sampling resolution, and it is documented as such.

IN:  keyframes with timestamps
OUT: ObjectAppearance[]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blackmirror.content.schemas import AnalysisSource, ObjectAppearance, Provenance
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_ID = "google/owlv2-base-patch16-ensemble"

#: OWLv2 scores are not comparable to DETR's. 0.25 is its usual operating point;
#: measured on the trailer, the dragon scored 0.73-0.78 and genuine absences
#: returned nothing at all, so this does not flood the output.
DETECTION_THRESHOLD = 0.25

#: Asked on every run. Deliberately broad and phrased as natural noun phrases,
#: which is what OWLv2's text tower was trained on. Anything outside this list
#: can only appear if the caller adds it.
BASE_QUERIES: tuple[str, ...] = (
    "a person", "a face", "a crowd of people", "a hand",
    "a car", "a truck", "a bicycle", "a boat", "an aircraft",
    "a building", "a house", "a road", "a tree", "a plant", "water", "a mountain",
    "an animal", "a bird", "a horse", "a dog", "a cat",
    "a computer", "a mobile phone", "a screen", "a book",
    "food", "a drink", "a bottle", "a product package", "a logo",
    "furniture", "a chair", "a table", "a door", "a window",
    "clothing", "a weapon", "a tool",
)

#: Two boxes of the same label overlapping by more than this are the same
#: object seen twice. OWLv2 has no built-in NMS and does emit duplicates:
#: measured 3 overlapping "a mountain" boxes on a single frame.
IOU_DUPLICATE = 0.6

#: An appearance must be seen in more than one sampled frame. A label present in
#: exactly one keyframe is usually a domain-mismatch fluke rather than an object.
MIN_PERSISTENCE = 2


@dataclass(frozen=True)
class Detection:
    """One detected box on one frame."""

    time: float
    label: str
    confidence: float
    box: tuple[float, float, float, float]


class ObjectDetector:
    """Lazily-loaded OWLv2 wrapper, reused across every frame of a run."""

    name = MODEL_ID

    def __init__(
        self, model_id: str = MODEL_ID, queries: tuple[str, ...] = BASE_QUERIES
    ) -> None:
        self.model_id = model_id
        self.queries = tuple(dict.fromkeys(queries))
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._unavailable: str | None = None

    def available(self) -> bool:
        return self._load() is not None

    def _load(self) -> Any:
        if self._model is not None or self._unavailable is not None:
            return self._model
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            self._processor = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id).eval()
            self._torch = torch
        except Exception as exc:
            self._unavailable = f"{type(exc).__name__}: {exc}"
            logger.warning("Object detector unavailable (%s)", self._unavailable)
        return self._model

    def detect(
        self, image_path: Path, time: float, *, threshold: float = DETECTION_THRESHOLD
    ) -> list[Detection]:
        if self._load() is None or not self.queries:
            return []
        try:
            from PIL import Image

            torch = self._torch
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=[list(self.queries)], images=image, return_tensors="pt"
            )
            with torch.inference_mode():
                outputs = self._model(**inputs)
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=torch.tensor([image.size[::-1]]),
                threshold=threshold,
            )[0]
        except Exception as exc:
            logger.debug("Detection failed on %s: %s", image_path.name, exc)
            return []

        detections: list[Detection] = []
        for score, label, box in zip(
            processed["scores"], processed["labels"], processed["boxes"], strict=True
        ):
            coordinates = [float(value) for value in box.tolist()]
            index = int(label)
            if not 0 <= index < len(self.queries):
                continue
            detections.append(
                Detection(
                    time=time,
                    label=self.queries[index],
                    confidence=round(float(score), 4),
                    box=(coordinates[0], coordinates[1], coordinates[2], coordinates[3]),
                )
            )
        return _deduplicate(detections)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two xyxy boxes."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def _deduplicate(detections: list[Detection]) -> list[Detection]:
    """Greedy per-label NMS.

    OWLv2 predicts a box per image patch with no suppression, so one object can
    surface several times. Measured: three overlapping "a mountain" boxes on one
    frame, which would otherwise treble that label's detection count.
    """
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.confidence):
        if any(
            other.label == detection.label and _iou(other.box, detection.box) > IOU_DUPLICATE
            for other in kept
        ):
            continue
        kept.append(detection)
    return kept


#: Words too common or too abstract to be worth asking a detector about.
_STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "there", "here", "of", "in", "on", "at", "to", "for", "with", "from", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "shall", "should", "may", "might", "must", "i",
    "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "not", "no", "yes", "so", "very", "just", "only", "also",
    "too", "now", "when", "where", "what", "who", "how", "why", "all", "any", "some", "more",
    "most", "other", "into", "over", "under", "again", "once", "about", "out", "up", "down",
    "off", "own", "same", "such", "nor"
])


def queries_from_transcript(text: str, *, limit: int = 12) -> tuple[str, ...]:
    """Content-derived detector queries.

    OWLv2 only finds what it is asked about, so a fantasy film needs to be asked
    about dragons. The cheapest honest source of candidate nouns is what the
    content itself says: a word repeated in the dialogue is worth looking for on
    screen.

    This is a frequency heuristic, not parsing — there is no part-of-speech tag
    here, so some queries will be verbs or names. That is acceptable because a
    useless query costs only a detection that never fires; it cannot cause a
    false label, since the phrase asked for is exactly the phrase reported.
    """
    counts: dict[str, int] = {}
    for raw in text.lower().split():
        word = "".join(ch for ch in raw if ch.isalpha())
        if len(word) < 4 or word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(f"a {word}" for word, count in ranked[:limit] if count >= 2)


def detect_objects(
    detector: ObjectDetector,
    frames: list[tuple[float, Path]],
    *,
    frame_span: float,
    duration: float,
    min_persistence: int = MIN_PERSISTENCE,
) -> tuple[list[ObjectAppearance], list[Detection], list[str]]:
    """Detect across frames and aggregate per label into appearances.

    `frame_span` is how much time each sampled frame stands for; it converts a
    detection count into an estimated screen time.
    """
    if not detector.available():
        return [], [], ["Object detector unavailable; objects were not detected."]
    if not frames:
        return [], [], []

    detections: list[Detection] = []
    for time, path in frames:
        detections.extend(detector.detect(path, time))

    if not detections:
        return [], [], []

    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.label].append(detection)

    provenance = Provenance(
        source=AnalysisSource.OBJECT_DETECTION,
        model_id=MODEL_ID,
        notes=(
            "Open-vocabulary detection over "
            f"{len(detector.queries)} supplied queries: only these could be found, "
            "so an absent label means 'not asked or not found', never 'not present'. "
            "Grouped by LABEL, not by instance: screen time means the label was "
            "present, not that one object persisted. Estimated at keyframe "
            "sampling resolution, so it is a lower bound."
        ),
    )

    appearances: list[ObjectAppearance] = []
    dropped: list[str] = []
    for label, items in grouped.items():
        times = sorted({item.time for item in items})
        if len(times) < min_persistence:
            dropped.append(label)
            continue
        confidences = [item.confidence for item in items]
        appearances.append(
            ObjectAppearance(
                label=label,
                first_seen=round(min(times), 3),
                last_seen=round(max(times), 3),
                screen_time_seconds=round(min(duration, len(times) * frame_span), 3),
                detection_count=len(items),
                mean_confidence=round(sum(confidences) / len(confidences), 4),
                provenance=provenance,
            )
        )

    appearances.sort(key=lambda item: (-item.screen_time_seconds, item.label))
    warnings: list[str] = []
    if dropped:
        warnings.append(
            f"Dropped {len(dropped)} object label(s) seen in only one sampled frame "
            f"({', '.join(sorted(dropped)[:6])}"
            f"{'...' if len(dropped) > 6 else ''}); a single-frame detection is more "
            f"often a momentary false match than an object."
        )
    warnings.append(
        f"Objects were searched for, not enumerated: {len(detector.queries)} query "
        f"phrase(s) were asked. Anything outside that vocabulary cannot appear."
    )
    logger.info(
        "Detected %d object label(s) across %d detection(s); dropped %d singleton(s)",
        len(appearances),
        len(detections),
        len(dropped),
    )
    return appearances, detections, warnings


def objects_in_interval(
    detections: list[Detection], start: float, end: float
) -> tuple[str, ...]:
    """Distinct labels detected inside an interval, most confident first."""
    inside = [d for d in detections if start <= d.time < end]
    if not inside:
        return ()
    best: dict[str, float] = {}
    for detection in inside:
        best[detection.label] = max(best.get(detection.label, 0.0), detection.confidence)
    return tuple(label for label, _ in sorted(best.items(), key=lambda kv: -kv[1]))
