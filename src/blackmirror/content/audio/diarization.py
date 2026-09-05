"""Speaker diarization over transcript segments.

WHAT IT DOES
    Groups utterances by who spoke them, producing `Speaker 1`, `Speaker 2`, ...

SCOPE — UTTERANCE-LEVEL, NOT FRAME-LEVEL
    Full diarization answers "who is speaking at every instant", including
    overlapped speech. This answers the narrower question "which utterances came
    from the same voice", by embedding each transcript segment and clustering.

    That is the right scope here because the transcript already supplies the
    segmentation, and because every downstream consumer (the UI, associations,
    Phase 5 comparison) works at utterance granularity. Overlapping speakers
    within one utterance are not resolved, and the docs say so.

MODEL
    `speechbrain/spkrec-ecapa-voxceleb` — ECAPA-TDNN trained for speaker
    verification. It maps a waveform to a 192-d embedding where distance
    reflects speaker identity rather than content.

CHOOSING THE SPEAKER COUNT
    Agglomerative clustering with a cosine *distance threshold*, not a fixed k:
    the number of speakers is unknown, and forcing k=2 on a monologue would
    invent a second speaker. A single utterance trivially yields one speaker.

ABSTENTION — WHY THIS MATTERS HERE
    Speaker embeddings degrade badly when speech is mixed under loud music.
    Measured on the Sintel trailer, whose dialogue sits under a full orchestral
    score: the two male utterances were correctly closest (cosine distance
    0.594), but the two female utterances scored 0.811 — indistinguishable from
    the 0.818 between *different* speakers. Clustering that produces four
    "speakers" for two people.

    So the clustering is scored before it is trusted. When within-cluster and
    between-cluster distances do not separate, diarization reports nothing and
    explains why, rather than emitting confident nonsense.

IN:  waveform + transcript segments
OUT: speaker label per segment
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blackmirror.content.schemas import TranscriptSegment
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

#: Cosine distance above which two utterances are treated as different speakers.
#: ECAPA same-speaker pairs typically sit below ~0.25 and different-speaker pairs
#: well above; 0.55 is deliberately conservative — merging two speakers is a
#: less misleading error than inventing one.
DISTANCE_THRESHOLD = 0.55

#: Utterances shorter than this carry too little voice evidence to embed.
MIN_UTTERANCE_SECONDS = 0.6

#: Minimum silhouette-style separation for a clustering to be reported.
#: Below this the embeddings are not carrying speaker identity — usually because
#: speech is buried under music — and any labelling would be arbitrary.
MIN_SEPARATION = 0.10


@dataclass(frozen=True)
class SpeakerAssignment:
    """One utterance's speaker label."""

    segment_index: int
    speaker: str
    confidence: float | None


class SpeakerEmbedder:
    """Lazily-loaded ECAPA speaker encoder."""

    name = MODEL_ID

    def __init__(self, model_id: str = MODEL_ID, cache_dir: Path | None = None) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._model: Any = None
        self._unavailable: str | None = None

    def available(self) -> bool:
        return self._load() is not None

    def _load(self) -> Any:
        if self._model is not None or self._unavailable is not None:
            return self._model
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=self.model_id,
                savedir=str(self.cache_dir) if self.cache_dir else None,
                run_opts={"device": "cpu"},
            )
        except Exception as exc:
            self._unavailable = f"{type(exc).__name__}: {exc}"
            logger.warning("Speaker embedder unavailable (%s)", self._unavailable)
        return self._model

    def embed(self, waveform: np.ndarray) -> np.ndarray | None:
        model = self._load()
        if model is None or waveform.size == 0:
            return None
        try:
            import torch

            tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
            with torch.inference_mode():
                embedding = model.encode_batch(tensor.unsqueeze(0)).squeeze()
            vector = embedding.detach().cpu().numpy().reshape(-1)
        except Exception as exc:
            logger.debug("Speaker embedding failed: %s", exc)
            return None
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None


def diarize(
    embedder: SpeakerEmbedder,
    samples: np.ndarray,
    sample_rate: int,
    transcript: list[TranscriptSegment],
    *,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> tuple[list[SpeakerAssignment], list[str]]:
    """Assign a speaker label to each transcript segment."""
    warnings: list[str] = []
    if not transcript:
        return [], []
    if samples.size == 0:
        return [], ["No audio available for diarization."]
    if not embedder.available():
        return [], ["Speaker embedder unavailable; utterances were not diarized."]

    usable: list[tuple[int, np.ndarray]] = []
    for segment in transcript:
        if segment.end_time - segment.start_time < MIN_UTTERANCE_SECONDS:
            continue
        start = int(segment.start_time * sample_rate)
        end = min(samples.size, int(segment.end_time * sample_rate))
        if end <= start:
            continue
        vector = embedder.embed(samples[start:end])
        if vector is not None:
            usable.append((segment.index, vector))

    if not usable:
        return [], ["No utterance was long enough to embed for diarization."]

    if len(usable) == 1:
        # One utterance cannot evidence more than one speaker.
        return [SpeakerAssignment(usable[0][0], "Speaker 1", None)], warnings

    matrix = np.stack([vector for _, vector in usable])
    labels = _cluster(matrix, distance_threshold)

    separation = _separation(matrix, labels)
    if len(set(labels)) > 1 and separation < MIN_SEPARATION:
        warnings.append(
            f"Diarization abstained: speaker embeddings separated by only "
            f"{separation:.3f} (need {MIN_SEPARATION}). Speech mixed under loud music "
            f"or noise degrades speaker embeddings, and any labelling here would be "
            f"arbitrary."
        )
        logger.info("Diarization abstained (separation %.3f)", separation)
        return [], warnings

    # Number speakers by first appearance so labels read in transcript order.
    order: dict[int, int] = {}
    assignments: list[SpeakerAssignment] = []
    for (index, _), cluster in zip(usable, labels, strict=True):
        if cluster not in order:
            order[cluster] = len(order) + 1
        assignments.append(
            SpeakerAssignment(
                segment_index=index, speaker=f"Speaker {order[cluster]}", confidence=None
            )
        )

    skipped = len(transcript) - len(assignments)
    if skipped > 0:
        warnings.append(
            f"{skipped} utterance(s) were too short (<{MIN_UTTERANCE_SECONDS}s) to diarize "
            f"and carry no speaker label."
        )
    if assignments:
        warnings.append(
            "Each utterance is assigned exactly one speaker; simultaneous speech is "
            "not detected. A mixed utterance is labelled with whichever voice "
            "dominates it, silently. Measured: a two-speaker mixture embedded 0.23 "
            "from the dominant speaker and 0.81 from the other."
        )
    logger.info("Diarized %d utterance(s) into %d speaker(s)", len(assignments), len(order))
    return assignments, warnings


def _separation(matrix: np.ndarray, labels: list[int]) -> float:
    """How much better within-cluster distances are than between-cluster ones.

    A silhouette-style score in roughly [-1, 1]. Near zero means the clustering
    carries no real structure — the vectors are equally far from everything.
    """
    unique = set(labels)
    if len(unique) < 2 or matrix.shape[0] < 3:
        return 1.0
    distances = 1.0 - (matrix @ matrix.T)
    scores: list[float] = []
    for index, label in enumerate(labels):
        same = [
            distances[index, other]
            for other, other_label in enumerate(labels)
            if other != index and other_label == label
        ]
        other = [
            distances[index, other]
            for other, other_label in enumerate(labels)
            if other_label != label
        ]
        if not same or not other:
            continue
        a = float(np.mean(same))
        b = float(np.min(other))
        denominator = max(a, b)
        if denominator > 0:
            scores.append((b - a) / denominator)
    return float(np.mean(scores)) if scores else 0.0


def _cluster(matrix: np.ndarray, threshold: float) -> list[int]:
    """Agglomerative clustering on cosine distance.

    scipy where available; otherwise a simple single-pass fallback so a missing
    optional dependency degrades rather than fails.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        distances = pdist(matrix, metric="cosine")
        if distances.size == 0:
            return [0] * matrix.shape[0]
        # Average linkage: robust to one outlying utterance, unlike single
        # linkage which chains distinct speakers together through a midpoint.
        tree = linkage(distances, method="average")
        return [int(value) for value in fcluster(tree, t=threshold, criterion="distance")]
    except ImportError:
        return _greedy_cluster(matrix, threshold)


def _greedy_cluster(matrix: np.ndarray, threshold: float) -> list[int]:
    """Assign each vector to the first centroid within threshold."""
    centroids: list[np.ndarray] = []
    labels: list[int] = []
    for vector in matrix:
        placed = False
        for index, centroid in enumerate(centroids):
            if 1.0 - float(np.dot(vector, centroid)) <= threshold:
                labels.append(index)
                centroids[index] = (centroid + vector) / np.linalg.norm(centroid + vector)
                placed = True
                break
        if not placed:
            centroids.append(vector)
            labels.append(len(centroids) - 1)
    return labels


def apply_speakers(
    transcript: list[TranscriptSegment], assignments: list[SpeakerAssignment]
) -> list[TranscriptSegment]:
    """Return the transcript with speaker labels attached."""
    by_index = {item.segment_index: item.speaker for item in assignments}
    return [
        segment.model_copy(update={"speaker": by_index.get(segment.index)})
        for segment in transcript
    ]
