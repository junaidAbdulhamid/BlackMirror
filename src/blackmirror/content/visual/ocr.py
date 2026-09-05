"""On-screen text detection.

WHAT IT DOES
    Reads text burned into the video — captions, titles, prices, CTAs — and
    reports each distinct string once with the interval it was visible.

WHY TEMPORAL DEDUPLICATION IS THE HARD PART
    OCR runs per frame. A title on screen for three seconds at 2 fps produces
    six near-identical detections, and OCR noise means they are rarely byte-
    identical ("50% OFF" / "50%OFF" / "5O% OFF"). Emitting six overlays would
    inflate every text metric sixfold and make two variants look different when
    they are not.

    So consecutive detections are merged when their normalised text is similar
    enough, producing one overlay with a start, an end, and an occurrence count.

MODEL
    RapidOCR (ONNXRuntime PP-OCRv4). Chosen over EasyOCR/Tesseract because it is
    small, CPU-fast, needs no system packages, and returns per-detection
    confidence.

IN:  keyframe images with timestamps
OUT: TextOverlay[]
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blackmirror.content.schemas import AnalysisSource, Provenance, TextOverlay
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Below this, detections are usually texture misread as glyphs.
MIN_CONFIDENCE = 0.5

#: Similarity above which two consecutive detections are the same overlay.
MERGE_SIMILARITY = 0.72

#: Single characters are almost always noise; require some substance.
MIN_TEXT_LENGTH = 2

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


@dataclass
class _Group:
    """An overlay being accumulated across consecutive frames."""

    key: str
    text: str
    first_time: float
    last_time: float
    count: int = 1
    best_confidence: float = 0.0
    confidences: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class FrameText:
    """One OCR detection on one frame."""

    time: float
    text: str
    confidence: float


#: Recognition models by script. The default ships with the package and covers
#: Chinese, Latin and digits. The others are fetched on demand and are PP-OCRv1
#: era, so they take a 32-pixel input height where v4 takes 48; passing the wrong
#: shape raises ONNX "Got: 48 Expected: 32" rather than degrading.
#:
#: Only `japanese` is verified. Measured on rendered text:
#:   default  on Japanese -> "世界" at confidence 1.00, silently dropping every
#:                           kana character, because they are not in its dictionary
#:   japanese on Japanese -> "こんにちは世界" correct
#:   korean   on Korean   -> nothing at all, so it is not offered
#: Cyrillic, Arabic, Devanagari and Hebrew have no published RapidOCR model.
OCR_SCRIPTS: dict[str, tuple[str | None, tuple[int, int, int]]] = {
    "default": (None, (3, 48, 320)),
    "japanese": ("PP-OCRv1/japan_rec_crnn.onnx", (3, 32, 320)),
}

#: Where the on-demand recognition models come from.
OCR_MODEL_REPO = "SWHL/RapidOCR"


class OcrReader:
    """Lazily-initialised RapidOCR wrapper.

    Loading the ONNX models costs a couple of seconds, so the instance is reused
    across every frame of a run rather than rebuilt per call.

    `script` selects the recognition model and must be set explicitly.
    Auto-detection is deliberately not offered: it would have to pick a model by
    comparing confidences across models, and that was measured to be unreliable
    — the default model rates its *incomplete* Japanese reading at 1.00, exactly
    as high as the correct reading from the Japanese model. A selector built on
    that signal would confidently choose the wrong model.
    """

    def __init__(self, script: str = "default") -> None:
        if script not in OCR_SCRIPTS:
            raise ValueError(
                f"unknown OCR script {script!r}; available: {sorted(OCR_SCRIPTS)}"
            )
        self.script = script
        self._engine: Any | None = None
        self._unavailable: str | None = None

    def available(self) -> bool:
        return self._load() is not None

    def _load(self) -> Any | None:
        if self._engine is not None or self._unavailable is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR

            filename, shape = OCR_SCRIPTS[self.script]
            if filename is None:
                self._engine = RapidOCR()
            else:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(OCR_MODEL_REPO, filename)
                self._engine = RapidOCR(
                    rec_model_path=path, rec_img_shape=list(shape)
                )
                logger.info("OCR using the %s recognition model", self.script)
        except Exception as exc:
            self._unavailable = f"{type(exc).__name__}: {exc}"
            logger.warning("OCR unavailable (%s)", self._unavailable)
        return self._engine

    def read_frame(self, image_path: Path, time: float) -> list[FrameText]:
        engine = self._load()
        if engine is None:
            return []
        try:
            result, _ = engine(str(image_path))
        except Exception as exc:
            logger.debug("OCR failed on %s: %s", image_path.name, exc)
            return []
        if not result:
            return []

        detections: list[FrameText] = []
        for entry in result:
            # RapidOCR returns [box, text, confidence].
            if len(entry) < 3:
                continue
            text = str(entry[1]).strip()
            try:
                confidence = float(entry[2])
            except (TypeError, ValueError):
                continue
            if len(text) < MIN_TEXT_LENGTH or confidence < MIN_CONFIDENCE:
                continue
            detections.append(FrameText(time=time, text=text, confidence=confidence))
        return detections


def detect_text_overlays(
    reader: OcrReader,
    frames: list[tuple[float, Path]],
    *,
    duration: float,
    frame_span: float,
) -> tuple[list[TextOverlay], list[str]]:
    """Run OCR across sampled frames and merge repeats into overlays.

    `frame_span` is how long each sampled frame is taken to represent; it sets
    the end time of an overlay seen on only one frame.
    """
    warnings: list[str] = []
    if not reader.available():
        return [], ["OCR engine unavailable; on-screen text was not analysed."]

    detections: list[FrameText] = []
    for time, path in frames:
        detections.extend(reader.read_frame(path, time))

    if not detections:
        return [], warnings

    overlays = _merge(detections, duration=duration, frame_span=frame_span)
    logger.info(
        "Detected %d text overlay(s) from %d raw detection(s)",
        len(overlays),
        len(detections),
    )
    return overlays, warnings


def _normalise(text: str) -> str:
    """Casefold and strip punctuation so OCR jitter does not split an overlay."""
    lowered = _NON_ALNUM.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _merge(
    detections: list[FrameText], *, duration: float, frame_span: float
) -> list[TextOverlay]:
    """Group detections into overlays by text similarity over time."""
    provenance = Provenance(
        source=AnalysisSource.RAPID_OCR,
        model_id="rapidocr-onnxruntime (PP-OCRv4)",
        notes="Consecutive similar detections merged into one overlay.",
    )

    groups: list[_Group] = []
    for detection in sorted(detections, key=lambda d: d.time):
        key = _normalise(detection.text)
        if not key:
            continue
        placed = False
        for group in groups:
            # Only merge into a group still "open" at this time — the same word
            # reappearing much later is a separate overlay.
            if detection.time - group.last_time > frame_span * 2.5:
                continue
            if _similar(key, group.key) >= MERGE_SIMILARITY:
                group.last_time = detection.time
                group.count += 1
                group.confidences.append(detection.confidence)
                # Keep the highest-confidence rendering as the display text.
                if detection.confidence > group.best_confidence:
                    group.text = detection.text
                    group.best_confidence = detection.confidence
                placed = True
                break
        if not placed:
            groups.append(
                _Group(
                    key=key,
                    text=detection.text,
                    first_time=detection.time,
                    last_time=detection.time,
                    best_confidence=detection.confidence,
                    confidences=[detection.confidence],
                )
            )

    overlays: list[TextOverlay] = []
    for group in groups:
        end = min(duration, group.last_time + frame_span)
        overlays.append(
            TextOverlay(
                text=group.text,
                start_time=round(group.first_time, 3),
                end_time=round(max(end, group.first_time), 3),
                confidence=round(sum(group.confidences) / len(group.confidences), 4),
                occurrences=group.count,
                provenance=provenance,
            )
        )
    return sorted(overlays, key=lambda o: o.start_time)
