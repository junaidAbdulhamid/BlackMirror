"""Visual semantic analysis.

WHAT IT DOES
    Looks at a keyframe and returns STRUCTURED attributes — setting, subjects,
    actions, shot scale, whether a person is present — each with a score.

WHY NOT A CAPTIONING MODEL OR A BIG VLM PROMPT
    Phase 4's design rule is structured, traceable, diffable data. A caption
    ("a man walks through a desert at sunset") is one sentence that Phase 5
    would have to parse back into fields, and two near-identical frames can
    produce very differently-worded captions. Zero-shot classification against a
    fixed label set gives the same fields every time with comparable scores,
    which is exactly what variant comparison needs.

HOW ZERO-SHOT CLASSIFICATION WORKS
    CLIP embeds images and text into one shared space. Scoring an image against
    "a photo of a gym" vs "a photo of a beach" is a cosine similarity in that
    space — no training, no fine-tuning. Softmax over a group of candidate
    labels turns those similarities into a ranking with confidences.

    The scores are RELATIVE TO THE LABEL SET. A frame scored against settings
    always returns *some* best setting, even when none fits.

ABSTENTION — WHY SOFTMAX CONFIDENCE WAS NOT ENOUGH
    The Sintel title card was originally reported as "a store or shop" at 0.62
    softmax confidence: confidently wrong, because no label described a title
    card. Measurement showed the failure is not fixable by calibration —
    CLIP's raw similarity for that frame (0.229) was *higher* than for a
    correctly-labelled dialogue frame (0.219), so no absolute threshold
    separates them.

    Two changes follow from that measurement:

      * Confidence is derived from the **top-1/top-2 raw cosine margin**, not
        softmax. Softmax must sum to 1 across a closed set, so it reports high
        confidence for a forced choice among equally-bad options. The margin
        asks a better question: did the model actually prefer this label?
        Below `SETTING_MARGIN_ABSTAIN` the setting is reported as None.

      * Frame *kind* (photographic / graphic / blank) is decided from evidence —
        OCR text coverage and image entropy — not from a CLIP prompt. Asking
        CLIP "is this a title card?" was measured to misfire on dark
        photographic frames, labelling a wilderness shot a title card at 0.95.
        A graphic frame has no meaningful "setting", so the field is suppressed.

BACKEND IS PLUGGABLE
    `VisualSemanticsBackend` is a protocol. A cloud VLM can replace CLIP later
    without touching fusion, storage, or the API — the same seam Phase 1 used
    for prediction models.

IN:  keyframe images
OUT: VisualAttributes[]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from blackmirror.content.schemas import (
    AnalysisSource,
    FrameKind,
    Provenance,
    VisualAttributes,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"

#: Label groups. Each is scored independently with its own softmax, because
#: "is this a gym or a beach" and "is this a close-up or a wide shot" are
#: separate questions and must not compete for probability mass.
LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "setting": (
        "indoors", "outdoors", "a city street", "nature or wilderness", "a desert",
        "a gym or fitness centre", "an office or workplace", "a home interior",
        "a store or shop", "a studio with a plain background", "a stage or venue",
    ),
    "shot_scale": (
        "an extreme close-up of a face", "a close-up shot", "a medium shot",
        "a wide establishing shot", "an aerial or drone shot",
    ),
    "action": (
        "a person talking to camera", "people walking or running", "a person fighting",
        "a person working or typing", "a person exercising", "hands using a product",
        "a vehicle moving", "a mostly static scene", "an animated or rendered scene",
    ),
    "lighting": ("bright daylight", "low light or night", "warm golden light", "cool blue light"),
}

#: Yes/no questions scored as a two-way contrast, which is more reliable than
#: thresholding a single similarity.
BINARY_QUESTIONS: dict[str, tuple[str, str]] = {
    "person_present": ("a photo containing a person", "a photo with no people in it"),
    "text_present": ("a frame with text or captions on screen", "a frame with no text on it"),
}

#: Reported as candidate objects when they clear this bar.
OBJECT_LABELS: tuple[str, ...] = (
    "a person", "a face", "a crowd of people", "a car or vehicle", "a building",
    "a computer or phone", "a product package", "food or drink", "an animal",
    "a weapon or sword", "trees or plants", "water", "sky or clouds", "a logo or brand mark",
)
OBJECT_THRESHOLD = 0.16

#: Prompts describing nothing visual. They establish what a *non-answer* scores
#: on this image, which is the quantity a closed label set cannot supply: the
#: top-two margin says whether the model can tell its labels apart, and says
#: nothing about whether any of them fit at all.
NULL_PROMPTS: tuple[str, ...] = (
    "a quadratic equation", "the smell of rain", "a legal contract clause",
    "an abstract concept", "the number seventeen", "a philosophical argument",
    "a musical key signature", "an unrelated phrase", "a random assortment",
    "the year 1834", "a chemical formula", "an untranslated word",
)

#: How far the best real label must beat the best null prompt. Calibrated
#: against eight ground-truthed frames from the Sintel trailer: the one frame
#: whose label was actually correct (a desert) cleared the null bank by 0.0676,
#: while every incorrect label cleared it by at most 0.0096.
NULL_MARGIN_ABSTAIN = 0.02

#: Minimum top-1/top-2 raw cosine gap for a setting label to be reported.
#: Calibrated against measured frames: a correctly-labelled dialogue shot scored
#: 0.0248 while forced guesses on out-of-distribution frames scored 0.0026-0.0144.
SETTING_MARGIN_ABSTAIN = 0.018

#: Margin at which confidence saturates to 1.0.
SETTING_MARGIN_CONFIDENT = 0.060

#: Image entropy below which a frame carries almost no pictorial detail.
GRAPHIC_ENTROPY = 1.2
BLANK_ENTROPY = 0.05


def classify_frame_kind(
    *, entropy: float | None, saturation: float | None, has_text: bool
) -> FrameKind:
    """Decide what sort of frame this is from measured evidence.

    Entropy alone is not enough: a dark photographic frame and a title card both
    score low (measured 0.47 vs 0.26 on real content). Text evidence from OCR is
    what separates them — a title card is low-detail *and* carries text.
    """
    if entropy is None:
        return FrameKind.UNKNOWN
    if entropy <= BLANK_ENTROPY:
        return FrameKind.BLANK
    if entropy < GRAPHIC_ENTROPY and has_text:
        return FrameKind.GRAPHIC
    # Near-zero saturation with almost no detail is a monochrome caption card
    # even when OCR failed to read it.
    if entropy < GRAPHIC_ENTROPY and saturation is not None and saturation < 0.02:
        return FrameKind.GRAPHIC
    return FrameKind.PHOTOGRAPHIC


def margin_confidence(margin: float) -> float:
    """Map a raw cosine margin onto [0, 1].

    Linear between the abstain and saturation thresholds. Deliberately not a
    softmax: this reports how much the model preferred its answer, which is the
    quantity a reader actually needs.
    """
    if margin <= SETTING_MARGIN_ABSTAIN:
        return 0.0
    span = SETTING_MARGIN_CONFIDENT - SETTING_MARGIN_ABSTAIN
    return round(min(1.0, (margin - SETTING_MARGIN_ABSTAIN) / span), 4)


class VisualSemanticsBackend(Protocol):
    """Anything that can describe a keyframe with structured attributes."""

    name: str

    def available(self) -> bool: ...

    def describe(
        self,
        image_path: Path,
        time: float,
        *,
        frame_kind: FrameKind = FrameKind.UNKNOWN,
    ) -> VisualAttributes | None: ...


class ClipSemanticsBackend:
    """Zero-shot structured description using CLIP."""

    name = MODEL_ID

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None
        self._text_cache: dict[str, Any] = {}
        self._unavailable: str | None = None

    def available(self) -> bool:
        return self._load() is not None

    def _load(self) -> Any:
        if self._model is not None or self._unavailable is not None:
            return self._model
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self.model_id).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._torch = torch
        except Exception as exc:
            self._unavailable = f"{type(exc).__name__}: {exc}"
            logger.warning("CLIP unavailable (%s)", self._unavailable)
        return self._model

    @staticmethod
    def _embedding(result: Any) -> Any:
        """Extract the projected embedding tensor from a CLIP call.

        transformers 4.x returns a bare tensor from `get_*_features`; 5.x wraps
        it in a `BaseModelOutputWithPooling` whose `pooler_output` holds the
        projected embedding. Supporting both keeps the pipeline working across
        the version range the rest of the project already spans.
        """
        for attribute in ("image_embeds", "text_embeds", "pooler_output"):
            candidate = getattr(result, attribute, None)
            if candidate is not None:
                return candidate
        return result

    def _text_features(self, prompts: tuple[str, ...]) -> Any:
        """Embed and cache a prompt group.

        Text embeddings are constant across every frame of every run, so
        computing them once per group is a large saving on long videos.
        """
        key = "|".join(prompts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        torch = self._torch
        inputs = self._processor(
            text=[f"a photo of {p}" for p in prompts], return_tensors="pt", padding=True
        )
        with torch.inference_mode():
            features = self._embedding(self._model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        self._text_cache[key] = features
        return features

    def describe(
        self,
        image_path: Path,
        time: float,
        *,
        frame_kind: FrameKind = FrameKind.UNKNOWN,
    ) -> VisualAttributes | None:
        if self._load() is None:
            return None
        try:
            from PIL import Image

            torch = self._torch
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                image_features = self._embedding(self._model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as exc:
            logger.debug("CLIP failed on %s: %s", image_path.name, exc)
            return None

        labels: dict[str, float] = {}
        best: dict[str, tuple[str, float, float]] = {}
        for group, prompts in LABEL_GROUPS.items():
            scores, raw = self._score_pair(image_features, prompts)
            for prompt, score in zip(prompts, scores, strict=True):
                labels[f"{group}:{prompt}"] = round(float(score), 4)
            order = sorted(range(len(prompts)), key=lambda i: raw[i], reverse=True)
            top, second = order[0], order[1] if len(order) > 1 else order[0]
            margin = float(raw[top] - raw[second])
            best[group] = (prompts[top], float(scores[top]), margin)

        binaries: dict[str, bool] = {}
        for question, (positive, negative) in BINARY_QUESTIONS.items():
            scores = self._score(image_features, (positive, negative))
            binaries[question] = bool(scores[0] > scores[1])
            labels[f"binary:{question}"] = round(float(scores[0]), 4)

        object_scores = self._score(image_features, OBJECT_LABELS)
        objects = tuple(
            label.removeprefix("a ").removeprefix("an ")
            for label, score in zip(OBJECT_LABELS, object_scores, strict=True)
            if score >= OBJECT_THRESHOLD
        )
        for label, score in zip(OBJECT_LABELS, object_scores, strict=True):
            labels[f"object:{label}"] = round(float(score), 4)

        setting, _, setting_margin = best["setting"]

        # How much better the best setting label scores than the best of a bank
        # of prompts describing nothing visual at all.
        _, setting_raw = self._score_pair(image_features, LABEL_GROUPS["setting"])
        _, null_raw = self._score_pair(image_features, NULL_PROMPTS)
        null_gap = float(max(setting_raw) - max(null_raw)) if null_raw else 1.0

        action, _, _ = best["action"]
        shot_scale, _, _ = best["shot_scale"]
        lighting, _, _ = best["lighting"]

        # Two independent reasons to withhold a setting label.
        abstained = False
        abstain_reason: str | None = None
        confidence = margin_confidence(setting_margin)

        if frame_kind in (FrameKind.GRAPHIC, FrameKind.BLANK):
            abstained = True
            abstain_reason = (
                f"frame is {frame_kind.value}; a scene setting is not a meaningful "
                f"property of it"
            )
        elif setting_margin < SETTING_MARGIN_ABSTAIN:
            abstained = True
            abstain_reason = (
                f"top-two label margin {setting_margin:.4f} < {SETTING_MARGIN_ABSTAIN}; "
                f"the classifier had no real preference among the label set"
            )
        elif null_gap < NULL_MARGIN_ABSTAIN:
            # The margin test and this one fail on different frames, which is
            # why both are applied. Measured on the Sintel trailer: the margin
            # test catches the title card (margin 0.0133) that this one passes,
            # and this one catches a person-in-a-cave close-up confidently
            # labelled "a desert" (margin 0.0192, null gap 0.0069) that the
            # margin test passes. Neither alone catches both.
            abstained = True
            abstain_reason = (
                f"best label beat unrelated control phrases by only {null_gap:.4f} "
                f"(< {NULL_MARGIN_ABSTAIN}); the image matches no label in the set "
                f"much better than it matches nonsense"
            )

        return VisualAttributes(
            time=round(time, 3),
            keyframe_path=None,  # filled in by the caller, which owns run-relative paths
            frame_kind=frame_kind,
            setting=None if abstained else setting,
            setting_confidence=None if abstained else confidence,
            setting_margin=round(setting_margin, 5),
            setting_null_gap=round(null_gap, 5),
            abstained=abstained,
            abstain_reason=abstain_reason,
            subjects=("person",) if binaries.get("person_present") else (),
            actions=(action,),
            objects=objects,
            person_present=binaries.get("person_present"),
            text_present=binaries.get("text_present"),
            shot_scale=_shorten_scale(shot_scale),
            labels=labels,
            provenance=Provenance(
                source=AnalysisSource.CLIP_ZERO_SHOT,
                model_id=self.model_id,
                confidence=None if abstained else confidence,
                notes=(
                    f"Zero-shot over a fixed label set; lighting={lighting}. Confidence is "
                    f"the top-two cosine margin, not softmax. Scores are relative to the "
                    f"label set, not absolute recognition."
                ),
            ),
        )


    def _score(self, image_features: Any, prompts: tuple[str, ...]) -> list[float]:
        """Softmax scores, for relative ranking within one label group."""
        return self._score_pair(image_features, prompts)[0]

    def _score_pair(
        self, image_features: Any, prompts: tuple[str, ...]
    ) -> tuple[list[float], list[float]]:
        """Both softmax scores and the RAW cosine similarities.

        The raw values are what the abstention margin is computed from; softmax
        discards exactly the information abstention needs, because it must sum
        to 1 across a closed label set however poorly those labels fit.
        """
        torch = self._torch
        text_features = self._text_features(prompts)
        with torch.inference_mode():
            similarity = (image_features @ text_features.T).squeeze(0)
            probabilities = torch.softmax(similarity * 100.0, dim=-1)
        return (
            [float(value) for value in probabilities],
            [float(value) for value in similarity],
        )


def _shorten_scale(prompt: str) -> str:
    if "extreme close-up" in prompt:
        return "extreme_close_up"
    if "close-up" in prompt:
        return "close_up"
    if "medium" in prompt:
        return "medium"
    if "wide" in prompt:
        return "wide"
    if "aerial" in prompt:
        return "aerial"
    return "unknown"


def describe_frames(
    backend: VisualSemanticsBackend,
    frames: list[tuple[float, Path]],
    *,
    relative_paths: dict[Path, str] | None = None,
    frame_kinds: dict[float, FrameKind] | None = None,
) -> tuple[list[VisualAttributes], list[str]]:
    """Describe every sampled keyframe.

    `frame_kinds` carries the evidence-based classification from the pipeline,
    which owns the entropy signal and the OCR results.
    """
    if not backend.available():
        return [], ["Visual semantics backend unavailable; keyframes were not described."]

    described: list[VisualAttributes] = []
    for time, path in frames:
        kind = (frame_kinds or {}).get(time, FrameKind.UNKNOWN)
        attributes = backend.describe(path, time, frame_kind=kind)
        if attributes is None:
            continue
        if relative_paths and path in relative_paths:
            attributes = attributes.model_copy(
                update={"keyframe_path": relative_paths[path]}
            )
        described.append(attributes)

    warnings: list[str] = []
    if len(described) < len(frames):
        warnings.append(
            f"Visual semantics succeeded on {len(described)} of {len(frames)} keyframes."
        )
    logger.info("Described %d keyframe(s)", len(described))
    return described, warnings
