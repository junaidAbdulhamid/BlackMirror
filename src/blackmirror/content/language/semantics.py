"""Language analysis over the transcript.

WHAT IT DOES
    Reads each utterance and returns structured features: is it a question, a
    call to action, does it carry a numeric claim or urgency language, what is
    its tone, and where does it sit in the content's arc.

WHY RULES *AND* EMBEDDINGS
    Two different jobs. Lexical patterns ("buy now", "sign up") are precise,
    auditable, and cheap — a regex match is fully explainable. Embeddings catch
    paraphrase that no pattern list will cover ("why not give it a try today").
    Combining them gives coverage without giving up traceability: every result
    records which mechanism fired.

WHAT TONE IS AND IS NOT
    Tone here describes the CONTENT — how it is written. It is not a claim about
    what any viewer feels. `ContentTone.MOTIVATIONAL` means the language is
    exhortative, not that anyone was motivated.

IN:  TranscriptSegment[] (and on-screen text for CTA detection)
OUT: LanguageAnalysis[], CallToAction[], embeddings
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from blackmirror.content.schemas import (
    AnalysisSource,
    CallToAction,
    ContentTone,
    LanguageAnalysis,
    Provenance,
    StructuralRole,
    TextOverlay,
    TranscriptSegment,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

#: Imperative openers and sign-up language. Ordered most- to least-specific so
#: the reported `matched_pattern` is the informative one.
CTA_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(buy|order|shop|purchase)\s+(now|today|here)\b", "purchase_imperative"),
    (r"\b(sign|signing)\s*up\b|\bsubscribe\b|\bjoin\s+(us|now|today)\b", "signup"),
    (r"\b(download|install|get)\s+(the\s+)?(app|it|yours)\b", "acquire"),
    (r"\b(click|tap|swipe|visit)\s+(the\s+)?(link|here|below|now)\b", "navigate"),
    (r"\b(learn|find\s+out)\s+more\b", "learn_more"),
    (r"\b(start|try|book|call)\s+(now|today|free|us)\b", "start_now"),
    (r"\bdon'?t\s+miss\b|\blimited\s+time\b|\bwhile\s+supplies\s+last\b", "scarcity"),
)

#: Paraphrase net for CTAs that dodge the patterns above.
CTA_EXEMPLARS: tuple[str, ...] = (
    "buy it now", "sign up today", "download the app", "click the link below",
    "learn more about our product", "start your free trial", "order yours today",
    "visit our website", "join us now", "get started today",
)
CTA_EMBEDDING_THRESHOLD = 0.62

URGENCY_TERMS: tuple[str, ...] = (
    "now", "today", "hurry", "fast", "immediately", "instantly", "limited",
    "deadline", "last chance", "don't wait", "act now", "expires",
)

#: Short exemplars per tone for zero-shot-style embedding comparison.
TONE_EXEMPLARS: dict[ContentTone, tuple[str, ...]] = {
    ContentTone.MOTIVATIONAL: (
        "you can do this", "keep going, you are stronger than you think",
        "push yourself further every day",
    ),
    ContentTone.INSTRUCTIONAL: (
        "first, open the lid and remove the cover", "follow these three steps carefully",
        "here is how to set it up",
    ),
    ContentTone.INFORMATIONAL: (
        "the study found a twenty percent increase", "it contains four grams of protein",
        "the service is available in twelve countries",
    ),
    ContentTone.URGENT: (
        "hurry, this offer ends tonight", "only a few left, act now",
        "do not miss this limited time deal",
    ),
    ContentTone.NARRATIVE: (
        "i have been alone for as long as i can remember",
        "when i was a child we lived by the sea", "and then everything changed",
    ),
    ContentTone.CONVERSATIONAL: (
        "so, what do you think about that", "honestly, i was not expecting it",
        "you know how it goes",
    ),
    ContentTone.HUMOROUS: (
        "well that went about as well as expected", "my plan was flawless, obviously",
        "nailed it, sort of",
    ),
}

#: Discourse markers that signal position in an argument or narrative. These are
#: direct textual evidence, unlike clock position, which is only a prior.
DISCOURSE_MARKERS: dict[StructuralRole, tuple[str, ...]] = {
    StructuralRole.INTRODUCTION: (
        "let me tell you", "have you ever", "imagine", "today i want",
        "in this video", "welcome", "hi everyone", "hello everyone",
        "let's talk about", "to begin", "first of all", "it all started",
    ),
    StructuralRole.DEVELOPMENT: (
        "first", "second", "third", "next", "then", "after that", "meanwhile",
        "for example", "because", "however", "on the other hand", "which means",
        "step one", "step two",
    ),
    StructuralRole.CONCLUSION: (
        "finally", "in conclusion", "to sum up", "in summary", "so remember",
        "that's why", "at the end of the day", "thanks for watching",
        "to wrap up", "the bottom line",
    ),
}

#: Short exemplars per role, for paraphrase that dodges the marker list.
ROLE_EXEMPLARS: dict[StructuralRole, tuple[str, ...]] = {
    StructuralRole.INTRODUCTION: (
        "have you ever wondered how this works",
        "today i want to show you something new",
        "it all began on a quiet morning",
    ),
    StructuralRole.DEVELOPMENT: (
        "next, we add the second ingredient and stir",
        "the reason this happens is quite simple",
        "for example, consider what occurs under pressure",
    ),
    StructuralRole.CONCLUSION: (
        "so that is how the whole thing fits together",
        "finally, remember to keep it somewhere safe",
        "thanks for watching, see you next time",
    ),
}

#: Minimum embedding margin for role evidence to outrank the positional prior.
ROLE_EVIDENCE_MARGIN = 0.08

_NUMERIC = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|x|times|dollars?|\$|€|£)?\b")
_QUESTION = re.compile(r"\?\s*$")


class EmbeddingModel:
    """Lazily-loaded sentence embedding model.

    Cached across a run: loading costs seconds, encoding costs milliseconds.
    """

    def __init__(self, model_id: str = EMBEDDING_MODEL_ID) -> None:
        self.model_id = model_id
        self._model: Any = None
        self._unavailable: str | None = None
        self._cache: dict[str, np.ndarray] = {}

    def available(self) -> bool:
        return self._load() is not None

    def _load(self) -> Any:
        if self._model is not None or self._unavailable is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        except Exception as exc:
            self._unavailable = f"{type(exc).__name__}: {exc}"
            logger.warning("Embedding model unavailable (%s)", self._unavailable)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray | None:
        model = self._load()
        if model is None or not texts:
            return None
        missing = [text for text in texts if text not in self._cache]
        if missing:
            vectors = model.encode(
                missing, normalize_embeddings=True, show_progress_bar=False
            )
            for text, vector in zip(missing, np.asarray(vectors), strict=True):
                self._cache[text] = np.asarray(vector, dtype=np.float32)
        return np.stack([self._cache[text] for text in texts])


def analyse_language(
    transcript: list[TranscriptSegment],
    *,
    duration: float,
    embedder: EmbeddingModel | None = None,
) -> tuple[list[LanguageAnalysis], list[str]]:
    """Produce structured language features per transcript segment."""
    warnings: list[str] = []
    if not transcript:
        return [], []

    texts = [segment.text for segment in transcript]
    embeddings = embedder.encode(texts) if embedder else None
    if embedder and embeddings is None:
        warnings.append("Embedding model unavailable; tone and CTA used lexical rules only.")

    tone_matrix = _tone_scores(embedder, embeddings) if embeddings is not None else None
    cta_scores = _cta_scores(embedder, embeddings) if embeddings is not None else None
    role_matrix = _role_scores(embedder, embeddings) if embeddings is not None else None

    analyses: list[LanguageAnalysis] = []
    for position, segment in enumerate(transcript):
        text = segment.text
        lowered = text.lower()

        pattern_hit = _match_cta_pattern(lowered)
        embedding_score = float(cta_scores[position]) if cta_scores is not None else 0.0
        is_cta = pattern_hit is not None or embedding_score >= CTA_EMBEDDING_THRESHOLD
        cta_confidence = round(embedding_score, 4) if pattern_hit is None and is_cta else None

        tone, tone_confidence = (
            _best_tone(tone_matrix[position])
            if tone_matrix is not None
            else (ContentTone.UNKNOWN, None)
        )
        if pattern_hit is not None and tone is ContentTone.UNKNOWN:
            tone = ContentTone.URGENT

        urgency = tuple(
            term
            for term in URGENCY_TERMS
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered)
        )

        role_scores = (
            {
                role: float(role_matrix[position][index])
                for index, role in enumerate(ROLE_EXEMPLARS)
            }
            if role_matrix is not None
            else None
        )
        structural_role, role_reason = _structural_role(
            segment, duration, embedding_scores=role_scores, is_cta=is_cta
        )

        analyses.append(
            LanguageAnalysis(
                segment_index=segment.index,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=text,
                tone=tone,
                tone_confidence=tone_confidence,
                is_question=bool(_QUESTION.search(text)),
                is_call_to_action=is_cta,
                cta_confidence=cta_confidence,
                contains_numeric_claim=bool(_NUMERIC.search(text)),
                urgency_markers=urgency,
                structural_role=structural_role,
                provenance=Provenance(
                    source=(
                        AnalysisSource.SENTENCE_EMBEDDING
                        if tone_matrix is not None
                        else AnalysisSource.LEXICON_RULE
                    ),
                    model_id=EMBEDDING_MODEL_ID if tone_matrix is not None else "regex lexicon",
                    confidence=tone_confidence,
                    notes="; ".join(
                        part
                        for part in (
                            f"CTA matched lexicon pattern: {pattern_hit}."
                            if pattern_hit is not None
                            else None,
                            f"Structural role from {role_reason}.",
                        )
                        if part
                    ),
                ),
            )
        )
    logger.info("Analysed %d transcript segment(s)", len(analyses))
    return analyses, warnings


def detect_calls_to_action(
    analyses: list[LanguageAnalysis],
    overlays: list[TextOverlay],
    *,
    embedder: EmbeddingModel | None = None,
) -> list[CallToAction]:
    """Collect CTAs from speech and on-screen text into one list."""
    calls: list[CallToAction] = []

    for analysis in analyses:
        if not analysis.is_call_to_action:
            continue
        calls.append(
            CallToAction(
                text=analysis.text,
                start_time=analysis.start_time,
                end_time=analysis.end_time,
                modality="speech",
                confidence=analysis.cta_confidence,
                matched_pattern=_match_cta_pattern(analysis.text.lower()),
                provenance=(
                    Provenance(
                        source=AnalysisSource.LEXICON_RULE,
                        model_id="CTA regex lexicon",
                        notes="Deterministic pattern match; no calibrated confidence.",
                    )
                    if _match_cta_pattern(analysis.text.lower())
                    else analysis.provenance
                ),
            )
        )

    if overlays:
        texts = [overlay.text.lower() for overlay in overlays]
        scores = _cta_scores(embedder, embedder.encode(texts)) if embedder else None
        for position, overlay in enumerate(overlays):
            pattern = _match_cta_pattern(texts[position])
            score = float(scores[position]) if scores is not None else 0.0
            if pattern is None and score < CTA_EMBEDDING_THRESHOLD:
                continue
            calls.append(
                CallToAction(
                    text=overlay.text,
                    start_time=overlay.start_time,
                    end_time=overlay.end_time,
                    modality="on_screen_text",
                    confidence=None if pattern else round(score, 4),
                    matched_pattern=pattern,
                    provenance=(
                        Provenance(
                            source=AnalysisSource.LEXICON_RULE,
                            model_id="CTA regex lexicon",
                            notes=(
                                "Deterministic pattern match over OCR text; "
                                "no calibrated confidence."
                            ),
                        )
                        if pattern
                        else Provenance(
                            source=AnalysisSource.SENTENCE_EMBEDDING,
                            model_id=EMBEDDING_MODEL_ID,
                            confidence=round(score, 4),
                        )
                    ),
                )
            )

    return sorted(calls, key=lambda call: call.start_time)


def _match_cta_pattern(lowered: str) -> str | None:
    for pattern, name in CTA_PATTERNS:
        if re.search(pattern, lowered):
            return name
    return None


def _cta_scores(
    embedder: EmbeddingModel | None, embeddings: np.ndarray | None
) -> np.ndarray | None:
    if embedder is None or embeddings is None:
        return None
    exemplars = embedder.encode(list(CTA_EXEMPLARS))
    if exemplars is None:
        return None
    # Max similarity to any exemplar: a CTA need only resemble one of them.
    return (embeddings @ exemplars.T).max(axis=1)


def _tone_scores(
    embedder: EmbeddingModel | None, embeddings: np.ndarray | None
) -> np.ndarray | None:
    if embedder is None or embeddings is None:
        return None
    tones = list(TONE_EXEMPLARS)
    columns = []
    for tone in tones:
        exemplars = embedder.encode(list(TONE_EXEMPLARS[tone]))
        if exemplars is None:
            return None
        columns.append((embeddings @ exemplars.T).max(axis=1))
    return np.stack(columns, axis=1)


def _role_scores(
    embedder: EmbeddingModel | None, embeddings: np.ndarray | None
) -> np.ndarray | None:
    """Similarity of each segment to each structural-role exemplar set."""
    if embedder is None or embeddings is None:
        return None
    columns = []
    for role in ROLE_EXEMPLARS:
        exemplars = embedder.encode(list(ROLE_EXEMPLARS[role]))
        if exemplars is None:
            return None
        columns.append((embeddings @ exemplars.T).max(axis=1))
    return np.stack(columns, axis=1)


def _best_tone(row: np.ndarray) -> tuple[ContentTone, float | None]:
    tones = list(TONE_EXEMPLARS)
    index = int(np.argmax(row))
    score = float(row[index])
    # Below this the "best" tone is barely better than the others; saying
    # UNKNOWN is more honest than reporting a coin flip.
    if score < 0.30:
        return ContentTone.UNKNOWN, round(score, 4)
    return tones[index], round(score, 4)


def _positional_prior(segment: TranscriptSegment, duration: float) -> StructuralRole:
    """Where the utterance sits on the clock. A weak prior, never the answer alone."""
    if duration <= 0:
        return StructuralRole.UNKNOWN
    midpoint = (segment.start_time + segment.end_time) / 2.0
    fraction = midpoint / duration
    if fraction < 0.25:
        return StructuralRole.INTRODUCTION
    if fraction > 0.75:
        return StructuralRole.CONCLUSION
    return StructuralRole.DEVELOPMENT


def _marker_role(text: str) -> tuple[StructuralRole, str] | None:
    """Role signalled by an explicit discourse marker, if any.

    Openers are only trusted at the START of an utterance: "finally" mid-sentence
    ("we finally arrived") is not a concluding marker.
    """
    lowered = text.lower().strip()
    for role, markers in DISCOURSE_MARKERS.items():
        for marker in markers:
            if lowered.startswith(marker) or f", {marker} " in lowered:
                return role, marker
    return None


def _structural_role(
    segment: TranscriptSegment,
    duration: float,
    *,
    embedding_scores: dict[StructuralRole, float] | None = None,
    is_cta: bool = False,
) -> tuple[StructuralRole, str]:
    """Position in the content's arc, from textual evidence where it exists.

    Evidence is preferred over position, in this order:

      1. An explicit discourse marker ("finally", "in this video"). Direct
         textual evidence and fully explainable.
      2. A call to action, which in practice closes a piece of content.
      3. Embedding similarity to role exemplars, but only when one role wins by
         a clear margin — otherwise it is a coin flip dressed as a result.
      4. Clock position, as a fallback prior.

    The reason is returned alongside the label so a surprising result can be
    traced to the rule that produced it. Previously this was position alone,
    which described the timeline rather than the content.
    """
    marker = _marker_role(segment.text)
    if marker is not None:
        role, matched = marker
        return role, f"discourse marker '{matched}'"

    if is_cta:
        return StructuralRole.CONCLUSION, "call to action"

    if embedding_scores:
        ranked = sorted(embedding_scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] >= ROLE_EVIDENCE_MARGIN:
            return ranked[0][0], f"role similarity {ranked[0][1]:.2f}"

    return _positional_prior(segment, duration), "position in timeline (no textual evidence)"
