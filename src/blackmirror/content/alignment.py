"""Content ↔ neural time alignment.

THE TWO CLOCKS
    Content events live on the stimulus's own continuous timeline: a cut happens
    at 13.24 s because that is when the pixels changed.

    Neural predictions live on TRIBE's sparse per-TR grid. Phase 1 established
    that its rows are ALREADY stimulus-aligned (`output_is_stimulus_aligned`):
    training shifted the fMRI recording by `-offset`, so a row's segment start
    is the content time it describes. Both clocks are therefore in the same
    units and origin — but they are not the same *grid*, and Phase 1 also showed
    that up to 96% of neural rows can be dropped.

    So alignment is a real operation, not an identity. It is done explicitly
    here rather than assumed anywhere else.

WHY A CONTEXT WINDOW
    A BOLD response reflects an interval of stimulation, not an instant, and the
    haemodynamic response is smeared over seconds. Asking "what was on screen at
    exactly 13.2 s" is the wrong question; asking "what was happening around
    13.2 s" is the right one. The half-width is explicit and recorded.

WHAT THIS IS NOT
    Association, never causation. Every object produced here carries that
    statement, and the field names say `context`, not `cause`.

IN:  ContentTimeline + Phase 3 NeuralEvents + neural time-series
OUT: NeuralContentAssociation[], FeatureCorrelation[]
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from blackmirror.content.schemas import (
    FeatureCorrelation,
    NeuralContentAssociation,
)
from blackmirror.content.timeline import ContentTimeline
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Half-width of the stimulus interval considered "around" a neural event.
#: 2 s spans roughly one TR either side and reflects BOLD's temporal blur.
DEFAULT_CONTEXT_WINDOW_SECONDS = 2.0

#: Lags searched when exploring whether a content feature leads a neural metric.
#: Positive means the neural series is shifted later — the direction BOLD's
#: delay would predict. Kept coarse: the neural grid is 1 s and short runs give
#: very few samples.
DEFAULT_LAGS_SECONDS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

#: Below this many paired samples a correlation is noise, not a finding.
MIN_CORRELATION_SAMPLES = 8


class ContentNeuralTimeMapper:
    """Deterministic mapping between content time and neural prediction rows.

    Holds the neural row times (Phase 1's `stimulus_time_seconds`) and answers
    both directions without any caller assuming a uniform grid.
    """

    def __init__(
        self,
        neural_times: np.ndarray,
        *,
        tr_seconds: float,
        output_is_stimulus_aligned: bool = True,
        hemodynamic_offset_seconds: float = 0.0,
    ) -> None:
        self.neural_times = np.asarray(neural_times, dtype=np.float64)
        self.tr_seconds = float(tr_seconds)
        self.output_is_stimulus_aligned = output_is_stimulus_aligned
        self.hemodynamic_offset_seconds = float(hemodynamic_offset_seconds)
        if self.neural_times.ndim != 1:
            raise ValueError("neural_times must be one-dimensional")
        if not np.isfinite(self.neural_times).all():
            raise ValueError("neural_times must contain only finite values")
        if self.neural_times.size > 1 and np.any(np.diff(self.neural_times) <= 0):
            raise ValueError("neural_times must be strictly increasing")
        if not np.isfinite(self.tr_seconds) or self.tr_seconds <= 0:
            raise ValueError("tr_seconds must be finite and greater than zero")
        if (
            not np.isfinite(self.hemodynamic_offset_seconds)
            or self.hemodynamic_offset_seconds < 0
        ):
            raise ValueError("hemodynamic_offset_seconds must be finite and non-negative")

    def content_time_for_row(self, index: int) -> float:
        """Stimulus time a neural row describes."""
        if self.neural_times.size == 0:
            return 0.0
        clamped = int(np.clip(index, 0, self.neural_times.size - 1))
        row_time = float(self.neural_times[clamped])
        if self.output_is_stimulus_aligned:
            return row_time
        # A backend that did NOT compensate the lag reports acquisition time, so
        # the driving content is earlier.
        return row_time - self.hemodynamic_offset_seconds

    def nearest_row(self, content_time: float) -> int:
        """Index of the neural row closest to a content time. -1 if none exist."""
        if self.neural_times.size == 0:
            return -1
        shifted = (
            content_time
            if self.output_is_stimulus_aligned
            else content_time + self.hemodynamic_offset_seconds
        )
        return int(np.argmin(np.abs(self.neural_times - shifted)))

    def covering_row(self, content_time: float) -> int:
        """Row whose [start, start + TR) interval contains `content_time`, else -1.

        Mirrors Phase 2's coverage rule: proximity is not coverage, and a gap in
        the neural timeline must be reported rather than papered over.
        """
        if self.neural_times.size == 0:
            return -1
        shifted = (
            content_time
            if self.output_is_stimulus_aligned
            else content_time + self.hemodynamic_offset_seconds
        )
        index = self.nearest_row(shifted if self.output_is_stimulus_aligned else content_time)
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < self.neural_times.size:
                start = float(self.neural_times[candidate])
                if start <= shifted < start + self.tr_seconds:
                    return candidate
        return -1

    def neural_window_for_content(self, start: float, end: float) -> tuple[int, int]:
        """Half-open row range overlapping a content interval."""
        if self.neural_times.size == 0 or end <= start:
            return (0, 0)
        starts = self.neural_times.copy()
        if not self.output_is_stimulus_aligned:
            starts -= self.hemodynamic_offset_seconds
        ends = starts + self.tr_seconds
        mask = (starts < end) & (ends > start)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return (0, 0)
        return (int(indices[0]), int(indices[-1]) + 1)


def associate_neural_events(
    neural_events: list[dict[str, Any]],
    timeline: ContentTimeline,
    *,
    context_window_seconds: float = DEFAULT_CONTEXT_WINDOW_SECONDS,
) -> list[NeuralContentAssociation]:
    """Attach surrounding content context to each Phase 3 neural event.

    `neural_events` are plain dicts read from Phase 3's persisted metadata, so
    this module does not import the analytics package — Phase 4 consumes Phase 3
    through its artifacts, exactly as Phase 2 consumes Phase 1.
    """
    if not np.isfinite(context_window_seconds) or context_window_seconds < 0:
        raise ValueError("context_window_seconds must be finite and non-negative")
    associations: list[NeuralContentAssociation] = []

    for event in neural_events:
        required = ("event_id", "event_type", "timestamp_seconds", "score")
        missing = [field for field in required if field not in event]
        if missing:
            raise ValueError(f"Neural event is missing required fields: {', '.join(missing)}")
        time = float(event["timestamp_seconds"])
        score = float(event["score"])
        if not np.isfinite(time) or time < 0 or not np.isfinite(score):
            raise ValueError(
                "Neural event time and score must be finite; time must be non-negative"
            )
        window_start = max(0.0, time - context_window_seconds)
        window_end = time + context_window_seconds
        nearby = timeline.get_events_between(window_start, window_end)

        associations.append(
            NeuralContentAssociation(
                neural_event_id=str(event["event_id"]),
                neural_event_type=str(event["event_type"]),
                neural_time=round(time, 3),
                neural_score=score,
                window_start=round(window_start, 3),
                window_end=round(window_end, 3),
                context_window_seconds=context_window_seconds,
                content_event_ids=tuple(item.event_id for item in nearby),
                visual_context=_unique(
                    item.visual_description for item in nearby if item.visual_description
                ),
                speech_context=_unique(
                    item.speech_text for item in nearby if item.speech_text
                ),
                on_screen_text_context=_unique(
                    text for item in nearby for text in item.on_screen_text
                ),
                audio_context=_unique(
                    item.audio_description for item in nearby if item.audio_description
                ),
                semantic_context=_unique(
                    item.semantic_description for item in nearby if item.semantic_description
                ),
            )
        )

    logger.info("Built %d neural-content association(s)", len(associations))
    return associations


def _unique(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def explore_correlations(
    content_matrix: np.ndarray,
    content_names: tuple[str, ...],
    content_times: np.ndarray,
    neural_series: dict[str, np.ndarray],
    neural_times: np.ndarray,
    *,
    lags: tuple[float, ...] = DEFAULT_LAGS_SECONDS,
    min_samples: int = MIN_CORRELATION_SAMPLES,
) -> tuple[list[FeatureCorrelation], list[str]]:
    """Exploratory correlations between content features and neural metrics.

    Content is resampled onto the neural grid (never the reverse: the neural
    grid is the sparse, authoritative one). For each pair the best-|r| lag is
    reported alongside the zero-lag value.

    STATISTICAL HONESTY
        Runs here have very few neural samples. With T < 8 no correlation is
        meaningful, so none is emitted. Nothing is corrected for multiple
        comparisons, and every record says so.
    """
    warnings: list[str] = []
    neural_times = np.asarray(neural_times, dtype=np.float64)
    content_times = np.asarray(content_times, dtype=np.float64)
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if neural_times.ndim != 1 or content_times.ndim != 1:
        raise ValueError("content_times and neural_times must be one-dimensional")
    if content_matrix.ndim != 2 or content_matrix.shape[0] != content_times.size:
        raise ValueError("content_matrix rows must match content_times")
    if neural_times.size < min_samples:
        return [], [
            f"Only {int(neural_times.size)} neural samples; correlation needs at least "
            f"{min_samples}. Skipped rather than reporting noise."
        ]
    if content_matrix.size == 0 or content_times.size == 0:
        return [], ["No content features available for correlation."]

    try:
        from scipy import stats
    except ImportError:
        return [], ["scipy unavailable; correlations skipped."]

    results: list[FeatureCorrelation] = []
    all_p_values: list[float] = []
    for index, name in enumerate(content_names):
        if index >= content_matrix.shape[1]:
            break
        column = np.asarray(content_matrix[:, index], dtype=np.float64)
        column_finite = np.isfinite(column) & np.isfinite(content_times)
        if column_finite.sum() < 2:
            continue  # constant feature: correlation is undefined
        source_times = content_times[column_finite]
        source_values = column[column_finite]
        if np.allclose(source_values, source_values[0]):
            continue

        for metric_name, metric_values in neural_series.items():
            metric = np.asarray(metric_values, dtype=np.float64)
            if metric.size != neural_times.size:
                continue

            best: FeatureCorrelation | None = None
            for lag in lags:
                targets = neural_times - lag
                paired = (
                    np.isfinite(targets)
                    & np.isfinite(metric)
                    & (targets >= source_times[0])
                    & (targets <= source_times[-1])
                )
                if paired.sum() < min_samples:
                    continue
                sampled = np.interp(targets[paired], source_times, source_values)
                paired_metric = metric[paired]
                if np.allclose(sampled, sampled[0]) or np.allclose(
                    paired_metric, paired_metric[0]
                ):
                    continue
                coefficient, p_value = stats.pearsonr(sampled, paired_metric)
                if not np.isfinite(coefficient):
                    continue
                candidate = FeatureCorrelation(
                    content_feature=name,
                    neural_metric=metric_name,
                    method="pearson",
                    coefficient=round(float(coefficient), 4),
                    # Selecting the maximum |r| across lags invalidates the
                    # ordinary single-test Pearson p-value. Preserve it only
                    # when no lag selection occurred.
                    # Reported, not suppressed: selecting the best offset biases
                    # this, and so does testing many pairs — but `lags_tested`
                    # and `p_value_bonferroni` let a reader judge, which a bare
                    # None does not.
                    p_value=round(float(p_value), 6),
                    lags_tested=len(lags),
                    sample_count=int(paired.sum()),
                    lag_seconds=lag,
                )
                all_p_values.append(float(p_value))
                if best is None or abs(candidate.coefficient) > abs(best.coefficient):
                    best = candidate
            if best is not None:
                results.append(best)

    # Multiplicity is counted across every test actually performed, not just the
    # reported best-per-pair: hundreds of tests on ~10 samples is the real
    # situation, and hiding it would make weak results look strong.
    comparisons = max(1, len(all_p_values))
    q_by_p = _benjamini_hochberg(all_p_values)
    results = [
        item.model_copy(
            update={
                "comparisons": comparisons,
                "p_value_bonferroni": (
                    round(min(1.0, float(item.p_value) * comparisons), 6)
                    if item.p_value is not None
                    else None
                ),
                "q_value_bh": (
                    q_by_p.get(round(float(item.p_value), 6))
                    if item.p_value is not None
                    else None
                ),
            }
        )
        for item in results
    ]

    results.sort(key=lambda item: abs(item.coefficient), reverse=True)
    survivors = [
        item
        for item in results
        if item.q_value_bh is not None and item.q_value_bh < 0.05
    ]
    if results and not survivors:
        warnings.append(
            f"No correlation survives correction for {comparisons} comparisons "
            f"(neither FDR at q<0.05 nor Bonferroni). Treat every coefficient below "
            f"as descriptive only."
        )
    elif survivors:
        warnings.append(
            f"{len(survivors)} of {len(results)} correlation(s) survive FDR correction "
            f"at q<0.05 across {comparisons} comparisons. Surviving correction means "
            f"the association is unlikely to be noise; it does NOT mean the content "
            f"feature caused the response."
        )
    return results, warnings


def _benjamini_hochberg(p_values: list[float]) -> dict[float, float]:
    """Map each p-value to its BH-adjusted q-value.

    Applied over *every* test performed rather than the reported best-per-pair.
    Correcting only the winners would understate the multiplicity by the number
    of offsets searched, which is exactly the bias that made a raw p=0.0118 look
    meaningful on the real 10-sample run.

    Returns a lookup keyed by the p-value rounded to the same 6 decimal places
    the schema stores, so a persisted result can recover its own q-value. Tests
    that tie on p share a q, which is what BH gives them anyway.
    """
    if not p_values:
        return {}
    total = len(p_values)
    ordered = sorted(p_values)
    adjusted: list[float] = [0.0] * total
    running = 1.0
    # Walk downward so the sequence stays monotone non-decreasing in p.
    for position in range(total - 1, -1, -1):
        value = ordered[position] * total / (position + 1)
        running = min(running, value)
        adjusted[position] = min(1.0, running)
    lookup: dict[float, float] = {}
    for value, q in zip(ordered, adjusted, strict=True):
        key = round(value, 6)
        lookup[key] = min(lookup.get(key, 1.0), round(q, 6))
    return lookup


def rank_associations(
    associations: list[NeuralContentAssociation], top_k: int = 5
) -> list[NeuralContentAssociation]:
    """Strongest neural events first, for the summary view."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    return sorted(associations, key=lambda item: item.neural_score, reverse=True)[:top_k]
