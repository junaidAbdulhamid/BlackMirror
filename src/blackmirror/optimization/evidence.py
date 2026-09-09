"""Building measured evidence for one weak interval.

WHY THIS EXISTS SEPARATELY FROM PHASE 5
    Phase 5 reports content-feature differences as a single mean over the whole
    comparison. An optimizer needs the difference *during the interval that
    underperformed* -- "motion was 0.31 here against 0.48 in the reference" --
    which is a different number. The per-time feature matrices are persisted by
    Phase 4, so this module restricts them to the interval and computes the
    delta itself.

CONSISTENCY IS THE POINT OF N VARIANTS
    One reference showing higher motion is an anecdote. Three of three showing
    it is a pattern. `consistency` records the fraction of references that
    differ in the same direction, and it feeds evidence confidence directly.

OVERFITTING GUARD
    Twenty-one features scanned across several intervals will always turn up
    *something*. The number of features examined is recorded on every run, and
    only differences clearing an explicit threshold become evidence, so a
    reader can judge how much selection produced the reported set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackmirror.optimization.schemas import (
    Evidence,
    EvidenceKind,
    ObjectiveGap,
    OptimizationInterval,
)

#: Features that describe availability rather than content. Comparing them
#: would produce evidence about whether analysis ran, not about the content.
_NON_CONTENT = frozenset(
    {"visual_available", "audio_available", "content_event_available"}
)

#: A normalised difference must clear this to become evidence. Below it the
#: difference is within the noise of a short interval.
MIN_RELATIVE_DIFFERENCE = 0.15

#: Below this many prediction samples an interval cannot support content
#: evidence of any kind. At TR = 1 s a one-sample interval is a single second
#: of predicted response, and a content change recommended from it rests on one
#: number. Applied uniformly to feature and event evidence -- they previously
#: disagreed, so a one-sample interval could still produce an event-based
#: recommendation while producing no feature-based one.
MIN_INTERVAL_SAMPLES = 2


@dataclass(frozen=True)
class ContentSeries:
    """One variant's Phase 4 feature matrix on its own clock.

    The matrix and its names are validated against each other on construction.
    Run directories accumulate a feature file per analysis, so it is easy to
    pair a current name list with a stale array; without this check that
    surfaces as an IndexError from inside a loop rather than as the version
    mismatch it actually is.
    """

    variant_id: str
    times: np.ndarray
    matrix: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(f"{self.variant_id}: feature matrix must be [time, feature]")
        if self.matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.variant_id}: feature matrix has {self.matrix.shape[1]} column(s) but "
                f"{len(self.feature_names)} feature name(s) were supplied. The array and the "
                f"name list come from different content-analysis versions; load the array the "
                f"run's metadata references rather than whichever file is on disk."
            )
        if self.matrix.shape[0] != self.times.shape[0]:
            raise ValueError(
                f"{self.variant_id}: feature matrix has {self.matrix.shape[0]} row(s) against "
                f"{self.times.shape[0]} timestamp(s)"
            )

    def interval_means(self, start: float, end: float) -> tuple[dict[str, float], int]:
        """Mean of each feature over [start, end), with the sample count."""
        window = (self.times >= start) & (self.times < end)
        count = int(window.sum())
        if not count:
            return {}, 0
        block = self.matrix[window]
        means: dict[str, float] = {}
        for index, name in enumerate(self.feature_names):
            column = block[:, index]
            finite = column[np.isfinite(column)]
            if finite.size:
                means[name] = float(finite.mean())
        return means, count


def build_feature_evidence(
    interval: OptimizationInterval,
    source: ContentSeries,
    references: list[ContentSeries],
    *,
    prefix: str,
) -> tuple[list[Evidence], int]:
    """Per-feature differences between the source and its references.

    Returns (evidence, features_considered). The second value is the honest
    denominator for "how much was scanned to find this".
    """
    source_means, source_count = source.interval_means(
        interval.start_seconds, interval.end_seconds
    )
    if source_count < MIN_INTERVAL_SAMPLES or not source_means:
        return [], 0

    comparable = [name for name in source.feature_names if name not in _NON_CONTENT]
    reference_means: dict[str, tuple[dict[str, float], int]] = {}
    for reference in references:
        means, count = reference.interval_means(
            interval.start_seconds, interval.end_seconds
        )
        if count >= MIN_INTERVAL_SAMPLES and means:
            reference_means[reference.variant_id] = (means, count)

    if not reference_means:
        return [], len(comparable)

    evidence: list[Evidence] = []
    for position, name in enumerate(comparable):
        if name not in source_means:
            continue
        source_value = source_means[name]
        deltas: dict[str, float] = {}
        for variant_id, (means, _count) in reference_means.items():
            if name in means:
                deltas[variant_id] = means[name] - source_value
        if not deltas:
            continue

        mean_delta = float(np.mean(list(deltas.values())))
        scale = max(abs(source_value), *(abs(v) for v in deltas.values()), 1e-9)
        if abs(mean_delta) / scale < MIN_RELATIVE_DIFFERENCE:
            continue

        # Consistency needs something to be consistent *across*. With one
        # reference the fraction agreeing with the average is 1.0 by
        # construction, which would make a single anecdote look exactly like a
        # three-of-three pattern. Below two references it is undefined.
        agreeing = sum(1 for value in deltas.values() if np.sign(value) == np.sign(mean_delta))
        consistency = agreeing / len(deltas) if len(deltas) >= 2 else None

        evidence.append(
            Evidence(
                evidence_id=f"{prefix}-F{position:02d}",
                kind=EvidenceKind.CONTENT_FEATURE_DELTA,
                description=(
                    f"During {interval.start_seconds:.2f}-{interval.end_seconds:.2f}s, "
                    f"{name} measured {source_value:.4f} in {source.variant_id} against "
                    f"{source_value + mean_delta:.4f} averaged across "
                    f"{len(deltas)} reference variant(s)."
                    + (
                        " Consistency is undefined with a single reference."
                        if len(deltas) < 2
                        else f" {agreeing} of {len(deltas)} references differ in the same "
                        f"direction."
                    )
                ),
                feature=name,
                source_value=round(source_value, 6),
                reference_value=round(source_value + mean_delta, 6),
                delta=round(mean_delta, 6),
                interval_start_seconds=interval.start_seconds,
                interval_end_seconds=interval.end_seconds,
                source_variant_id=source.variant_id,
                reference_variant_ids=tuple(sorted(deltas)),
                consistency=round(consistency, 4) if consistency is not None else None,
                reference_count=len(deltas),
                sample_count=source_count,
                association="temporal_comparison",
            )
        )
    return evidence, len(comparable)


def gap_evidence(gap: ObjectiveGap, prefix: str) -> Evidence | None:
    """The objective gap itself, as a citable fact."""
    if gap.gap is None:
        return None
    return Evidence(
        evidence_id=f"{prefix}-GAP",
        kind=EvidenceKind.OBJECTIVE_GAP,
        description=(
            f"On objective {gap.objective_id} ({gap.direction}), the source measured "
            f"{gap.source_value:.4f} against a reference value of "
            f"{gap.reference_value:.4f}; gap {gap.gap:+.4f} using {gap.formula}."
        ),
        feature=gap.objective_id,
        source_value=gap.source_value,
        reference_value=gap.reference_value,
        delta=gap.gap,
        reference_variant_ids=(gap.reference_variant_id,) if gap.reference_variant_id else (),
        association="measured_value",
    )


def interval_evidence(interval: OptimizationInterval, prefix: str) -> Evidence:
    """The contribution shortfall that made this interval a candidate."""
    reference_note = (
        f" The same wall-clock interval carried "
        f"{interval.reference_contribution_share:.1%} of the reference's contribution."
        if interval.reference_contribution_share is not None
        else ""
    )
    return Evidence(
        evidence_id=f"{prefix}-CONTRIB",
        kind=EvidenceKind.CONTRIBUTION_SHARE,
        description=(
            f"Interval {interval.start_seconds:.2f}-{interval.end_seconds:.2f}s carried "
            f"{interval.contribution_share:.1%} of the objective's total contribution "
            f"magnitude across {interval.sample_count} sample(s)."
            + reference_note
        ),
        feature=interval.objective_id,
        source_value=round(interval.contribution_share, 6),
        reference_value=interval.reference_contribution_share,
        interval_start_seconds=interval.start_seconds,
        interval_end_seconds=interval.end_seconds,
        sample_count=interval.sample_count,
        association="measured_value",
    )


def regional_evidence(
    source_evaluation: object,
    reference_evaluation: object,
    prefix: str,
    *,
    top_k: int = 3,
) -> list[Evidence]:
    """Which regions differ most between source and reference on this objective.

    Only produced for aggregate targets: an ROI objective is already one region,
    so a decomposition would restate the question and Phase 6 returns None for
    it. This is what lets a recommendation say *where* a gap sits rather than
    only *when*.

    Regions are ranked by the size of the difference in their contribution
    share, not by their absolute response, because a region can carry a large
    share in both variants and explain none of the gap.
    """
    source = getattr(source_evaluation, "regional_contribution", None)
    reference = getattr(reference_evaluation, "regional_contribution", None)
    if source is None or reference is None:
        return []

    reference_shares = dict(zip(reference.region_ids, reference.shares, strict=True))
    reference_values = dict(zip(reference.region_ids, reference.values, strict=True))
    names = dict(zip(source.region_ids, source.region_names, strict=True))

    differences: list[tuple[int, float, float, float]] = []
    for region_id, share, value in zip(
        source.region_ids, source.shares, source.values, strict=True
    ):
        if region_id not in reference_shares:
            continue
        differences.append(
            (
                region_id,
                reference_shares[region_id] - share,
                value,
                reference_values[region_id],
            )
        )
    differences.sort(key=lambda item: -abs(item[1]))

    evidence: list[Evidence] = []
    for position, (region_id, share_delta, source_value, reference_value) in enumerate(
        differences[:top_k]
    ):
        name = names.get(region_id, f"region {region_id}")
        evidence.append(
            Evidence(
                evidence_id=f"{prefix}-RG{position:02d}",
                kind=EvidenceKind.REGIONAL_CONTRIBUTION,
                description=(
                    f"Region {name} carried {share_delta:+.1%} more of the objective's "
                    f"magnitude in the reference than in the source (source "
                    f"{source_value:+.4f}, reference {reference_value:+.4f})."
                ),
                feature=f"region:{name}",
                source_value=round(source_value, 6),
                reference_value=round(reference_value, 6),
                delta=round(share_delta, 6),
                association="measured_value",
                note=(
                    "Contribution to an aggregate metric. It does not establish that this "
                    "region is functionally responsible for anything."
                ),
            )
        )
    return evidence


def event_presence_evidence(
    interval: OptimizationInterval,
    source_events: list[dict[str, object]],
    reference_events: dict[str, list[dict[str, object]]],
    prefix: str,
) -> list[Evidence]:
    """Content event types present in references but absent in the source here.

    This is what turns "motion differs" into something actionable: it records
    that, say, no speech_segment overlaps the weak interval in the source while
    it does in the references.
    """
    if interval.sample_count < MIN_INTERVAL_SAMPLES:
        return []

    def kinds(events: list[dict[str, object]]) -> set[str]:
        found: set[str] = set()
        for event in events:
            start = float(event.get("start_time", 0.0))  # type: ignore[arg-type]
            end = float(event.get("end_time", 0.0))  # type: ignore[arg-type]
            if start < interval.end_seconds and end > interval.start_seconds:
                found.add(str(event.get("event_type")))
        return found

    source_kinds = kinds(source_events)
    evidence: list[Evidence] = []
    counts: dict[str, list[str]] = {}
    for variant_id, events in reference_events.items():
        for kind in kinds(events) - source_kinds:
            counts.setdefault(kind, []).append(variant_id)

    for position, (kind, variants) in enumerate(sorted(counts.items())):
        total_references = len(reference_events)
        consistency = (
            len(variants) / total_references if total_references >= 2 else None
        )
        evidence.append(
            Evidence(
                evidence_id=f"{prefix}-E{position:02d}",
                kind=EvidenceKind.CONTENT_EVENT_PRESENCE,
                description=(
                    f"No {kind} event overlaps {interval.start_seconds:.2f}-"
                    f"{interval.end_seconds:.2f}s in {interval.interval_id.split('-')[-1]}'s "
                    f"source variant, while {len(variants)} of {len(reference_events)} "
                    f"reference variant(s) do have one there."
                ),
                feature=f"event:{kind}",
                interval_start_seconds=interval.start_seconds,
                interval_end_seconds=interval.end_seconds,
                reference_variant_ids=tuple(sorted(variants)),
                consistency=round(consistency, 4) if consistency is not None else None,
                reference_count=total_references,
                association="temporal_comparison",
            )
        )
    return evidence
