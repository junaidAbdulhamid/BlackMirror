"""The optimization pipeline.

A CONTROLLED PIPELINE, NOT AN AGENT LOOP
    Each stage has one job and a typed output, so a bad recommendation can be
    traced to the stage that produced it:

        gap analysis -> reference selection -> interval detection
        -> evidence  -> generation -> validation -> dedup -> ranking
        -> hypotheses -> traces

    There is no reasoning loop and no iteration limit to tune, because there is
    no iteration. Everything here is deterministic: the same context produces
    the same evidence graph and the same ranking, which is what makes Phase 8's
    eventual pass/fail attributable to the recommendation rather than to a
    sampling seed.
"""

from __future__ import annotations

import numpy as np

from blackmirror.optimization.evidence import (
    ContentSeries,
    build_feature_evidence,
    event_presence_evidence,
    gap_evidence,
    interval_evidence,
    regional_evidence,
)
from blackmirror.optimization.intervals import (
    analyse_gap,
    detect_intervals,
    select_references,
)
from blackmirror.optimization.recommend import generate
from blackmirror.optimization.schemas import (
    OPTIMIZATION_VERSION,
    Evidence,
    IntervalKind,
    OptimizationHypothesis,
    OptimizationMetadata,
    OptimizationRecommendation,
    OptimizationRequest,
    OptimizationResult,
    OptimizationTrace,
)
from blackmirror.optimization.validate import deduplicate, rank, validate
from blackmirror.scoring.schemas import ExperimentScoreResult, VariantScore
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationOrchestrator:
    def optimize(
        self,
        request: OptimizationRequest,
        score: ExperimentScoreResult,
        *,
        content: dict[str, ContentSeries] | None = None,
        events: dict[str, list[dict[str, object]]] | None = None,
    ) -> OptimizationResult:
        warnings: list[str] = []
        content = content or {}
        events = events or {}

        source = _variant(score, request.source_variant_id)
        if source is None:
            raise ValueError(
                f"variant {request.source_variant_id!r} is not in this score result"
            )

        reference_ids, reference_warnings = select_references(
            score,
            request.source_variant_id,
            request.reference_strategy.value,
            manual_ids=request.comparison_variant_ids,
            baseline_id=request.baseline_variant_id or score.baseline_variant_id,
        )
        warnings.extend(reference_warnings)
        references = [v for v in (_variant(score, rid) for rid in reference_ids) if v]

        gaps = []
        weak: list = []
        strong: list = []
        evidence: list[Evidence] = []
        recommendations: list[OptimizationRecommendation] = []
        rejected: list[dict[str, str]] = []
        features_considered = 0

        for objective in score.objectives:
            best_reference = references[0] if references else None
            gap = analyse_gap(objective, source, best_reference)
            gaps.append(gap)

            intervals, interval_warnings = detect_intervals(objective, source, best_reference)
            warnings.extend(interval_warnings)
            weak.extend(i for i in intervals if i.kind is IntervalKind.WEAK)
            strong.extend(i for i in intervals if i.kind is IntervalKind.STRONG)

            source_series = content.get(request.source_variant_id)
            reference_series = [content[r] for r in reference_ids if r in content]

            for interval in (i for i in intervals if i.kind is IntervalKind.WEAK):
                prefix = interval.interval_id.replace("weak-", "W")
                items: list[Evidence] = [interval_evidence(interval, prefix)]
                item = gap_evidence(gap, prefix)
                if item is not None:
                    items.append(item)

                if source_series is not None and reference_series:
                    feature_items, considered = build_feature_evidence(
                        interval, source_series, reference_series, prefix=prefix
                    )
                    items.extend(feature_items)
                    features_considered = max(features_considered, considered)
                    if not feature_items:
                        warnings.append(
                            _no_feature_evidence_reason(
                                interval, source_series, reference_series
                            )
                        )

                # Where the gap sits spatially, when the target aggregates
                # many vertices. An ROI objective is already one region.
                if best_reference is not None:
                    source_eval = next(
                        (
                            e
                            for e in source.objective_scores
                            if e.objective_id == objective.objective_id
                        ),
                        None,
                    )
                    reference_eval = next(
                        (
                            e
                            for e in best_reference.objective_scores
                            if e.objective_id == objective.objective_id
                        ),
                        None,
                    )
                    if source_eval is not None and reference_eval is not None:
                        items.extend(
                            regional_evidence(source_eval, reference_eval, prefix)
                        )

                if request.source_variant_id in events and reference_ids:
                    items.extend(
                        event_presence_evidence(
                            interval,
                            events.get(request.source_variant_id, []),
                            {r: events[r] for r in reference_ids if r in events},
                            prefix,
                        )
                    )

                evidence.extend(items)
                recommendations.extend(
                    generate(
                        interval,
                        gap,
                        items,
                        strategy=request.strategy,
                        source_variant_id=request.source_variant_id,
                        reference_variant_ids=tuple(reference_ids),
                        prefix=prefix,
                    )
                )

        by_id = {item.evidence_id: item for item in evidence}
        accepted: list[OptimizationRecommendation] = []
        for recommendation in recommendations:
            problems = validate(recommendation, by_id, request.constraints)
            if problems:
                rejected.append(
                    {
                        "recommendation_id": recommendation.recommendation_id,
                        "reason": "; ".join(problems),
                    }
                )
                continue
            accepted.append(recommendation)

        accepted, duplicates = deduplicate(accepted)
        rejected.extend(duplicates)
        ranked = rank(accepted, max_recommendations=request.max_recommendations)

        hypotheses = [_hypothesis(item) for item in ranked]
        linked = [
            item.model_copy(update={"hypothesis_id": hypothesis.hypothesis_id})
            for item, hypothesis in zip(ranked, hypotheses, strict=True)
        ]
        traces = [
            OptimizationTrace(
                recommendation_id=item.recommendation_id,
                objective_id=item.objective_id,
                source_variant_id=item.source_variant_id,
                interval_id=_interval_for(item, weak),
                reference_variant_ids=item.reference_variant_ids,
                evidence_ids=item.evidence_ids,
                feature_differences=tuple(
                    by_id[e].feature or "" for e in item.evidence_ids if e in by_id
                ),
                hypothesis_id=item.hypothesis_id,
            )
            for item in linked
        ]

        if not linked and not warnings:
            warnings.append(
                "no recommendation survived validation; the source may already match its "
                "references over the intervals examined"
            )
        if features_considered:
            warnings.append(
                f"{features_considered} content feature(s) were examined per interval. "
                f"Scanning many features will surface differences by chance; treat the "
                f"reported set as candidates to test, not as findings."
            )

        return OptimizationResult(
            request=request,
            gaps=tuple(gaps),
            weak_intervals=tuple(weak),
            strong_intervals=tuple(strong),
            reference_variant_ids=tuple(reference_ids),
            evidence=tuple(evidence),
            recommendations=tuple(linked),
            hypotheses=tuple(hypotheses),
            traces=tuple(traces),
            rejected=tuple(rejected),
            metadata=OptimizationMetadata(
                optimization_version=OPTIMIZATION_VERSION,
                scoring_version=score.metadata.scoring_version,
                objective_set_hash=score.metadata.objective_set_hash,
                features_considered=features_considered,
                generator="deterministic",
            ),
            warnings=tuple(warnings),
        )


def _no_feature_evidence_reason(
    interval: object, source: ContentSeries, references: list[ContentSeries]
) -> str:
    """Say why an interval produced no content evidence.

    Producing nothing is often correct -- the reference may not cover the
    interval at all, or the differences may be within noise -- but a bare
    "0 recommendations" leaves a reader unable to tell a genuine absence of
    difference from a coverage hole.
    """
    start = float(getattr(interval, "start_seconds", 0.0))
    end = float(getattr(interval, "end_seconds", 0.0))
    samples = int(getattr(interval, "sample_count", 0))
    label = f"{start:.2f}-{end:.2f}s"

    if samples < 2:
        return (
            f"Interval {label} holds {samples} prediction sample(s); below two it cannot "
            f"support content evidence, so no recommendation was generated for it."
        )

    uncovered = [
        reference.variant_id
        for reference in references
        if reference.interval_means(start, end)[1] < 2
    ]
    if len(uncovered) == len(references):
        spans = ", ".join(
            f"{r.variant_id} covers {r.times.min():.2f}-{r.times.max():.2f}s"
            for r in references
        )
        return (
            f"No reference variant has content analysis covering {label}, so the source "
            f"could not be compared there ({spans}). This is a coverage gap, not a finding "
            f"that the variants are alike."
        )
    if uncovered:
        return (
            f"Reference variant(s) {uncovered} do not cover {label}; the comparison there "
            f"rests on the remaining reference(s)."
        )
    return (
        f"No content feature differed by enough over {label} to clear the reporting "
        f"threshold. The variants measured alike there on every feature examined."
    )


def _hypothesis(recommendation: OptimizationRecommendation) -> OptimizationHypothesis:
    """Restate a recommendation as something Phase 8 can pass or fail."""
    features = tuple(
        str(intervention.parameters.get("feature") or intervention.parameters.get("event_type"))
        for intervention in recommendation.interventions
        if intervention.parameters
    )
    return OptimizationHypothesis(
        hypothesis_id=f"HYP-{recommendation.recommendation_id}",
        statement=(
            f"If {recommendation.source_variant_id} is modified as described "
            f"({recommendation.title}) over "
            f"{recommendation.target_interval_start:.2f}-"
            f"{recommendation.target_interval_end:.2f}s while other content is preserved, "
            f"the resulting candidate can test whether that change is associated with a "
            f"different value on objective {recommendation.objective_id}. The direction to "
            f"test is {recommendation.expected_direction.value}. This is a hypothesis; it is "
            f"established only by re-simulating the candidate and rescoring it."
        ),
        objective_id=recommendation.objective_id,
        source_variant_id=recommendation.source_variant_id,
        recommendation_id=recommendation.recommendation_id,
        changed_features=features,
        interval_start_seconds=recommendation.target_interval_start,
        interval_end_seconds=recommendation.target_interval_end,
        expected_direction=recommendation.expected_direction,
        evidence_ids=recommendation.evidence_ids,
    )


def _interval_for(recommendation: OptimizationRecommendation, weak: list) -> str | None:
    for interval in weak:
        if (
            interval.objective_id == recommendation.objective_id
            and abs(interval.start_seconds - recommendation.target_interval_start) < 1e-6
        ):
            return interval.interval_id
    return None


def _variant(score: ExperimentScoreResult, variant_id: str) -> VariantScore | None:
    return next((v for v in score.variant_scores if v.variant_id == variant_id), None)


def content_series_from_arrays(
    variant_id: str, times: np.ndarray, matrix: np.ndarray, names: tuple[str, ...]
) -> ContentSeries:
    return ContentSeries(
        variant_id=variant_id,
        times=np.asarray(times, dtype=np.float64),
        matrix=np.asarray(matrix, dtype=np.float64),
        feature_names=names,
    )
