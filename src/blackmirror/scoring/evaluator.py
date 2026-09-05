"""Resolving a target to a response series, and evaluating one objective.

THE PIPELINE THIS IMPLEMENTS
    variant + objective
        -> target resolution   (which vertices / which precomputed series)
        -> window resolution   (which samples, in this variant's own timeline)
        -> metric              (one raw number, in model units)
        -> ObjectiveEvaluation (raw value + statistics + provenance, NO score)

    Scoring deliberately stops before producing a score. Normalisation needs to
    see every variant in the experiment, so `score` and `normalized_value` stay
    None here and are filled in by the engine. That ordering is what keeps a
    single-variant evaluation honest: there is no scale to place it on yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from blackmirror.scoring.metrics import MetricInput, MetricRegistry, default_registry
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveEvaluation,
    TargetResolution,
    TargetType,
    TemporalContribution,
)
from blackmirror.scoring.windows import (
    ContentEventInterval,
    WindowResolutionError,
    resolve_window,
)


@dataclass(frozen=True)
class ScoringVariant:
    """One variant, with only what scoring needs.

    A "variant" is a completed run. There is no separate Experiment entity in
    this repository, so a variant is identified by its run id and a scoring
    experiment is simply an explicit set of them plus a declared baseline.
    """

    variant_id: str
    #: Prediction samples, [time, vertex]. Raw model output, never modified.
    responses: NDArray[np.floating]
    #: Stimulus-time axis, identical to the analytics time axis.
    times: NDArray[np.floating]
    #: [time, region] Destrieux aggregation from Phase 3.
    roi_timeseries: NDArray[np.floating]
    region_names: tuple[str, ...]
    medial_wall_mask: NDArray[np.bool_]
    duration_seconds: float
    hemisphere_ranges: dict[str, tuple[int, int]]
    atlas_name: str | None = None
    atlas_version: str | None = None
    #: [time, network] Yeo aggregation, when a verified mapping was supplied.
    network_timeseries: NDArray[np.floating] | None = None
    network_names: tuple[str, ...] = ()
    network_mapping_sha256: str | None = None
    analytics_version: str | None = None
    model_fingerprint: str | None = None
    content_events: list[ContentEventInterval] | None = None
    #: Reference patterns by name, each supplied with provenance by the caller.
    reference_patterns: dict[str, NDArray[np.floating]] = field(default_factory=dict)


class TargetResolutionError(ValueError):
    """The objective's target does not exist in this variant."""


def resolve_target(
    variant: ScoringVariant, objective: NeuralObjective
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, TargetResolution]:
    """Return (series over full timeline, vertex block or None, description)."""
    target = objective.target
    responses = np.asarray(variant.responses, dtype=np.float64)
    wall = np.asarray(variant.medial_wall_mask, dtype=bool)

    if target.type is TargetType.ROI:
        assert target.region_id is not None
        roi = np.asarray(variant.roi_timeseries, dtype=np.float64)
        if not 0 <= target.region_id < roi.shape[1]:
            raise TargetResolutionError(
                f"region_id {target.region_id} is outside this variant's atlas, which has "
                f"{roi.shape[1]} regions"
            )
        name = (
            variant.region_names[target.region_id]
            if target.region_id < len(variant.region_names)
            else f"region {target.region_id}"
        )
        return (
            roi[:, target.region_id],
            None,
            TargetResolution(
                type=target.type,
                label=name,
                vertex_count=0,
                region_id=target.region_id,
                atlas_name=variant.atlas_name,
                atlas_version=variant.atlas_version,
            ),
        )

    if target.type is TargetType.NETWORK:
        assert target.network_id is not None
        if variant.network_timeseries is None:
            raise TargetResolutionError(
                "this variant has no functional-network series; a verified Yeo mapping must "
                "be supplied for network objectives"
            )
        networks = np.asarray(variant.network_timeseries, dtype=np.float64)
        if not 0 <= target.network_id < networks.shape[1]:
            raise TargetResolutionError(
                f"network_id {target.network_id} is outside the {networks.shape[1]} available"
            )
        label = (
            variant.network_names[target.network_id]
            if target.network_id < len(variant.network_names)
            else f"network {target.network_id}"
        )
        return (
            networks[:, target.network_id],
            None,
            TargetResolution(
                type=target.type,
                label=label,
                vertex_count=0,
                network_id=target.network_id,
                mapping_sha256=variant.network_mapping_sha256,
            ),
        )

    if target.type is TargetType.HEMISPHERE:
        assert target.hemisphere is not None
        if target.hemisphere not in variant.hemisphere_ranges:
            raise TargetResolutionError(
                f"hemisphere {target.hemisphere!r} is not in this variant's cortical metadata"
            )
        start, end = variant.hemisphere_ranges[target.hemisphere]
        selected = np.zeros(responses.shape[1], dtype=bool)
        selected[start:end] = True
        selected &= ~wall
        label = f"{target.hemisphere} hemisphere"
        series, block = _mean_over(responses, selected)
        return series, block, _describe(target.type, label, selected)

    if target.type is TargetType.CUSTOM_VERTEX_SET:
        chosen = np.asarray(target.vertex_indices, dtype=np.int64)
        if chosen.size and (chosen.min() < 0 or chosen.max() >= responses.shape[1]):
            raise TargetResolutionError(
                f"custom vertex set references vertices outside [0, {responses.shape[1]})"
            )
        selected = np.zeros(responses.shape[1], dtype=bool)
        selected[chosen] = True
        excluded = int((selected & wall).sum())
        selected &= ~wall
        if not selected.any():
            raise TargetResolutionError(
                "every vertex in the custom set lies on the medial wall, where the model "
                "makes no meaningful prediction"
            )
        resolution = _describe(target.type, f"{int(selected.sum())} vertices", selected)
        series, block = _mean_over(responses, selected)
        if excluded:
            # Recorded on the evaluation's warnings by the caller.
            resolution = resolution.model_copy(
                update={"label": f"{int(selected.sum())} vertices ({excluded} on the wall dropped)"}
            )
        return series, block, resolution

    # WHOLE_CORTEX
    cortex = np.asarray(~wall, dtype=bool)
    series, block = _mean_over(responses, cortex)
    return series, block, _describe(TargetType.WHOLE_CORTEX, "whole cortex", cortex)


def _mean_over(
    responses: NDArray[np.float64], selected: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean across selected vertices per sample, plus the vertex block itself."""
    block = responses[:, selected]
    finite = np.isfinite(block)
    counts = finite.sum(axis=1)
    totals = np.where(finite, block, 0.0).sum(axis=1)
    series = np.divide(
        totals, counts, out=np.full(block.shape[0], np.nan, dtype=np.float64), where=counts > 0
    )
    return series, block


def _describe(
    target_type: TargetType, label: str, selected: NDArray[np.bool_]
) -> TargetResolution:
    return TargetResolution(
        type=target_type, label=label, vertex_count=int(selected.sum())
    )


class ObjectiveEvaluator:
    """Evaluates one objective against one variant, producing a raw value."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def evaluate(
        self,
        variant: ScoringVariant,
        objective: NeuralObjective,
        *,
        baseline: ScoringVariant | None = None,
    ) -> ObjectiveEvaluation:
        metric = self.registry.get(objective.metric)
        warnings: list[str] = []

        if objective.target.type.value not in metric.supported_targets:
            return self._invalid(
                variant, objective,
                f"metric {objective.metric} does not support target type "
                f"{objective.target.type.value}",
            )
        try:
            series, block, target = resolve_target(variant, objective)
        except TargetResolutionError as exc:
            return self._invalid(variant, objective, str(exc))
        try:
            window = resolve_window(
                objective.temporal_scope,
                times=variant.times,
                duration_seconds=variant.duration_seconds,
                events=variant.content_events,
            )
        except WindowResolutionError as exc:
            return self._invalid(variant, objective, str(exc), target=target)

        indices = np.asarray(window.sample_indices, dtype=np.int64)
        windowed = series[indices]
        windowed_times = np.asarray(variant.times, dtype=np.float64)[indices]
        windowed_block = block[indices] if block is not None else None

        baseline_series: NDArray[np.float64] | None = None
        baseline_times: NDArray[np.float64] | None = None
        if metric.requires_baseline:
            if baseline is None:
                return self._invalid(
                    variant, objective,
                    f"metric {objective.metric} requires a declared baseline variant",
                    target=target, window=window,
                )
            try:
                base_series, _, _ = resolve_target(baseline, objective)
                base_window = resolve_window(
                    objective.temporal_scope,
                    times=baseline.times,
                    duration_seconds=baseline.duration_seconds,
                    events=baseline.content_events,
                )
            except (TargetResolutionError, WindowResolutionError) as exc:
                return self._invalid(
                    variant, objective, f"baseline could not be resolved: {exc}",
                    target=target, window=window,
                )
            baseline_series = base_series[np.asarray(base_window.sample_indices, dtype=np.int64)]
            baseline_times = np.asarray(baseline.times, dtype=np.float64)[
                np.asarray(base_window.sample_indices, dtype=np.int64)
            ]

        reference: NDArray[np.float64] | None = None
        if metric.requires_reference:
            name = str(objective.parameters.get("reference_pattern", ""))
            if not name or name not in variant.reference_patterns:
                return self._invalid(
                    variant, objective,
                    f"metric {objective.metric} requires parameters['reference_pattern'] to "
                    f"name a supplied reference; available: "
                    f"{sorted(variant.reference_patterns) or 'none'}",
                    target=target, window=window,
                )
            reference = np.asarray(variant.reference_patterns[name], dtype=np.float64)

        data = MetricInput(
            series=windowed,
            times=windowed_times,
            vertex_block=windowed_block,
            reference_pattern=reference,
            baseline_series=baseline_series,
            baseline_times=baseline_times,
            parameters=dict(objective.parameters),
        )
        problems = metric.validate(data)
        if problems:
            return self._invalid(
                variant, objective, "; ".join(problems), target=target, window=window
            )
        output = metric.compute(data)
        warnings.extend(output.warnings)

        temporal = None
        shares = metric.decompose(data)
        if shares is not None and output.value is not None:
            total = float(np.abs(shares).sum())
            # The contract is that shares sum to the raw value. Verified here
            # rather than trusted, because a decomposition that does not add up
            # would silently misattribute which seconds produced the score.
            if not np.isclose(float(shares.sum()), float(output.value), rtol=1e-6, atol=1e-9):
                warnings.append(
                    f"metric {objective.metric} returned a temporal decomposition summing to "
                    f"{float(shares.sum()):.6g} against a raw value of {output.value:.6g}; the "
                    f"decomposition was discarded rather than reported as misattribution"
                )
            else:
                peak = int(np.argmax(np.abs(shares)))
                temporal = TemporalContribution(
                    times=tuple(float(value) for value in windowed_times),
                    contributions=tuple(float(value) for value in shares),
                    peak_index=peak,
                    peak_time_seconds=float(windowed_times[peak]),
                    peak_fraction=(
                        float(abs(shares[peak]) / total) if total > 0 else None
                    ),
                )
        if window.sample_count < 3:
            warnings.append(
                f"the resolved window holds only {window.sample_count} prediction sample(s); "
                f"at this run's sampling period that is a very small basis for any metric"
            )
        return ObjectiveEvaluation(
            objective_id=objective.objective_id,
            variant_id=variant.variant_id,
            raw_value=output.value,
            window=window,
            target=target,
            statistics={key: float(value) for key, value in output.statistics.items()},
            temporal_contribution=temporal,
            valid=output.value is not None,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _invalid(
        variant: ScoringVariant,
        objective: NeuralObjective,
        reason: str,
        *,
        target: TargetResolution | None = None,
        window: object | None = None,
    ) -> ObjectiveEvaluation:
        from blackmirror.scoring.schemas import ResolvedWindow

        return ObjectiveEvaluation(
            objective_id=objective.objective_id,
            variant_id=variant.variant_id,
            raw_value=None,
            window=window  # type: ignore[arg-type]
            or ResolvedWindow(
                start_seconds=0.0, end_seconds=0.0, sample_indices=(), sample_count=0,
                source="unresolved",
            ),
            target=target
            or TargetResolution(type=objective.target.type, label="unresolved", vertex_count=0),
            valid=False,
            warnings=(reason,),
        )
