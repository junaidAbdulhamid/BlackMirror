"""Strict provenance, cortical, and analytics comparability gates."""

from __future__ import annotations

from blackmirror.analytics.schemas import NeuralAnalyticsResult
from blackmirror.comparison.schemas import (
    ComparabilityIssue,
    ComparabilityReport,
    ComparabilitySeverity,
)
from blackmirror.schemas.prediction import PredictionResult


def validate_comparability(
    reference: PredictionResult,
    candidate: PredictionResult,
    reference_analytics: NeuralAnalyticsResult,
    candidate_analytics: NeuralAnalyticsResult,
) -> ComparabilityReport:
    checks: dict[str, bool] = {}
    issues: list[ComparabilityIssue] = []

    def check(field: str, first: object, second: object) -> None:
        passed = first == second
        checks[field] = passed
        if not passed:
            issues.append(
                ComparabilityIssue(
                    field=field,
                    severity=ComparabilitySeverity.ERROR,
                    message=f"reference and candidate differ in {field}",
                )
            )

    def require(field: str, condition: bool, message: str) -> None:
        checks[field] = condition
        if not condition:
            issues.append(
                ComparabilityIssue(
                    field=field,
                    severity=ComparabilitySeverity.ERROR,
                    message=message,
                )
            )

    require("distinct_run_id", reference.run_id != candidate.run_id, "run ids must be distinct")
    require(
        "distinct_stimulus_sha256",
        reference.stimulus.sha256 != candidate.stimulus.sha256,
        "stimulus content must differ for a controlled variant comparison",
    )
    check("synthetic_status", reference.model.is_synthetic, candidate.model.is_synthetic)
    check(
        "model_fingerprint",
        reference.provenance.model_fingerprint,
        candidate.provenance.model_fingerprint,
    )
    check(
        "preprocessing_config",
        reference.provenance.preprocessing_config,
        candidate.provenance.preprocessing_config,
    )
    check(
        "compatibility_patches",
        reference.provenance.applied_compat_patches,
        candidate.provenance.applied_compat_patches,
    )
    check("surface_space", reference.cortical.surface_space, candidate.cortical.surface_space)
    check("vertex_count", reference.cortical.vertex_count, candidate.cortical.vertex_count)
    check(
        "hemisphere_order", reference.cortical.hemisphere_order, candidate.cortical.hemisphere_order
    )
    check(
        "hemisphere_ranges",
        reference.cortical.hemisphere_index_ranges,
        candidate.cortical.hemisphere_index_ranges,
    )
    check("prediction_semantics", reference.prediction.semantics, candidate.prediction.semantics)
    check("prediction_units", reference.prediction.units, candidate.prediction.units)
    check(
        "analytics_schema", reference_analytics.schema_version, candidate_analytics.schema_version
    )
    check(
        "analytics_version",
        reference_analytics.metadata.analytics_version,
        candidate_analytics.metadata.analytics_version,
    )
    check(
        "analytics_aggregation",
        reference_analytics.metadata.aggregation,
        candidate_analytics.metadata.aggregation,
    )
    check("atlas_name", reference_analytics.atlas.name, candidate_analytics.atlas.name)
    check("atlas_version", reference_analytics.atlas.version, candidate_analytics.atlas.version)
    check(
        "atlas_surface",
        reference_analytics.atlas.surface_space,
        candidate_analytics.atlas.surface_space,
    )
    reference_regions = tuple(
        (r.region_id, r.atlas_label, r.name, r.hemisphere)
        for r in reference_analytics.atlas.regions
    )
    candidate_regions = tuple(
        (r.region_id, r.atlas_label, r.name, r.hemisphere)
        for r in candidate_analytics.atlas.regions
    )
    check("ordered_regions", reference_regions, candidate_regions)
    return ComparabilityReport(
        comparable=not issues,
        reference_run_id=reference.run_id,
        candidate_run_id=candidate.run_id,
        checks=checks,
        issues=tuple(issues),
    )
