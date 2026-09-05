"""Load persisted Phase 1/3 artifacts and create a Phase 5 comparison."""

from __future__ import annotations

import datetime as dt
import secrets
from pathlib import Path

import numpy as np

from blackmirror.analytics.storage import AnalyticsStore
from blackmirror.comparison.content import ContentVariant, compare_content_features
from blackmirror.comparison.engine import NeuralComparisonEngine, VariantData
from blackmirror.comparison.network import load_yeo7_mapping
from blackmirror.comparison.schemas import (
    ContentComparisonIssue,
    ContentFeatureDifferenceSummary,
    NeuralComparisonResult,
)
from blackmirror.comparison.storage import ComparisonStore, resolve_artifact_path, validate_id
from blackmirror.content.storage import ContentStore
from blackmirror.errors import ArtifactReadError
from blackmirror.storage.artifact_store import ArtifactStore


def generate_comparison_id() -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"cmp-{stamp}-{secrets.token_hex(4)}"


def compare_runs(
    artifact_root: Path,
    reference_run_id: str,
    candidate_run_ids: tuple[str, ...],
    *,
    comparison_id: str | None = None,
    tolerance_seconds: float = 1e-6,
    alignment_method: str = "exact_observed_intersection",
    max_interpolation_gap_seconds: float = 10.0,
) -> NeuralComparisonResult:
    """Compare candidates to one declared reference and persist immutable output."""
    if len(set((reference_run_id, *candidate_run_ids))) != 1 + len(candidate_run_ids):
        raise ValueError("reference and candidate run ids must be unique")
    if comparison_id is not None:
        validate_id(comparison_id, "comparison")
    raw_store = ArtifactStore(artifact_root)
    analytics_store = AnalyticsStore(artifact_root)
    network = load_yeo7_mapping(
        artifact_root / "atlases" / "yeo2011_fsaverage5",
        artifact_root / "mesh" / "fsaverage5",
    )

    def load(run_id: str) -> VariantData:
        prediction = raw_store.read_manifest(run_id)
        analytics = analytics_store.read(run_id)
        if analytics.run_id != run_id:
            raise ValueError(f"analytics run id does not match requested run {run_id}")
        responses = raw_store.load_predictions(run_id)
        temporal_path = prediction.temporal.arrays_artifact_path
        if temporal_path is None:
            raise ValueError(f"run {run_id} has no temporal arrays")
        run_root = raw_store.run_dir(run_id)
        with np.load(resolve_artifact_path(run_root, temporal_path), allow_pickle=False) as archive:
            times = np.asarray(archive["stimulus_time_seconds"], dtype=np.float64)
        analytics_path = resolve_artifact_path(run_root, analytics.arrays["timeseries"].path)
        with np.load(analytics_path, allow_pickle=False) as archive:
            roi = np.asarray(archive["roi_timeseries"], dtype=np.float64)
        mapping_path = resolve_artifact_path(run_root, analytics.arrays["atlas_mapping"].path)
        with np.load(mapping_path, allow_pickle=False) as archive:
            vertex_to_region = np.asarray(archive["vertex_to_region"], dtype=np.int32)
            medial_wall = np.asarray(archive["medial_wall_mask"], dtype=bool)
        return VariantData(
            prediction,
            analytics,
            responses,
            times,
            roi,
            vertex_to_region,
            medial_wall,
            network.aggregate(responses),
            network.network_names,
            network.mapping_sha256,
        )

    reference = load(reference_run_id)
    candidates = tuple(load(run_id) for run_id in candidate_run_ids)
    result, arrays = NeuralComparisonEngine(
        tolerance_seconds=tolerance_seconds,
        alignment_method=alignment_method,
        max_interpolation_gap_seconds=max_interpolation_gap_seconds,
    ).compare(comparison_id or generate_comparison_id(), reference, candidates)
    content_store = ContentStore(artifact_root)
    if content_store.exists(reference_run_id):
        try:
            reference_content = _load_content(content_store, reference_run_id)
        except (ArtifactReadError, ValueError, OSError) as exc:
            result = _with_reference_content_failure(result, exc)
            reference_content = None
        if reference_content is None:
            ComparisonStore(artifact_root).write(result, arrays)
            return result
        updated_pairs = []
        for pair in result.pairs:
            if not content_store.exists(pair.candidate_run_id):
                updated_pairs.append(pair)
                continue
            try:
                difference = compare_content_features(
                    reference_content,
                    _load_content(content_store, pair.candidate_run_id),
                    tolerance_seconds=tolerance_seconds,
                )
            except (ArtifactReadError, ValueError, OSError) as exc:
                updated_pairs.append(
                    pair.model_copy(
                        update={
                            "content_status": "incomparable_candidate_artifact",
                            "content_issue": ContentComparisonIssue(
                                code="candidate_content_incomparable",
                                message=str(exc),
                                reference_run_id=reference_run_id,
                                candidate_run_id=pair.candidate_run_id,
                            ),
                        }
                    )
                )
                continue
            arrays[pair.candidate_run_id].update(
                {
                    "content_times": difference.times,
                    "content_signed_delta": difference.signed_delta,
                    "content_jointly_available": difference.jointly_available,
                }
            )
            summaries = []
            for index, name in enumerate(difference.feature_names):
                values = difference.signed_delta[:, index]
                finite = np.isfinite(values)
                summaries.append(
                    ContentFeatureDifferenceSummary(
                        name=name,
                        jointly_available_fraction=float(difference.coverage_fraction[index]),
                        mean_signed_delta=float(np.mean(values[finite])) if finite.any() else None,
                        mean_absolute_delta=(
                            float(np.mean(np.abs(values[finite]))) if finite.any() else None
                        ),
                    )
                )
            updated_pairs.append(
                pair.model_copy(
                    update={
                        "content_status": "comparable_joint_availability_only",
                        "content_analysis_version": reference_content.analysis_version,
                        "content_models": reference_content.models,
                        "content_configuration": reference_content.configuration,
                        "content_features": tuple(summaries),
                        "array_keys": {
                            **pair.array_keys,
                            "content_times": "Aligned observed Phase 4 feature timestamps",
                            "content_signed_delta": "Candidate minus reference content features",
                            "content_jointly_available": "Per-feature joint availability mask",
                        },
                    }
                )
            )
        result = result.model_copy(update={"pairs": tuple(updated_pairs)})
    ComparisonStore(artifact_root).write(result, arrays)
    return result


def _with_reference_content_failure(
    result: NeuralComparisonResult, error: Exception
) -> NeuralComparisonResult:
    """Downgrade content for every pair without weakening neural validation."""
    return result.model_copy(
        update={
            "pairs": tuple(
                pair.model_copy(
                    update={
                        "content_status": "unavailable_reference_artifact",
                        "content_issue": ContentComparisonIssue(
                            code="reference_content_unreadable",
                            message=str(error),
                            reference_run_id=result.metadata.reference_run_id,
                            candidate_run_id=pair.candidate_run_id,
                        ),
                    }
                )
                for pair in result.pairs
            )
        }
    )


def _load_content(store: ContentStore, run_id: str) -> ContentVariant:
    result = store.read(run_id)
    arrays = store.read_arrays(run_id)
    if result.arrays is None or "features" not in arrays or "feature_times" not in arrays:
        raise ValueError(f"content artifact for {run_id} lacks versioned dense features")
    return ContentVariant(
        schema_version=result.schema_version,
        analysis_version=result.metadata.analysis_version,
        models=result.metadata.models,
        configuration=result.metadata.configuration,
        feature_names=result.arrays.feature_names,
        times=arrays["feature_times"],
        matrix=arrays["features"],
    )
