"""Orchestration for the Phase 3 T-by-V to T-by-ROI analytics pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blackmirror.analytics.atlas import AtlasMapping
from blackmirror.analytics.events import NeuralEventDetector
from blackmirror.analytics.metrics import (
    aggregate_hemispheres,
    aggregate_regions,
    change_magnitude,
    global_metrics,
    roi_correlation_matrix,
    spatial_concentration,
)
from blackmirror.analytics.schemas import (
    AnalyticsMetadata,
    ArrayArtifact,
    AtlasValidationReport,
    NeuralAnalyticsResult,
    ROIResponse,
)

ANALYTICS_VERSION = "1.1"


class NeuralAnalyticsEngine:
    """Compute reproducible mathematical summaries of cortical predictions."""

    def __init__(self, event_detector: NeuralEventDetector | None = None) -> None:
        self.event_detector = event_detector or NeuralEventDetector()

    def analyze(
        self,
        *,
        run_id: str,
        responses: NDArray[np.floating],
        times: NDArray[np.floating],
        atlas: AtlasMapping,
        validation: AtlasValidationReport,
        hemisphere_ranges: dict[str, tuple[int, int]],
    ) -> tuple[NeuralAnalyticsResult, dict[str, NDArray[np.generic]]]:
        if not validation.valid:
            raise ValueError(f"atlas validation failed: {'; '.join(validation.errors)}")
        values = np.asarray(responses)
        clock = np.asarray(times, dtype=np.float64)
        if values.ndim != 2 or clock.shape != (values.shape[0],):
            raise ValueError("responses must be T-by-V and times must have length T")

        roi = aggregate_regions(values, atlas.vertex_to_region, atlas.metadata.region_count)
        excluded = atlas.medial_wall_mask
        global_arrays = global_metrics(values, excluded)
        changes = change_magnitude(values, excluded)
        hemispheres = aggregate_hemispheres(values, hemisphere_ranges, excluded)
        concentration = spatial_concentration(values, excluded)
        events = self.event_detector.detect(
            global_arrays["mean"], global_arrays["l2_magnitude"], changes, roi, clock
        )
        roi_responses = tuple(
            self._roi_summary(region, roi[:, region.region_id], clock)
            for region in atlas.metadata.regions
        )

        arrays: dict[str, NDArray[np.generic]] = {
            "times": clock,
            "roi_timeseries": roi,
            "change_magnitude": changes,
            "spatial_concentration": concentration,
            "roi_response_correlation": roi_correlation_matrix(roi),
            **{f"global_{key}": value for key, value in global_arrays.items()},
            **{f"hemisphere_{key}": value for key, value in hemispheres.items()},
        }
        descriptors = {
            "timeseries": ArrayArtifact(
                path="analytics/timeseries.npz",
                keys={key: _array_description(key) for key in arrays},
            ),
            "atlas_mapping": ArrayArtifact(
                path="analytics/atlas_mapping.npz",
                keys={
                    "vertex_to_region": "Dense int32 vertex-to-region mapping; -1 is unmapped",
                    "medial_wall_mask": "Boolean mask; true vertices are excluded from ROIs",
                },
            ),
        }
        result = NeuralAnalyticsResult(
            run_id=run_id,
            atlas=atlas.metadata,
            validation=validation,
            roi_responses=roi_responses,
            events=events,
            arrays=descriptors,
            metadata=AnalyticsMetadata(
                analytics_version=ANALYTICS_VERSION,
                aggregation="finite-value arithmetic mean over non-medial-wall ROI vertices",
                parameters={
                    "event_z_threshold": self.event_detector.z_threshold,
                    "event_min_distance_samples": self.event_detector.min_distance,
                    "event_max_events": self.event_detector.max_events,
                },
            ),
        )
        return result, arrays

    @staticmethod
    def _roi_summary(
        region: Any, values: NDArray[np.float64], times: NDArray[np.float64]
    ) -> ROIResponse:
        if not np.isfinite(values).any():
            raise ValueError(f"region {region.region_id} has no finite responses")
        peak = int(np.nanargmax(np.abs(values)))
        return ROIResponse(
            region=region,
            mean_response=float(np.nanmean(values)),
            peak_response=float(values[peak]),
            peak_timestep=peak,
            peak_time_seconds=float(times[peak]),
            std_response=float(np.nanstd(values)),
            min_response=float(np.nanmin(values)),
            max_response=float(np.nanmax(values)),
        )


def _array_description(key: str) -> str:
    descriptions = {
        "times": "Stimulus-aligned seconds for each response row",
        "roi_timeseries": "Mean predicted response [time, region]",
        "change_magnitude": "L2 norm between consecutive cortical patterns; first value is zero",
        "spatial_concentration": "Normalized Herfindahl concentration of absolute response",
        "roi_response_correlation": (
            "Pearson correlation among ROI response time-series; not connectivity"
        ),
    }
    return descriptions.get(key, key.replace("_", " "))
