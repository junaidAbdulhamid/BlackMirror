"""Vectorized pairwise and reference-based A/B/N neural comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blackmirror.analytics.schemas import NeuralAnalyticsResult
from blackmirror.comparison.alignment import (
    AlignedTimeline,
    InterpolatedTimeline,
    align_linear_candidate_to_reference,
    align_observed_times,
    interpolate_rows,
)
from blackmirror.comparison.schemas import (
    ComparisonMetadata,
    DivergenceEvent,
    DivergenceWindow,
    NetworkDifferenceSummary,
    NeuralComparisonResult,
    PairwiseComparison,
    RegionDifferenceSummary,
    TimelineAlignment,
)
from blackmirror.comparison.validation import validate_comparability
from blackmirror.schemas.prediction import PredictionResult

COMPARISON_VERSION = "1.1"


@dataclass(frozen=True)
class VariantData:
    prediction: PredictionResult
    analytics: NeuralAnalyticsResult
    responses: NDArray[np.floating]
    times: NDArray[np.floating]
    roi_timeseries: NDArray[np.floating]
    vertex_to_region: NDArray[np.integer]
    medial_wall_mask: NDArray[np.bool_]
    network_timeseries: NDArray[np.floating]
    network_names: tuple[str, ...]
    network_mapping_sha256: str


class NeuralComparisonEngine:
    def __init__(
        self,
        *,
        tolerance_seconds: float = 1e-6,
        max_events: int = 20,
        top_regions_per_event: int = 5,
        window_radius_samples: int = 1,
        alignment_method: str = "exact_observed_intersection",
        max_interpolation_gap_seconds: float = 10.0,
    ) -> None:
        invalid_tolerance = not np.isfinite(tolerance_seconds) or tolerance_seconds < 0
        if invalid_tolerance or max_events < 1 or top_regions_per_event < 1:
            raise ValueError("invalid comparison parameters")
        if window_radius_samples < 0:
            raise ValueError("window radius must be nonnegative")
        self.tolerance_seconds = tolerance_seconds
        self.max_events = max_events
        self.top_regions_per_event = top_regions_per_event
        self.window_radius_samples = window_radius_samples
        if alignment_method not in {
            "exact_observed_intersection",
            "linear_candidate_to_reference",
        }:
            raise ValueError("unsupported timeline alignment method")
        self.alignment_method = alignment_method
        if not np.isfinite(max_interpolation_gap_seconds) or max_interpolation_gap_seconds <= 0:
            raise ValueError("maximum interpolation gap must be finite and positive")
        self.max_interpolation_gap_seconds = max_interpolation_gap_seconds

    def compare(
        self,
        comparison_id: str,
        reference: VariantData,
        candidates: tuple[VariantData, ...],
    ) -> tuple[NeuralComparisonResult, dict[str, dict[str, NDArray[np.generic]]]]:
        if not candidates:
            raise ValueError("at least one candidate run is required")
        run_ids = (reference.prediction.run_id, *(item.prediction.run_id for item in candidates))
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("reference and candidate run ids must be unique")
        pairs: list[PairwiseComparison] = []
        output_arrays: dict[str, dict[str, NDArray[np.generic]]] = {}
        for candidate in candidates:
            pair, arrays = self._compare_pair(reference, candidate)
            pairs.append(pair)
            output_arrays[candidate.prediction.run_id] = arrays
        result = NeuralComparisonResult(
            comparison_id=comparison_id,
            metadata=ComparisonMetadata(
                comparison_version=COMPARISON_VERSION,
                reference_run_id=reference.prediction.run_id,
                candidate_run_ids=tuple(item.prediction.run_id for item in candidates),
                stimulus_sha256={
                    reference.prediction.run_id: reference.prediction.stimulus.sha256,
                    **{
                        item.prediction.run_id: item.prediction.stimulus.sha256
                        for item in candidates
                    },
                },
                model_fingerprint=reference.prediction.provenance.model_fingerprint,
                analytics_version=reference.analytics.metadata.analytics_version,
                atlas_name=reference.analytics.atlas.name,
                atlas_version=reference.analytics.atlas.version,
                functional_atlas_name="Yeo2011_7Networks_N1000",
                functional_atlas_version="2011-native-fsaverage5",
                functional_atlas_mapping_sha256=reference.network_mapping_sha256,
                functional_atlas_source_url=(
                    "https://www.freesurfer.net/pub/data/Yeo_JNeurophysiol11_FreeSurfer.zip"
                ),
                parameters={
                    "timeline_tolerance_seconds": self.tolerance_seconds,
                    "max_events": self.max_events,
                    "top_regions_per_event": self.top_regions_per_event,
                    "window_radius_samples": self.window_radius_samples,
                    "event_minimum_separation_samples": 2 * self.window_radius_samples + 1,
                    "delta_convention": "candidate_minus_reference",
                    "alignment_method": self.alignment_method,
                    "interpolation": self.alignment_method != "exact_observed_intersection",
                    "max_interpolation_gap_seconds": self.max_interpolation_gap_seconds,
                },
            ),
            pairs=tuple(pairs),
        )
        return result, output_arrays

    def _compare_pair(
        self, reference: VariantData, candidate: VariantData
    ) -> tuple[PairwiseComparison, dict[str, NDArray[np.generic]]]:
        report = validate_comparability(
            reference.prediction,
            candidate.prediction,
            reference.analytics,
            candidate.analytics,
        )
        if not report.comparable:
            fields = ", ".join(issue.field for issue in report.issues)
            raise ValueError(f"runs are not comparable: {fields}")
        if not np.array_equal(reference.vertex_to_region, candidate.vertex_to_region):
            raise ValueError("runs are not comparable: atlas vertex-to-region ordering differs")
        if not np.array_equal(reference.medial_wall_mask, candidate.medial_wall_mask):
            raise ValueError("runs are not comparable: medial-wall masks differ")
        if reference.network_names != candidate.network_names:
            raise ValueError("runs are not comparable: functional-network ordering differs")
        if reference.network_mapping_sha256 != candidate.network_mapping_sha256:
            raise ValueError("runs are not comparable: functional-network mappings differ")
        self._validate_arrays(reference, "reference")
        self._validate_arrays(candidate, "candidate")
        interpolated_samples = 0
        aligned: AlignedTimeline | InterpolatedTimeline
        interpolation: InterpolatedTimeline | None = None
        if self.alignment_method == "exact_observed_intersection":
            aligned = align_observed_times(
                reference.times, candidate.times, tolerance_seconds=self.tolerance_seconds
            )
            ref_response = self._rows(reference.responses, aligned.reference_indices, "reference")
            cand_response = self._rows(candidate.responses, aligned.candidate_indices, "candidate")
            ref_roi = self._rows(
                reference.roi_timeseries, aligned.reference_indices, "reference ROI"
            )
            cand_roi = self._rows(
                candidate.roi_timeseries, aligned.candidate_indices, "candidate ROI"
            )
            ref_network = self._rows(
                reference.network_timeseries, aligned.reference_indices, "reference network"
            )
            cand_network = self._rows(
                candidate.network_timeseries, aligned.candidate_indices, "candidate network"
            )
            candidate_indices = aligned.candidate_indices
        else:
            interpolation = align_linear_candidate_to_reference(
                reference.times,
                candidate.times,
                tolerance_seconds=self.tolerance_seconds,
                max_gap_seconds=self.max_interpolation_gap_seconds,
            )
            aligned = interpolation
            ref_response = self._rows(
                reference.responses, interpolation.reference_indices, "reference"
            )
            ref_roi = self._rows(
                reference.roi_timeseries, interpolation.reference_indices, "reference ROI"
            )
            ref_network = self._rows(
                reference.network_timeseries, interpolation.reference_indices, "reference network"
            )
            cand_response = interpolate_rows(candidate.responses, interpolation)
            cand_roi = interpolate_rows(candidate.roi_timeseries, interpolation)
            cand_network = interpolate_rows(candidate.network_timeseries, interpolation)
            candidate_indices = np.where(
                interpolation.candidate_observed_mask,
                interpolation.candidate_left_indices,
                -1,
            ).astype(np.int64)
            interpolated_samples = int((~interpolation.candidate_observed_mask).sum())
        if ref_response.shape != cand_response.shape:
            raise ValueError("aligned cortical response shapes differ")
        if ref_roi.shape != cand_roi.shape:
            raise ValueError("aligned ROI response shapes differ")
        if ref_network.shape != cand_network.shape:
            raise ValueError("aligned functional-network response shapes differ")

        cortical_delta = cand_response.astype(np.float64) - ref_response.astype(np.float64)
        cortical_delta[:, reference.medial_wall_mask] = np.nan
        roi_delta = cand_roi.astype(np.float64) - ref_roi.astype(np.float64)
        network_delta = cand_network.astype(np.float64) - ref_network.astype(np.float64)
        jointly_finite = np.isfinite(ref_response) & np.isfinite(cand_response)
        jointly_finite[:, reference.medial_wall_mask] = False
        if np.any(jointly_finite.sum(axis=1) == 0):
            raise ValueError("an aligned row has no jointly finite cortical values")
        clean_delta = np.where(jointly_finite, cortical_delta, 0.0)
        cortical_l2 = np.linalg.norm(clean_delta, axis=1)
        cortical_mean_delta = np.divide(
            clean_delta.sum(axis=1), jointly_finite.sum(axis=1), dtype=np.float64
        )
        cortical_mean_absolute_delta = np.divide(
            np.abs(clean_delta).sum(axis=1), jointly_finite.sum(axis=1), dtype=np.float64
        )
        cosine = _row_cosine(ref_response, cand_response, jointly_finite)
        hemisphere_delta = _hemisphere_mean_deltas(
            cortical_delta, reference.prediction.cortical.hemisphere_index_ranges
        )
        region_summaries = self._region_summaries(roi_delta, aligned.times, reference.analytics)
        network_summaries = self._network_summaries(
            network_delta, aligned.times, reference.network_names
        )
        events = self._rank_events(cortical_l2, cosine, roi_delta, aligned.times)
        windows = self._rank_windows(events, cortical_l2, aligned.times)
        arrays: dict[str, NDArray[np.generic]] = {
            "times": aligned.times,
            "reference_row_indices": aligned.reference_indices,
            "candidate_row_indices": candidate_indices,
            "cortical_signed_delta": cortical_delta,
            "cortical_absolute_delta": np.abs(cortical_delta),
            "roi_signed_delta": roi_delta,
            "roi_absolute_delta": np.abs(roi_delta),
            "network_signed_delta": network_delta,
            "network_absolute_delta": np.abs(network_delta),
            "global_mean_signed_delta": cortical_mean_delta,
            "global_mean_absolute_delta": cortical_mean_absolute_delta,
            "cortical_l2_difference": cortical_l2,
            "cortical_cosine_similarity": cosine,
            **{
                f"hemisphere_{name}_mean_signed_delta": values
                for name, values in hemisphere_delta.items()
            },
        }
        if interpolation is not None:
            arrays.update(
                {
                    "candidate_left_row_indices": interpolation.candidate_left_indices,
                    "candidate_right_row_indices": interpolation.candidate_right_indices,
                    "candidate_right_weights": interpolation.candidate_right_weights,
                }
            )
            contributing_candidate_rows = np.unique(
                np.concatenate(
                    (interpolation.candidate_left_indices, interpolation.candidate_right_indices)
                )
            ).size
        else:
            contributing_candidate_rows = len(candidate_indices)
        candidate_id = candidate.prediction.run_id
        return (
            PairwiseComparison(
                candidate_run_id=candidate_id,
                comparability=report,
                alignment=TimelineAlignment(
                    method=self.alignment_method,
                    tolerance_seconds=self.tolerance_seconds,
                    reference_samples=len(reference.times),
                    candidate_samples=len(candidate.times),
                    aligned_samples=len(aligned.times),
                    reference_coverage_fraction=len(aligned.times) / len(reference.times),
                    candidate_coverage_fraction=(
                        contributing_candidate_rows / len(candidate.times)
                    ),
                    first_time_seconds=float(aligned.times[0]),
                    last_time_seconds=float(aligned.times[-1]),
                    interpolation_applied=interpolated_samples > 0,
                    interpolated_candidate_samples=interpolated_samples,
                ),
                regions=region_summaries,
                networks=network_summaries,
                events=events,
                windows=windows,
                arrays_path=f"pairs/{candidate_id}.npz",
                array_keys={key: _describe(key) for key in arrays},
            ),
            arrays,
        )

    @staticmethod
    def _validate_arrays(item: VariantData, label: str) -> None:
        responses = np.asarray(item.responses)
        roi = np.asarray(item.roi_timeseries)
        network = np.asarray(item.network_timeseries)
        mapping = np.asarray(item.vertex_to_region)
        wall = np.asarray(item.medial_wall_mask)
        vertex_count = item.prediction.cortical.vertex_count
        region_count = len(item.analytics.atlas.regions)
        if responses.ndim != 2 or responses.shape != (len(item.times), vertex_count):
            raise ValueError(f"{label} cortical response shape does not match metadata")
        if roi.ndim != 2 or roi.shape != (len(item.times), region_count):
            raise ValueError(f"{label} ROI response shape does not match atlas")
        if network.ndim != 2 or network.shape != (len(item.times), len(item.network_names)):
            raise ValueError(f"{label} network response shape does not match network labels")
        if not item.network_names or len(set(item.network_names)) != len(item.network_names):
            raise ValueError(f"{label} network labels must be nonempty and unique")
        if np.any(np.isfinite(network).sum(axis=0) == 0):
            raise ValueError(f"{label} has a network with no finite response samples")
        if mapping.shape != (vertex_count,) or wall.shape != (vertex_count,):
            raise ValueError(f"{label} atlas arrays do not match cortical vertex count")
        if not np.issubdtype(mapping.dtype, np.integer) or wall.dtype != np.bool_:
            raise ValueError(f"{label} atlas arrays have invalid dtypes")
        if np.any((mapping[~wall] < 0) | (mapping[~wall] >= region_count)):
            raise ValueError(f"{label} atlas mapping contains an invalid cortical region")

    @staticmethod
    def _rows(
        array: NDArray[np.floating], indices: NDArray[np.int64], label: str
    ) -> NDArray[np.floating]:
        values = np.asarray(array)
        if values.ndim != 2 or len(values) <= int(indices.max()):
            raise ValueError(f"{label} array does not match its timeline")
        return values[indices]

    @staticmethod
    def _region_summaries(
        delta: NDArray[np.float64], times: NDArray[np.float64], analytics: NeuralAnalyticsResult
    ) -> tuple[RegionDifferenceSummary, ...]:
        summaries: list[RegionDifferenceSummary] = []
        for region in analytics.atlas.regions:
            values = delta[:, region.region_id]
            finite = np.isfinite(values)
            if not finite.any():
                raise ValueError(f"ROI {region.region_id} has no finite aligned differences")
            peak = int(np.nanargmax(np.abs(values)))
            summaries.append(
                RegionDifferenceSummary(
                    region_id=region.region_id,
                    name=region.name,
                    hemisphere=region.hemisphere.value,
                    mean_signed_delta=float(np.nanmean(values)),
                    mean_absolute_delta=float(np.nanmean(np.abs(values))),
                    rms_delta=float(np.sqrt(np.nanmean(values * values))),
                    peak_absolute_delta=float(abs(values[peak])),
                    peak_signed_delta=float(values[peak]),
                    peak_time_seconds=float(times[peak]),
                )
            )
        return tuple(summaries)

    @staticmethod
    def _network_summaries(
        delta: NDArray[np.float64], times: NDArray[np.float64], names: tuple[str, ...]
    ) -> tuple[NetworkDifferenceSummary, ...]:
        summaries: list[NetworkDifferenceSummary] = []
        for network_id, name in enumerate(names):
            values = delta[:, network_id]
            if not np.isfinite(values).any():
                raise ValueError(f"network {network_id} has no finite aligned differences")
            peak = int(np.nanargmax(np.abs(values)))
            summaries.append(
                NetworkDifferenceSummary(
                    network_id=network_id,
                    name=name,
                    mean_signed_delta=float(np.nanmean(values)),
                    mean_absolute_delta=float(np.nanmean(np.abs(values))),
                    rms_delta=float(np.sqrt(np.nanmean(values * values))),
                    peak_absolute_delta=float(abs(values[peak])),
                    peak_signed_delta=float(values[peak]),
                    peak_time_seconds=float(times[peak]),
                )
            )
        return tuple(summaries)

    def _rank_events(
        self,
        l2: NDArray[np.float64],
        cosine: NDArray[np.float64],
        roi_delta: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> tuple[DivergenceEvent, ...]:
        order = np.argsort(l2)[::-1]
        events: list[DivergenceEvent] = []
        selected: list[int] = []
        for index in order:
            if not np.isfinite(l2[index]) or l2[index] <= 0:
                continue
            too_close = any(
                abs(int(index) - previous) <= 2 * self.window_radius_samples
                for previous in selected
            )
            if too_close:
                continue
            finite_regions = np.flatnonzero(np.isfinite(roi_delta[index]))
            top = finite_regions[np.argsort(np.abs(roi_delta[index, finite_regions]))[::-1]]
            rank = len(events) + 1
            events.append(
                DivergenceEvent(
                    rank=rank,
                    aligned_index=int(index),
                    timestamp_seconds=float(times[index]),
                    score=float(l2[index]),
                    cortical_l2_difference=float(l2[index]),
                    cortical_cosine_similarity=(
                        float(cosine[index]) if np.isfinite(cosine[index]) else None
                    ),
                    top_region_ids=tuple(int(value) for value in top[: self.top_regions_per_event]),
                )
            )
            selected.append(int(index))
            if len(events) == self.max_events:
                break
        return tuple(events)

    def _rank_windows(
        self,
        events: tuple[DivergenceEvent, ...],
        l2: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> tuple[DivergenceWindow, ...]:
        candidates: list[tuple[int, int]] = []
        for event in events:
            start = max(0, event.aligned_index - self.window_radius_samples)
            end = min(len(times), event.aligned_index + self.window_radius_samples + 1)
            bounds = (start, end)
            if bounds not in candidates:
                candidates.append(bounds)
        windows: list[DivergenceWindow] = []
        for start, end in candidates:
            values = l2[start:end]
            peak_local = int(np.argmax(values))
            windows.append(
                DivergenceWindow(
                    rank=1,
                    start_time_seconds=float(times[start]),
                    end_time_seconds=float(times[end - 1]),
                    observed_samples=end - start,
                    mean_l2_difference=float(values.mean()),
                    peak_l2_difference=float(values[peak_local]),
                    peak_time_seconds=float(times[start + peak_local]),
                )
            )
        windows.sort(key=lambda item: item.mean_l2_difference, reverse=True)
        return tuple(item.model_copy(update={"rank": rank}) for rank, item in enumerate(windows, 1))


def _row_cosine(
    reference: NDArray[np.floating],
    candidate: NDArray[np.floating],
    finite: NDArray[np.bool_],
) -> NDArray[np.float64]:
    first = np.where(finite, reference, 0.0).astype(np.float64)
    second = np.where(finite, candidate, 0.0).astype(np.float64)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.full(len(first), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _hemisphere_mean_deltas(
    delta: NDArray[np.float64], ranges: dict[str, tuple[int, int]]
) -> dict[str, NDArray[np.float64]]:
    output: dict[str, NDArray[np.float64]] = {}
    for name, (start, end) in ranges.items():
        if not 0 <= start < end <= delta.shape[1]:
            raise ValueError(f"invalid {name} hemisphere range")
        output[name] = np.nanmean(delta[:, start:end], axis=1)
    if "left" in output and "right" in output:
        output["left_minus_right"] = output["left"] - output["right"]
    return output


def _describe(key: str) -> str:
    descriptions = {
        "cortical_signed_delta": "Candidate minus reference predicted response [time, vertex]",
        "cortical_absolute_delta": "Absolute candidate-minus-reference response [time, vertex]",
        "roi_signed_delta": "Candidate minus reference ROI mean response [time, region]",
        "roi_absolute_delta": "Absolute candidate-minus-reference ROI response [time, region]",
        "network_signed_delta": "Candidate minus reference Yeo response [time, network]",
        "network_absolute_delta": "Absolute Yeo response difference [time, network]",
        "cortical_l2_difference": "L2 magnitude of cortical signed difference per aligned time",
        "cortical_cosine_similarity": "Cosine similarity of cortical patterns per aligned time",
    }
    return descriptions.get(key, key.replace("_", " "))
