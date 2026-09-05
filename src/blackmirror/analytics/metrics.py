"""Numerically defined Phase 3 metrics over predicted response arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blackmirror.analytics.atlas import UNMAPPED


def _matrix(responses: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(responses, dtype=np.float64)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("responses must be a non-empty [time, vertex] matrix")
    return array


def aggregate_regions(
    responses: NDArray[np.floating], vertex_to_region: NDArray[np.integer], n_regions: int
) -> NDArray[np.float64]:
    """Vectorized finite-value mean from ``T x V`` to ``T x R``.

    ``numpy.add.at`` performs indexed reductions in compiled code without a
    time-by-region-by-vertex Python loop or a large dense membership matrix.
    NaN/Inf samples are omitted independently per timestep and region; an ROI
    with no finite samples at a timestep is NaN.
    """
    values = _matrix(responses)
    mapping = np.asarray(vertex_to_region)
    if mapping.shape != (values.shape[1],):
        raise ValueError("vertex mapping length must equal response vertex count")
    if n_regions <= 0:
        raise ValueError("n_regions must be positive")
    if np.any((mapping < UNMAPPED) | (mapping >= n_regions)):
        raise ValueError("vertex mapping contains invalid region ids")
    mapped = mapping >= 0
    selected = values[:, mapped]
    finite = np.isfinite(selected)
    totals = np.zeros((n_regions, values.shape[0]), dtype=np.float64)
    counts = np.zeros_like(totals)
    np.add.at(totals, mapping[mapped], np.where(finite, selected, 0.0).T)
    np.add.at(counts, mapping[mapped], finite.T)
    means = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    return means.T


def aggregate_hemispheres(
    responses: NDArray[np.floating],
    hemisphere_ranges: dict[str, tuple[int, int]],
    excluded_vertices: NDArray[np.bool_] | None = None,
) -> dict[str, NDArray[np.float64]]:
    values = _matrix(responses)
    excluded = _excluded_mask(excluded_vertices, values.shape[1])
    result: dict[str, NDArray[np.float64]] = {}
    for name, (start, end) in hemisphere_ranges.items():
        if not 0 <= start < end <= values.shape[1]:
            raise ValueError(f"invalid {name} hemisphere range {(start, end)}")
        selected = values[:, start:end].copy()
        selected[:, excluded[start:end]] = np.nan
        result[name] = np.nanmean(np.where(np.isfinite(selected), selected, np.nan), axis=1)
    if "left" in result and "right" in result:
        result["left_minus_right"] = result["left"] - result["right"]
    return result


def global_metrics(
    responses: NDArray[np.floating], excluded_vertices: NDArray[np.bool_] | None = None
) -> dict[str, NDArray[np.float64]]:
    values = _matrix(responses)
    excluded = _excluded_mask(excluded_vertices, values.shape[1])
    values = values.copy()
    values[:, excluded] = np.nan
    clean = np.where(np.isfinite(values), values, np.nan)
    # L2 is scaled by sqrt(number finite) as an additional RMS magnitude so
    # comparisons remain meaningful if a few anomalous vertices are omitted.
    squared = np.where(np.isfinite(values), values * values, 0.0)
    finite_count = np.isfinite(values).sum(axis=1)
    valid_rows = finite_count > 0
    l2 = np.sqrt(squared.sum(axis=1))
    rms = np.sqrt(
        np.divide(
            squared.sum(axis=1),
            finite_count,
            out=np.full(values.shape[0], np.nan, dtype=np.float64),
            where=finite_count > 0,
        )
    )
    l2[finite_count == 0] = np.nan
    mean = np.full(values.shape[0], np.nan, dtype=np.float64)
    median = mean.copy()
    std = mean.copy()
    variance = mean.copy()
    mean[valid_rows] = np.nansum(clean[valid_rows], axis=1) / finite_count[valid_rows]
    median[valid_rows] = np.nanmedian(clean[valid_rows], axis=1)
    std[valid_rows] = np.nanstd(clean[valid_rows], axis=1)
    variance[valid_rows] = np.nanvar(clean[valid_rows], axis=1)
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "spatial_variance": variance,
        "l2_magnitude": l2,
        "rms_magnitude": rms,
    }


def change_magnitude(
    responses: NDArray[np.floating], excluded_vertices: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    values = _matrix(responses)
    excluded = _excluded_mask(excluded_vertices, values.shape[1])
    delta = np.diff(values, axis=0, prepend=values[[0]])
    finite = np.isfinite(delta) & ~excluded[None, :]
    squared = np.where(finite, delta * delta, 0.0).sum(axis=1)
    result = np.sqrt(squared)
    result[finite.sum(axis=1) == 0] = np.nan
    # By definition there is no transition before the first sample.
    result[0] = 0.0
    return result


def spatial_concentration(
    responses: NDArray[np.floating], excluded_vertices: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """Normalized Herfindahl concentration of absolute response, in [0, 1].

    Zero means equal absolute magnitude at every finite vertex; one means all
    absolute magnitude is concentrated at one vertex. All-zero rows return 0.
    """
    values = _matrix(responses)
    excluded = _excluded_mask(excluded_vertices, values.shape[1])
    values = values.copy()
    values[:, excluded] = np.nan
    absolute = np.where(np.isfinite(values), np.abs(values), 0.0)
    totals = absolute.sum(axis=1, keepdims=True)
    shares = np.divide(absolute, totals, out=np.zeros_like(absolute), where=totals > 0)
    hhi = np.square(shares).sum(axis=1)
    n = np.maximum(1, np.isfinite(values).sum(axis=1))
    return np.where(n > 1, (hhi - 1 / n) / (1 - 1 / n), 0.0)


def _excluded_mask(mask: NDArray[np.bool_] | None, vertex_count: int) -> NDArray[np.bool_]:
    if mask is None:
        return np.zeros(vertex_count, dtype=bool)
    excluded = np.asarray(mask, dtype=bool)
    if excluded.shape != (vertex_count,):
        raise ValueError("excluded vertex mask length must equal response vertex count")
    return excluded


def cosine_similarity(first: NDArray[np.floating], second: NDArray[np.floating]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("patterns must be same-length vectors")
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    a, b = a[finite], b[finite]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def temporal_similarity_matrix(responses: NDArray[np.floating]) -> NDArray[np.float64]:
    values = _matrix(responses)
    if not np.isfinite(values).all():
        raise ValueError("temporal similarity matrix requires finite responses")
    norms = np.linalg.norm(values, axis=1)
    denominator = np.outer(norms, norms)
    output = np.full((len(values), len(values)), np.nan, dtype=np.float64)
    return np.divide(values @ values.T, denominator, out=output, where=denominator > 0)


def roi_correlation_matrix(roi_timeseries: NDArray[np.floating]) -> NDArray[np.float64]:
    values = _matrix(roi_timeseries)
    n_regions = values.shape[1]
    output = np.full((n_regions, n_regions), np.nan, dtype=np.float64)
    for first in range(n_regions):
        for second in range(first, n_regions):
            a, b = values[:, first], values[:, second]
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() < 2 or np.std(a[finite]) == 0 or np.std(b[finite]) == 0:
                continue
            correlation = float(np.corrcoef(a[finite], b[finite])[0, 1])
            output[first, second] = output[second, first] = correlation
    return output


@dataclass(frozen=True)
class WindowAnalytics:
    start_time: float
    end_time: float
    mean_response: float
    mean_magnitude: float
    largest_change: float
    peak_timestep: int
    top_region_ids: tuple[int, ...]


def analyze_window(
    responses: NDArray[np.floating],
    roi_timeseries: NDArray[np.floating],
    times: NDArray[np.floating],
    start: float,
    end: float,
    *,
    top_k: int = 5,
) -> WindowAnalytics:
    values = _matrix(responses)
    clock = np.asarray(times, dtype=np.float64)
    if clock.shape != (values.shape[0],):
        raise ValueError("times must have one value per response row")
    if start > end:
        raise ValueError("window start must not exceed end")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    indices = np.flatnonzero((clock >= start) & (clock <= end))
    if not len(indices):
        raise ValueError("window contains no response samples")
    subset = values[indices]
    metrics = global_metrics(subset)
    changes = change_magnitude(subset)
    region_means = np.nanmean(np.asarray(roi_timeseries)[indices], axis=0)
    order = np.argsort(np.nan_to_num(np.abs(region_means), nan=-np.inf))[::-1][
        : min(top_k, region_means.size)
    ]
    peak_local = int(np.nanargmax(metrics["l2_magnitude"]))
    return WindowAnalytics(
        start_time=float(clock[indices[0]]),
        end_time=float(clock[indices[-1]]),
        mean_response=float(np.nanmean(subset)),
        mean_magnitude=float(np.nanmean(metrics["l2_magnitude"])),
        largest_change=float(np.nanmax(changes)),
        peak_timestep=int(indices[peak_local]),
        top_region_ids=tuple(int(value) for value in order),
    )


def top_vertices(
    responses: NDArray[np.floating], timestep: int, k: int
) -> tuple[tuple[int, float], ...]:
    values = _matrix(responses)
    if not 0 <= timestep < values.shape[0] or not 1 <= k <= values.shape[1]:
        raise ValueError("invalid timestep or k")
    row = values[timestep]
    order = np.argsort(np.nan_to_num(np.abs(row), nan=-np.inf))[::-1][:k]
    return tuple((int(index), float(row[index])) for index in order)


def top_regions(
    roi_timeseries: NDArray[np.floating], timestep: int, k: int
) -> tuple[tuple[int, float], ...]:
    return top_vertices(roi_timeseries, timestep, k)
