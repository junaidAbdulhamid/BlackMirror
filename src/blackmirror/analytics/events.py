"""Transparent, parameterized event detection over mathematical time series."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from blackmirror.analytics.schemas import NeuralEvent, NeuralEventType


class NeuralEventDetector:
    def __init__(self, *, z_threshold: float = 1.5, min_distance: int = 2, max_events: int = 20):
        if z_threshold < 0 or min_distance < 1 or max_events < 1:
            raise ValueError("invalid event detector parameters")
        self.z_threshold = z_threshold
        self.min_distance = min_distance
        self.max_events = max_events

    def detect(
        self,
        global_response: NDArray[np.floating],
        magnitude: NDArray[np.floating],
        changes: NDArray[np.floating],
        roi_timeseries: NDArray[np.floating],
        times: NDArray[np.floating],
    ) -> tuple[NeuralEvent, ...]:
        series = (
            (NeuralEventType.RESPONSE_PEAK, np.asarray(magnitude, dtype=float)),
            (NeuralEventType.LARGE_TRANSITION, np.asarray(changes, dtype=float)),
        )
        candidates: list[NeuralEvent] = []
        for event_type, values in series:
            for index, score in self._peaks(values):
                region_ids: tuple[int, ...] = ()
                if roi_timeseries.shape[1]:
                    ranked = np.nan_to_num(
                        np.abs(roi_timeseries[index]), nan=-np.inf, posinf=-np.inf
                    )
                    region_ids = tuple(
                        int(i) for i in np.argsort(ranked)[::-1][:3] if np.isfinite(ranked[i])
                    )
                identity = f"{event_type}:{index}:{float(times[index]):.9g}"
                candidates.append(
                    NeuralEvent(
                        event_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                        timestep=index,
                        timestamp_seconds=float(times[index]),
                        event_type=event_type,
                        score=score,
                        affected_region_ids=region_ids,
                        global_response=float(global_response[index]),
                        change_magnitude=float(changes[index]),
                    )
                )
        candidates.sort(key=lambda event: event.score, reverse=True)
        return tuple(candidates[: self.max_events])

    def _peaks(self, values: NDArray[np.float64]) -> tuple[tuple[int, float], ...]:
        if len(values) < 3 or not np.isfinite(values).any():
            return ()
        mean, std = float(np.nanmean(values)), float(np.nanstd(values))
        threshold = mean + self.z_threshold * std
        raw = [
            i
            for i in range(1, len(values) - 1)
            if np.isfinite(values[i])
            and values[i] >= threshold
            and values[i] > values[i - 1]
            and values[i] >= values[i + 1]
        ]
        selected: list[int] = []
        for index in sorted(raw, key=lambda i: values[i], reverse=True):
            if all(abs(index - other) >= self.min_distance for other in selected):
                selected.append(index)
        return tuple((i, float(max(0.0, (values[i] - mean) / (std or 1.0)))) for i in selected)
