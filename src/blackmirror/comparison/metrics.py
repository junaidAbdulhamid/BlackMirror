"""Allowlisted mathematical outputs for scientifically bounded comparisons."""

from __future__ import annotations

from enum import StrEnum


class SafeComparisonMetric(StrEnum):
    SIGNED_DELTA = "signed_delta"
    ABSOLUTE_DELTA = "absolute_delta"
    RMS_DELTA = "rms_delta"
    L2_DIFFERENCE = "l2_difference"
    COSINE_SIMILARITY = "cosine_similarity"


class UnsupportedComparisonMetricError(ValueError):
    """Raised instead of fabricating psychological, causal, or goal scores."""


def require_safe_metrics(names: tuple[str, ...]) -> tuple[SafeComparisonMetric, ...]:
    safe: list[SafeComparisonMetric] = []
    for name in names:
        try:
            safe.append(SafeComparisonMetric(name))
        except ValueError as exc:
            raise UnsupportedComparisonMetricError(
                f"unsupported comparison metric {name!r}; only descriptive mathematical "
                "differences/similarities are available, not causal, psychological, or "
                "goal-conditioned scores"
            ) from exc
    return tuple(safe)
