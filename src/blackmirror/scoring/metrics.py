"""The objective metric registry.

WHAT IT IS
    A metric turns a windowed slice of predicted response into one number.
    Each is a small class implementing `validate`, `compute` and `describe`,
    registered under a stable name that objectives refer to by string.

WHY A REGISTRY RATHER THAN A BRANCH
    Objectives are user-authored data, and the set of defensible metrics will
    grow. A dispatch chain forces every addition to edit the scoring engine,
    and makes it easy to add a metric with no documented formula. A registry
    makes the contract explicit: a metric that cannot state its formula, its
    supported targets and its limitations cannot be registered.

WHAT A METRIC RECEIVES
    A `MetricInput` carrying the already-resolved series for the target and
    window, plus the vertex-level slice when the metric genuinely needs it.
    Resolution happens once, outside the metric, so every metric measures the
    same interval and no metric can quietly redefine the window.

WHAT NO METRIC MAY DO
    Return a psychological quantity, or a value whose formula is not written in
    its `describe()`. `SafeComparisonMetric` in Phase 5 established this
    boundary for differences; this is the same boundary for objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MetricInput:
    """Everything a metric may look at, resolved once by the evaluator.

    `series` is the target's response over the window: for an ROI or network
    target it is the aggregated timeseries, for a vertex-set target it is the
    mean across the selected vertices. `vertex_block` is the raw
    [window_sample, selected_vertex] slice, for metrics that need the spatial
    pattern rather than its average.
    """

    series: NDArray[np.float64]
    times: NDArray[np.float64]
    vertex_block: NDArray[np.float64] | None = None
    #: Reference pattern for similarity metrics, with provenance recorded by
    #: the caller. Never synthesised.
    reference_pattern: NDArray[np.float64] | None = None
    #: Baseline variant's series over its own resolved window, for divergence.
    baseline_series: NDArray[np.float64] | None = None
    baseline_times: NDArray[np.float64] | None = None
    parameters: dict[str, float | int | str | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricOutput:
    value: float | None
    statistics: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class ObjectiveMetric(Protocol):
    """The contract every metric satisfies."""

    name: str
    #: Target types this metric is meaningful for.
    supported_targets: frozenset[str]
    #: True when the metric needs a baseline variant to be declared.
    requires_baseline: bool
    #: True when the metric needs an explicit reference pattern.
    requires_reference: bool

    def validate(self, data: MetricInput) -> tuple[str, ...]:
        """Return blocking problems with this input. Empty means computable."""
        ...

    def compute(self, data: MetricInput) -> MetricOutput:
        """Return the metric value in model units."""
        ...

    def describe(self) -> dict[str, str]:
        """Formula, interpretation and limitations, for docs and provenance."""
        ...

    def decompose(self, data: MetricInput) -> NDArray[np.float64] | None:
        """Per-sample contribution to the raw value, or None if not decomposable.

        The contract is strict: when an array is returned its entries MUST sum
        to the raw value. That is what makes a "this interval contributed 40% of
        the score" statement true rather than decorative, and it is asserted in
        the test suite for every decomposable metric.

        A metric returns None when its value is not a sum over samples -- an RMS
        or a cosine cannot be split this way, and inventing a split would
        manufacture structure that is not in the mathematics.
        """
        ...


_ALL_TARGETS = frozenset(
    {"whole_cortex", "roi", "network", "hemisphere", "custom_vertex_set"}
)


def _finite(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values[np.isfinite(values)]


def _base_statistics(series: NDArray[np.float64]) -> dict[str, float]:
    finite = _finite(series)
    if not finite.size:
        return {"sample_count": float(series.size), "finite_count": 0.0}
    return {
        "sample_count": float(series.size),
        "finite_count": float(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


class _SeriesMetric:
    """Shared behaviour for metrics that read the target's 1-D series."""

    name = "abstract"
    supported_targets = _ALL_TARGETS
    requires_baseline = False
    requires_reference = False
    #: Minimum finite samples for the value to mean anything.
    minimum_samples = 1

    def validate(self, data: MetricInput) -> tuple[str, ...]:
        problems: list[str] = []
        if data.series.ndim != 1:
            problems.append("metric expects a one-dimensional response series")
        finite = _finite(data.series)
        if finite.size < self.minimum_samples:
            problems.append(
                f"{self.name} needs at least {self.minimum_samples} finite sample(s) in the "
                f"window; the resolved window has {finite.size}"
            )
        return tuple(problems)

    def decompose(self, data: MetricInput) -> NDArray[np.float64] | None:
        return None

    def describe(self) -> dict[str, str]:  # pragma: no cover - overridden
        raise NotImplementedError


class MeanResponseMetric(_SeriesMetric):
    name = "MEAN_RESPONSE"

    def compute(self, data: MetricInput) -> MetricOutput:
        finite = _finite(data.series)
        return MetricOutput(
            value=float(finite.mean()) if finite.size else None,
            statistics=_base_statistics(data.series),
        )

    def decompose(self, data: MetricInput) -> NDArray[np.float64] | None:
        mask = np.isfinite(data.series)
        count = int(mask.sum())
        if not count:
            return None
        shares = np.zeros_like(data.series, dtype=np.float64)
        shares[mask] = data.series[mask] / count
        return shares

    def describe(self) -> dict[str, str]:
        return {
            "formula": "mean over finite samples t in W of R_target(t)",
            "raw_output": "model response units, the same units as the prediction",
            "interpretation": (
                "The average predicted response of the selected target across the "
                "selected interval."
            ),
            "limitations": (
                "A mean hides shape: a brief spike and a sustained plateau with the "
                "same average are indistinguishable. Use INTEGRATED_RESPONSE or "
                "PEAK_RESPONSE when shape matters. Signed responses can cancel."
            ),
        }


class PeakResponseMetric(_SeriesMetric):
    name = "PEAK_RESPONSE"

    def compute(self, data: MetricInput) -> MetricOutput:
        finite = _finite(data.series)
        if not finite.size:
            return MetricOutput(value=None, statistics=_base_statistics(data.series))
        statistics = _base_statistics(data.series)
        index = int(np.nanargmax(np.where(np.isfinite(data.series), data.series, -np.inf)))
        statistics["peak_time_seconds"] = float(data.times[index])
        statistics["peak_sample_index"] = float(index)
        warnings: tuple[str, ...] = ()
        if finite.size < 5:
            warnings = (
                f"peak taken over only {finite.size} sample(s); a maximum over few "
                f"points is highly sensitive to a single value",
            )
        return MetricOutput(value=float(finite.max()), statistics=statistics, warnings=warnings)

    def decompose(self, data: MetricInput) -> NDArray[np.float64] | None:
        mask = np.isfinite(data.series)
        if not mask.any():
            return None
        shares = np.zeros_like(data.series, dtype=np.float64)
        index = int(np.argmax(np.where(mask, data.series, -np.inf)))
        # A maximum is produced by one sample. Spreading it would be fiction.
        shares[index] = float(data.series[index])
        return shares

    def describe(self) -> dict[str, str]:
        return {
            "formula": "max over finite samples t in W of R_target(t)",
            "raw_output": "model response units",
            "interpretation": "The single largest predicted response within the interval.",
            "limitations": (
                "An extremum, so it is determined by one sample and is the most "
                "outlier-sensitive metric here. At TR=1s a 5-second window holds 5 "
                "samples, and the metric warns below 5 finite samples."
            ),
        }


class IntegratedResponseMetric(_SeriesMetric):
    name = "INTEGRATED_RESPONSE"
    minimum_samples = 2

    def compute(self, data: MetricInput) -> MetricOutput:
        mask = np.isfinite(data.series)
        values, times = data.series[mask], data.times[mask]
        if values.size < 2:
            return MetricOutput(value=None, statistics=_base_statistics(data.series))
        statistics = _base_statistics(data.series)
        statistics["integration_span_seconds"] = float(times[-1] - times[0])
        warnings: tuple[str, ...] = ()
        if int(mask.size - values.size):
            warnings = (
                f"{int(mask.size - values.size)} non-finite sample(s) were dropped before "
                f"integration, so the trapezoid spans across the gap rather than "
                f"treating it as missing area",
            )
        return MetricOutput(
            value=float(np.trapezoid(values, times)),
            statistics=statistics,
            warnings=warnings,
        )

    def decompose(self, data: MetricInput) -> NDArray[np.float64] | None:
        mask = np.isfinite(data.series)
        positions = np.flatnonzero(mask)
        if positions.size < 2:
            return None
        shares = np.zeros_like(data.series, dtype=np.float64)
        values, times = data.series[positions], data.times[positions]
        # Each trapezoid's area is credited half to each of its endpoints, so
        # the per-sample shares sum exactly to the reported integral.
        areas = (values[:-1] + values[1:]) / 2.0 * np.diff(times)
        for offset, area in enumerate(areas):
            shares[positions[offset]] += area / 2.0
            shares[positions[offset + 1]] += area / 2.0
        return shares

    def describe(self) -> dict[str, str]:
        return {
            "formula": "trapezoidal integral of R_target(t) dt over W",
            "raw_output": "model response units x seconds",
            "interpretation": (
                "Area under the response curve. Unlike the mean it grows with window "
                "length, so it distinguishes a brief spike from a sustained response."
            ),
            "limitations": (
                "Because it scales with duration it is NOT comparable across windows "
                "of different length, which content-event scopes routinely produce. "
                "Trapezoidal integration assumes linear behaviour between samples; at "
                "TR=1s that is a coarse assumption."
            ),
        }


class ResponseStabilityMetric(_SeriesMetric):
    name = "RESPONSE_STABILITY"
    minimum_samples = 2

    def compute(self, data: MetricInput) -> MetricOutput:
        finite = _finite(data.series)
        if finite.size < 2:
            return MetricOutput(value=None, statistics=_base_statistics(data.series))
        return MetricOutput(
            value=float(finite.std(ddof=1)),
            statistics=_base_statistics(data.series),
        )

    def describe(self) -> dict[str, str]:
        return {
            "formula": "sample standard deviation (ddof=1) of R_target(t) over W",
            "raw_output": "model response units",
            "interpretation": (
                "Temporal variability of the predicted response. Pair it with "
                "direction MINIMIZE to prefer a steadier response."
            ),
            "limitations": (
                "This is Temporal Response Variability and nothing else. It is not "
                "attention stability, engagement consistency, or any viewer state. "
                "A coefficient of variation is deliberately not offered: these "
                "responses are signed and cross zero, so dividing by the mean is "
                "undefined in the region that matters."
            ),
        }


class TemporalChangeMagnitudeMetric(_SeriesMetric):
    name = "TEMPORAL_CHANGE_MAGNITUDE"
    minimum_samples = 2

    def compute(self, data: MetricInput) -> MetricOutput:
        mask = np.isfinite(data.series)
        values = data.series[mask]
        if values.size < 2:
            return MetricOutput(value=None, statistics=_base_statistics(data.series))
        differences = np.abs(np.diff(values))
        statistics = _base_statistics(data.series)
        statistics["max_step_change"] = float(differences.max())
        return MetricOutput(value=float(differences.mean()), statistics=statistics)

    def describe(self) -> dict[str, str]:
        return {
            "formula": "mean of |R_target(t+1) - R_target(t)| over consecutive finite samples in W",
            "raw_output": "model response units per sample",
            "interpretation": (
                "How much the predicted response moves between samples. Larger means "
                "a more dynamic response over the interval."
            ),
            "limitations": (
                "Per-sample, not per-second: at a fixed TR the two coincide, but the "
                "value is not comparable across runs with different TRs. Non-finite "
                "samples are dropped, so a difference can span a gap."
            ),
        }


class PatternSimilarityMetric:
    """Cosine similarity between the target's spatial pattern and a reference.

    The reference must be supplied with provenance by the caller. There is no
    "ideal" cortical pattern to compare against, and this metric will not
    invent one: a valid reference is another stimulus, a baseline variant, or a
    researcher-supplied template that can be cited.
    """

    name = "PATTERN_SIMILARITY_TO_REFERENCE"
    supported_targets = _ALL_TARGETS
    requires_baseline = False
    requires_reference = True
    minimum_samples = 1

    def validate(self, data: MetricInput) -> tuple[str, ...]:
        problems: list[str] = []
        if data.vertex_block is None:
            problems.append("pattern similarity needs the vertex-level response block")
        if data.reference_pattern is None:
            problems.append(
                "pattern similarity needs an explicit reference pattern with provenance"
            )
        if data.vertex_block is not None and data.reference_pattern is not None:
            if data.vertex_block.ndim != 2:
                problems.append("vertex block must be [sample, vertex]")
            elif data.vertex_block.shape[1] != data.reference_pattern.shape[-1]:
                problems.append(
                    f"reference pattern has {data.reference_pattern.shape[-1]} vertices but "
                    f"the target selects {data.vertex_block.shape[1]}"
                )
        return tuple(problems)

    def compute(self, data: MetricInput) -> MetricOutput:
        assert data.vertex_block is not None and data.reference_pattern is not None
        # The window's mean spatial pattern, compared against the reference.
        block = data.vertex_block
        finite_rows = np.isfinite(block).all(axis=1)
        if not finite_rows.any():
            return MetricOutput(
                value=None,
                warnings=("no window sample had a fully finite spatial pattern",),
            )
        pattern = block[finite_rows].mean(axis=0)
        reference = np.asarray(data.reference_pattern, dtype=np.float64).reshape(-1)
        jointly = np.isfinite(pattern) & np.isfinite(reference)
        if not jointly.any():
            return MetricOutput(value=None, warnings=("no jointly finite vertices",))
        left, right = pattern[jointly], reference[jointly]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0:
            return MetricOutput(
                value=None, warnings=("a pattern has zero magnitude; cosine is undefined",)
            )
        return MetricOutput(
            value=float(np.dot(left, right) / denominator),
            statistics={
                "compared_vertices": float(jointly.sum()),
                "window_samples_used": float(finite_rows.sum()),
            },
        )

    def decompose(self, data: MetricInput) -> NDArray[np.float64] | None:
        # Cosine of a window-averaged pattern is not a sum over samples, so
        # there is no per-sample share that adds up to it.
        return None

    def describe(self) -> dict[str, str]:
        return {
            "formula": (
                "cosine(mean_t in W of R_target(t, v), reference(v)) over jointly finite v"
            ),
            "raw_output": "dimensionless, in [-1, 1]",
            "interpretation": (
                "Directional similarity between the window's average spatial response "
                "pattern and a supplied reference pattern."
            ),
            "limitations": (
                "Uncentred cosine is dominated by any shared offset, so two patterns "
                "can both score near 1 while differing substantially in structure. It "
                "measures direction, not magnitude. The reference is an input with its "
                "own provenance; the metric asserts nothing about the reference being "
                "desirable."
            ),
        }


class DivergenceFromBaselineMetric:
    """RMS difference between this variant and a declared baseline variant.

    Availability-normalised for the same reason Phase 5 ranks on RMS rather than
    L2: a raw L2 grows with how many samples and vertices happened to be finite,
    which would rank variants by data availability.
    """

    name = "DIVERGENCE_FROM_BASELINE"
    supported_targets = _ALL_TARGETS
    requires_baseline = True
    requires_reference = False
    minimum_samples = 1

    def validate(self, data: MetricInput) -> tuple[str, ...]:
        problems: list[str] = []
        if data.baseline_series is None:
            problems.append("divergence needs a declared baseline variant")
        elif data.baseline_series.shape != data.series.shape:
            problems.append(
                f"baseline window has {data.baseline_series.shape[0]} sample(s) but this "
                f"variant's window has {data.series.shape[0]}; divergence requires windows "
                f"of equal length, so compare on an absolute or normalized window rather "
                f"than a content event whose duration differs between variants"
            )
        return tuple(problems)

    def compute(self, data: MetricInput) -> MetricOutput:
        assert data.baseline_series is not None
        difference = data.series - data.baseline_series
        finite = _finite(difference)
        if not finite.size:
            return MetricOutput(value=None, warnings=("no jointly finite samples",))
        warnings: tuple[str, ...] = ()
        if data.baseline_times is not None and not np.array_equal(data.times, data.baseline_times):
            warnings = (
                "baseline and variant timestamps differ; equal-length samples were paired "
                "by ordered position within their separately resolved windows",
            )
        return MetricOutput(
            value=float(np.sqrt(np.mean(finite**2))),
            statistics={
                "finite_count": float(finite.size),
                "mean_signed_difference": float(finite.mean()),
                "max_absolute_difference": float(np.abs(finite).max()),
            },
            warnings=warnings,
        )

    def decompose(self, data: MetricInput) -> NDArray[np.float64] | None:
        # An RMS is not additive over samples: the square root means per-sample
        # shares cannot sum to the value. Phase 5's signed delta arrays are the
        # right place to ask where two variants differ most.
        return None

    def describe(self) -> dict[str, str]:
        return {
            "formula": "sqrt(mean over finite t in W of (R_variant(t) - R_baseline(t))^2)",
            "raw_output": "model response units",
            "interpretation": (
                "How far this variant's predicted response departs from the baseline's "
                "over the window. Sign is discarded; the mean signed difference is "
                "reported separately in statistics."
            ),
            "limitations": (
                "Divergence is not quality. A variant that differs most from the "
                "baseline is the most different, not the best, and the direction of "
                "the objective decides nothing about that. Requires equal-length "
                "windows in both variants. Samples are paired by ordered position within "
                "each resolved window; differing timestamp grids are reported explicitly."
            ),
        }


class MetricRegistry:
    """Name to metric, with registration refused for an undocumented metric."""

    def __init__(self) -> None:
        self._metrics: dict[str, ObjectiveMetric] = {}

    def register(self, metric: ObjectiveMetric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"metric {metric.name} is already registered")
        description = metric.describe()
        missing = {"formula", "interpretation", "limitations"} - set(description)
        if missing:
            raise ValueError(
                f"metric {metric.name} cannot be registered without {sorted(missing)}; a "
                f"metric whose formula and limits are not stated cannot be audited"
            )
        self._metrics[metric.name] = metric

    def get(self, name: str) -> ObjectiveMetric:
        try:
            return self._metrics[name]
        except KeyError:
            raise UnknownMetricError(
                f"unknown objective metric {name!r}; registered metrics are "
                f"{sorted(self._metrics)}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._metrics))

    def describe_all(self) -> dict[str, dict[str, str]]:
        return {name: metric.describe() for name, metric in sorted(self._metrics.items())}


class UnknownMetricError(ValueError):
    """Raised instead of silently scoring nothing."""


def default_registry() -> MetricRegistry:
    registry = MetricRegistry()
    for metric in (
        MeanResponseMetric(),
        PeakResponseMetric(),
        IntegratedResponseMetric(),
        ResponseStabilityMetric(),
        TemporalChangeMagnitudeMetric(),
        PatternSimilarityMetric(),
        DivergenceFromBaselineMetric(),
    ):
        registry.register(metric)
    return registry
