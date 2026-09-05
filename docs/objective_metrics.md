# Objective Metrics

**Generated from the metric registry** (`blackmirror.scoring.metrics.default_registry`).
A metric cannot be registered without a formula, an interpretation and its
limitations, so this file cannot silently drift from the code.

Every metric below is a mathematical function of a *predicted* cortical response.
None of them measures a psychological state, and none should be renamed as though
it did.

| Metric | Targets | Needs baseline | Needs reference |
|---|---|---|---|
| `DIVERGENCE_FROM_BASELINE` | all | yes | no |
| `INTEGRATED_RESPONSE` | all | no | no |
| `MEAN_RESPONSE` | all | no | no |
| `PATTERN_SIMILARITY_TO_REFERENCE` | all | no | yes |
| `PEAK_RESPONSE` | all | no | no |
| `RESPONSE_STABILITY` | all | no | no |
| `TEMPORAL_CHANGE_MAGNITUDE` | all | no | no |

## DIVERGENCE_FROM_BASELINE

**Formula** — `sqrt(mean over finite t in W of (R_variant(t) - R_baseline(t))^2)`

**Raw output** — model response units

**Target types** — all target types

**Temporal support** — every temporal scope: full stimulus, absolute window, normalized window, content event, content-event-relative window.

**Direction support** — MAXIMIZE, MINIMIZE and TARGET. Direction is applied after normalization, never inside the metric.

**Interpretation** — How far this variant's predicted response departs from the baseline's over the window. Sign is discarded; the mean signed difference is reported separately in statistics.

**Limitations** — Divergence is not quality. A variant that differs most from the baseline is the most different, not the best, and the direction of the objective decides nothing about that. Requires equal-length windows in both variants.

## INTEGRATED_RESPONSE

**Formula** — `trapezoidal integral of R_target(t) dt over W`

**Raw output** — model response units x seconds

**Target types** — all target types

**Temporal support** — every temporal scope: full stimulus, absolute window, normalized window, content event, content-event-relative window.

**Direction support** — MAXIMIZE, MINIMIZE and TARGET. Direction is applied after normalization, never inside the metric.

**Interpretation** — Area under the response curve. Unlike the mean it grows with window length, so it distinguishes a brief spike from a sustained response.

**Limitations** — Because it scales with duration it is NOT comparable across windows of different length, which content-event scopes routinely produce. Trapezoidal integration assumes linear behaviour between samples; at TR=1s that is a coarse assumption.

## MEAN_RESPONSE

**Formula** — `mean over finite samples t in W of R_target(t)`

**Raw output** — model response units, the same units as the prediction

**Target types** — all target types

**Temporal support** — every temporal scope: full stimulus, absolute window, normalized window, content event, content-event-relative window.

**Direction support** — MAXIMIZE, MINIMIZE and TARGET. Direction is applied after normalization, never inside the metric.

**Interpretation** — The average predicted response of the selected target across the selected interval.

**Limitations** — A mean hides shape: a brief spike and a sustained plateau with the same average are indistinguishable. Use INTEGRATED_RESPONSE or PEAK_RESPONSE when shape matters. Signed responses can cancel.

## PATTERN_SIMILARITY_TO_REFERENCE

**Formula** — `cosine(mean_t in W of R_target(t, v), reference(v)) over jointly finite v`

**Raw output** — dimensionless, in [-1, 1]

**Target types** — all target types

**Temporal support** — every temporal scope: full stimulus, absolute window, normalized window, content event, content-event-relative window.

**Direction support** — MAXIMIZE, MINIMIZE and TARGET. Direction is applied after normalization, never inside the metric.

**Interpretation** — Directional similarity between the window's average spatial response pattern and a supplied reference pattern.

**Limitations** — Uncentred cosine is dominated by any shared offset, so two patterns can both score near 1 while differing substantially in structure. It measures direction, not magnitude. The reference is an input with its own provenance; the metric asserts nothing about the reference being desirable.

## PEAK_RESPONSE

**Formula** — `max over finite samples t in W of R_target(t)`

**Raw output** — model response units

**Target types** — all target types

**Temporal support** — every temporal scope: full stimulus, absolute window, normalized window, content event, content-event-relative window.

**Direction support** — MAXIMIZE, MINIMIZE and TARGET. Direction is applied after normalization, never inside the metric.

**Interpretation** — The single largest predicted response within the interval.

**Limitations** — An extremum, so it is determined by one sample and is the most outlier-sensitive metric here. At TR=1s a 5-second window holds 5 samples, and the metric warns below 5 finite samples.

## RESPONSE_STABILITY

**Formula** — `sample standard deviation (ddof=1) of R_target(t) over W`

**Raw output** — model response units

**Target types** — all target types

**Temporal support** — every temporal scope: full stimulus, absolute window, normalized window, content event, content-event-relative window.

**Direction support** — MAXIMIZE, MINIMIZE and TARGET. Direction is applied after normalization, never inside the metric.

**Interpretation** — Temporal variability of the predicted response. Pair it with direction MINIMIZE to prefer a steadier response.

**Limitations** — This is Temporal Response Variability and nothing else. It is not attention stability, engagement consistency, or any viewer state. A coefficient of variation is deliberately not offered: these responses are signed and cross zero, so dividing by the mean is undefined in the region that matters.

## TEMPORAL_CHANGE_MAGNITUDE

**Formula** — `mean of |R_target(t+1) - R_target(t)| over consecutive finite samples in W`

**Raw output** — model response units per sample

**Target types** — all target types

**Temporal support** — every temporal scope: full stimulus, absolute window, normalized window, content event, content-event-relative window.

**Direction support** — MAXIMIZE, MINIMIZE and TARGET. Direction is applied after normalization, never inside the metric.

**Interpretation** — How much the predicted response moves between samples. Larger means a more dynamic response over the interval.

**Limitations** — Per-sample, not per-second: at a fixed TR the two coincide, but the value is not comparable across runs with different TRs. Non-finite samples are dropped, so a difference can span a gap.
