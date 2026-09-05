# Scoring Normalization

Three numbers are kept separate on every evaluation, and conflating them is the
single easiest way for a scoring engine to mislead.

| Quantity | What it is | Depends on other variants? |
|---|---|---|
| `raw_value` | The metric in model units, e.g. mean ROI response `0.381` | No |
| `normalized_value` | That value placed on a comparable scale | Usually **yes** |
| `score` | The normalized value after direction is applied | Yes |

Only `score` may be ranked. Only `raw_value` is a measurement. The middle one is
a presentation choice.

## Why min–max can lie

Min–max maps the lowest variant to 0 and the highest to 1 **by construction**,
regardless of how far apart they actually are. Given

    A = 0.401
    B = 0.402

it reports

    A = 0.0
    B = 1.0

A 0.25% difference is rendered as total dominance. The arithmetic is identical
whether the spread is 0.001 or 10, so the normalizer cannot detect the problem
from its own output.

Three things are done about this:

1. **`NONE` is the default.** A raw metric already lives on a meaningful scale.
   Normalizing is a decision, and the engine does not make it silently.
2. **The raw value is always preserved** and travels with the score.
3. **A negligible-spread warning fires** when the range is under 1% of the
   values' magnitude. On the example above the run emits: *"min-max stretched a
   spread of 0.001 across values of magnitude ~0.402 (0.25%) to the full 0–1
   range. The normalized scores will look decisive; the raw values are nearly
   identical. Read the raw values."*

There is a test asserting exactly this, including that A really does score 0.0
and B 1.0 — the warning exists because the arithmetic is *not* wrong, only
misleading.

## Strategies

| Strategy | Definition | When it is appropriate | How it misleads |
|---|---|---|---|
| `NONE` | score = raw | Default. Raw units are already meaningful | Values from different metrics are not comparable, so weighted sums across metrics mix units |
| `MIN_MAX_WITHIN_EXPERIMENT` | `(v − min) / (max − min)` | Presenting several variants on one axis | Exaggerates tiny spreads; the best and worst are always 0 and 1 |
| `Z_SCORE_WITHIN_EXPERIMENT` | `(v − mean) / sd`, population sd | Many variants, roughly symmetric spread | Meaningless below ~3 variants: with two it always returns −1 and +1, and the run warns |
| `REFERENCE_BASELINE` | `(v − base) / abs(base)` | A declared original exists | Undefined near a zero baseline; the engine then reports the plain difference and says so |
| `ROBUST_PERCENTILE` | clipped `(v − p10) / (p90 − p10)` | Outliers present | Clipping hides how extreme an outlier was |
| `TARGET_DISTANCE` | `max(0, 1 − abs(v − target) / tolerance)` | A specific level is wanted | The tolerance sets the whole shape and must be chosen deliberately |

## Direction is applied last

Normalization answers *"where does this sit among the variants?"*. Direction
answers *"is higher preferable?"*. They are separate steps so that a `MINIMIZE`
objective is never implemented by negating a metric somewhere inside the metric
itself, where the sign could be lost.

`MINIMIZE` negates rather than inverting. A reciprocal would distort the spacing
between variants and divide by zero at 0; negation preserves spacing exactly.

## Baseline-relative reporting

With a declared baseline, each variant also carries `baseline_delta` and
`baseline_relative_delta`. The relative figure is **withheld** — set to `None`
rather than printed — when the baseline total is at or near zero, because a
percentage against ~0 is not a number anyone should read.
