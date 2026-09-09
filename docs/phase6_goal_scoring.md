# Phase 6 — Goal-Conditioned Neural Scoring

Phase 5 answers *"are A and B different?"*. Phase 6 answers *"given objective G,
what did A and B score?"*. The difference is that Phase 6 requires the user to
say what "better" means **before** any variant is measured.

## The chain

```
variant i
    R_i(t, v)                 predicted response, [time, vertex]
        |
        +-- target resolution      ROI / network / hemisphere / vertex set / whole cortex
        |
        +-- window resolution      per variant, on its OWN timeline
        |
        +-- metric                 one raw number, in model units
        |
      raw_value                    <-- the measurement
        |
        +-- normalization          across variants
        |
      normalized_value
        |
        +-- direction              MAXIMIZE / MINIMIZE / TARGET
        |
      score                        <-- the only rankable quantity
        |
        +-- weights                composite across objectives
        |
      total_score  ->  ranking, Pareto, sensitivity, contributions
```

## Objective model

`NeuralObjective` carries `objective_id`, `name`, `metric`, `target`,
`temporal_scope`, `direction`, `normalization`, `weight`, `parameters` and
`provenance`. It is data, hashable via `objective_set_hash()` for caching and
reproducibility, and excludes `provenance` from the hash so that two identically
defined objectives created at different times share an identity.

The persisted cache key additionally includes the ordered run IDs, declared
baseline, network-loading mode, and SHA-256 fingerprints of every prediction,
analytics, manifest, content, and functional-atlas input. Changing an input
therefore creates a new immutable score artifact instead of returning or
overwriting stale evidence. The objective-only definition hash is retained
separately for comparing objective definitions across experiments.

**Target types** — `WHOLE_CORTEX`, `ROI` (148 Destrieux regions), `NETWORK`
(Yeo 2011 7-network, only with a verified mapping), `HEMISPHERE`,
`CUSTOM_VERTEX_SET`. Medial-wall vertices are excluded from every vertex-based
target; a custom set consisting entirely of wall vertices is refused.

**Temporal scopes** — `FULL_STIMULUS`, `ABSOLUTE_TIME_WINDOW`,
`NORMALIZED_TIME_WINDOW` (fractions, for variants of different length),
`CONTENT_EVENT`, `CONTENT_EVENT_RELATIVE_WINDOW` (offsets anchored to the
event's **start**, so "2 s before to 3 s after" means the same thing regardless
of the event's own duration).

Windows are half-open `[start, end)`, matching Phases 2, 4 and 5, so a boundary
never assigns one sample to two adjacent windows.

## Why content-event scopes matter

Two variants can place the same event at different timestamps. An objective that
says "during the CTA" resolves to a **different absolute interval in each
variant**. Measured in the test suite: variant A's CTA at 20–25 s and variant B's
at 17–22 s each yield their own 5-sample window and their own correct mean. The
control test shows what a shared 20–25 s window would have done to B — measured
0.8 instead of 2.0, because most of that interval is after B's CTA has ended.

The resolved interval is persisted on every evaluation, because a score is
unreadable without knowing which seconds produced it.

## Sub-TR events: a real limitation, found on real data

Phase 4 detects events at sub-second resolution; predictions are sampled at the
TR (1.0 s on current runs). An event shorter than one TR can therefore contain
**no prediction sample at all**. Measured on a real run, the first
`speech_segment` spans `[0.031, 0.500)` and holds zero samples, which made the
whole objective unmeasurable for that variant even though a 1.9 s speech segment
followed immediately.

Default behaviour is to **fail loudly**, naming the sampling-period mismatch,
because silently sliding to the next occurrence would change what "first" means.
`skip_events_without_samples=True` opts into considering only occurrences that
contain a sample, and is off by default.

## Multi-objective scoring

Weights are rescaled to sum to 1 and the applied weights are persisted on each
contribution. A weight is a statement of what the user values and is never
inferred from the data — inferring it would mean the engine choosing its own
objective.

A variant missing a score on **any** objective gets no composite score at all,
rather than a partial sum. A partial sum would rank it against variants measured
on more objectives.

**Conflicts stay visible.** Per-objective scores, per-objective contributions and
contribution fractions are all retained. On the real two-run experiment, each
variant won one objective and neither dominated — reported as a two-member
Pareto front rather than hidden behind the composite.

## Effective weights: when a stated weight is not the real weight

A weighted sum only honours its weights when the objectives are on comparable
scales. Under `NONE` normalization each metric keeps its own units, so an
objective measured in unit-seconds (`INTEGRATED_RESPONSE`) swamps one measured in
units (`MEAN_RESPONSE`) no matter what the weights say.

Measured on a deliberate example: two objectives declared at **0.50 / 0.50**
behaved as **5% / 95%**, and the original engine emitted no warning at all.

Every result now carries `effective_weights` — the share of composite magnitude
each objective actually accounted for, averaged across variants. Two warnings
fire: one structural (several un-normalised objectives using different metrics
are being added in different units) and one empirical (an objective's realised
share drifted more than 0.15 from its stated weight).

This is reported rather than silently corrected, because the fix depends on
intent: normalise, reweight, or stop combining these particular objectives.

## Temporal contribution: which seconds produced the raw metric value

Every evaluation carries an optional `temporal_contribution` giving a per-sample
share of the raw value. The contract is strict and **verified at runtime**: the
shares must sum to the raw value, and a decomposition that does not add up is
discarded with a warning rather than reported as misattribution. That is what
makes "this second supplied 40% of the raw metric magnitude" arithmetic rather
than rhetoric. It does not attribute the normalized, direction-adjusted, or
composite score to time samples.

| Metric | Decomposition |
|---|---|
| `MEAN_RESPONSE` | each finite sample contributes `value / n` |
| `INTEGRATED_RESPONSE` | each trapezoid's area split half to each endpoint |
| `PEAK_RESPONSE` | the single argmax sample carries the whole value |
| `RESPONSE_STABILITY`, `TEMPORAL_CHANGE_MAGNITUDE` | none — not a sum over samples |
| `DIVERGENCE_FROM_BASELINE` | none — an RMS is not additive |
| `PATTERN_SIMILARITY_TO_REFERENCE` | none — a cosine of a window average is not additive |

Returning `None` is the honest answer for the last three. Splitting an RMS or a
maximum across samples would manufacture structure that is not in the
mathematics. Phase 5's signed-delta arrays remain the right place to ask where
two variants differ most.

## Loading variants, and networks

`scoring.loader.load_variant` assembles a variant from persisted Phase 1/3/4
artifacts, refusing stale analytics (a row count that disagrees with the
prediction) and non-dense region ids (`region_id` is used as a column index).

Phase 3 persists no functional-network series, so network objectives aggregate
at scoring time using **Phase 5's verified Yeo 2011 mapping** — checksum-pinned,
with atlas surface coordinates required to equal this project's mesh exactly.
Verified on the two real runs: all seven networks resolve and both variants
report the same mapping checksum.

## Regional contribution

An aggregate target (whole cortex, a hemisphere, a custom vertex set) is
decomposed into per-region shares, so a score can be traced to *where* on the
cortex it came from as well as *when*.

Shares are of the aggregate's **magnitude**, so a region contributing strongly
in the negative direction stays visible instead of cancelling silently against a
positive one. `vertex_counts` accompanies every region, because a large share
from a 12-vertex region is not the same finding as a large share from a
400-vertex one.

Two cases deliberately return nothing rather than a number:

- an **ROI target**, which is already one region — decomposing it restates the
  question;
- a **network target**, whose regions are Yeo networks — splitting those by
  Destrieux region would mix two atlases into one meaningless number.

Measured on the real 30 s run: 148 regions, shares summing to 1.0, led by
superior temporal sulcus and middle temporal gyrus.

This is contribution to a mathematical aggregate. It is not evidence that a
region is functionally responsible for anything, and the schema says so in a
`note` field.

## Pareto analysis

A variant is dominated when another is at least as good on every objective and
strictly better on at least one. The non-dominated set needs **no weights**, so
it embeds no opinion about tradeoffs, which makes it the honest answer when
objectives conflict. Variants lacking a score on any objective are excluded and
the exclusion is reported.

## Sensitivity analysis

One objective's weight is swept from 0 to 1 while the remainder is split among
the others **in their existing proportions**, so the sweep moves one dimension
rather than silently reshaping the whole weighting. Flip points are reported.

On the real experiment the ranking flips at weight 0.7 — that ranking is
weight-dependent, and saying so is more useful than presenting a winner.

## Ranking honesty

`ranking_margin` is first minus second. Below 0.01 the run emits a warning that
the ordering should not be treated as a meaningful difference. On the real
experiment the margin was 0.0036 and the warning fired.

Correct phrasing: *"Variant B ranked highest for the selected ROI-response
objective."* Not: *"Variant B is the better advertisement."*

## Performance

Measured on synthetic variants at the real shape (30 samples x 20,484 vertices
x 148 regions), recorded in `artifacts/benchmarks/phase6_scoring.json`:

Regenerate the machine-readable report with the deterministic seed and fixture:

```bash
python scripts/benchmark_scoring.py
```

| Variants | Objectives | Runtime | Peak memory | Result JSON |
|---|---|---|---|---|
| 2 | 1 | 2.8 ms | 4.9 MB | 6.1 KB |
| 2 | 3 | 15.3 ms | 14.6 MB | 18.9 KB |
| 10 | 1 | 9.8 ms | 5.0 MB | 21.6 KB |
| 10 | 3 | 54.5 ms | 14.6 MB | 58.0 KB |
| 50 | 1 | 45.9 ms | 5.3 MB | 99.8 KB |
| 50 | 3 | 268.4 ms | 15.1 MB | 255.1 KB |

Scoring is trivial next to inference: the 30 s TRIBE run that produced one of
these variants took **2 h 34 m**. Scoring 50 variants on 3 objectives takes a
third of a second. Runtime is linear in variants and dominated by whole-cortex
targets, which touch all 20,484 vertices; ROI targets read a precomputed series.

## Scientific limitations

- A score is the value of a chosen function of a **predicted** response. It is
  not persuasion, memory, attention, preference or conversion, and ranking
  language must never imply otherwise.
- Predictions are average-subject model output, not a measurement of any viewer.
- **No p-values, no confidence intervals.** The predictions are deterministic;
  there is no sampling distribution, so statistical significance is not
  available and is not invented (Step 37).
- At TR = 1.0 s a 5-second window contains 5 samples. Peak and variance over so
  few points are fragile, and every evaluation reports its sample count; peak
  warns below 5 finite samples and any window under 3 samples is flagged.
- `INTEGRATED_RESPONSE` scales with window length and is therefore **not**
  comparable across content-event windows of differing duration.
- Divergence from a baseline measures difference, not quality.
- A reference pattern is an input with its own provenance. There is no "ideal"
  cortical response and none is synthesised.
