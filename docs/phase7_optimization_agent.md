# Phase 7 — Neural Content Optimization Agent

Phase 6 answers *"how well did this variant satisfy objective G?"*. Phase 7
answers *"where did it fall short, what was on screen there, how did better
variants differ, and what is worth testing next?"*.

It produces **hypotheses**. Nothing here is a validated improvement.

## The pipeline

A controlled sequence, not an agent loop. Each stage has one job and a typed
output, so a bad recommendation can be traced to the stage that produced it.

```
objective + stored score
    -> gap analysis          how far the source sits from a reference
    -> reference selection   which variants to learn from, explicitly
    -> interval detection    where the objective was and was not carried
    -> evidence              measured per-interval differences, with ids
    -> generation            one atomic intervention per qualifying difference
    -> validation            grounding, constraints, causal language, actionability
    -> dedup + ranking       transparent priority, diversified by type
    -> hypotheses + traces   restated so Phase 8 can pass or fail them
```

There is no reasoning loop and no iteration limit to tune, because there is no
iteration. The whole pipeline is deterministic: the same inputs produce the same
evidence graph and the same ranking, asserted by a test. That is what makes
Phase 8's eventual pass/fail attributable to the recommendation rather than to a
sampling seed.

## Gap analysis

A single "gap" number means three different things, so the formula is computed
from the direction and **recorded on the result**:

| Direction | Formula | Reference |
|---|---|---|
| MAXIMIZE | `reference − source` | best-scoring variant |
| MINIMIZE | `source − reference` | best-scoring variant |
| TARGET | `abs(source − target)` | the target value, not a variant |

A source already at or ahead of its reference gets a note saying so rather than a
negative number presented as an opportunity.

## Weak and strong intervals

**Weakness is objective-relative.** It would be wrong to call an interval weak
because the response is low: a MINIMIZE objective wants a low value and a TARGET
objective wants a specific one. An interval is weak when it contributed
comparatively little *to the objective as defined*, measured from Phase 6's
per-sample contribution decomposition — whose entries sum exactly to the raw
value, which is what makes a contribution *share* meaningful.

- Weak: share below 25% of an even share.
- Strong: share above 2× an even share.
- Adjacent qualifying samples within 1.5 s are merged into one interval.

**Strong intervals matter as much as weak ones.** An optimizer that only knows
where a variant is weak will happily propose edits that destroy what already
works. Strong intervals are detected specifically so they can be marked
PRESERVE, and they flow through to the proposed variant spec.

**A refusal, not a guess.** Four of the seven Phase 6 metrics (stability,
temporal change, divergence, pattern similarity) are not sums over samples and
expose no temporal decomposition. For those, interval detection returns nothing
and says why, rather than inventing a split.

## Reference selection

Never implicit — a recommendation's whole meaning depends on what it is compared
against. `BEST_SCORE` (every variant outranking the source, best first),
`BASELINE`, `PARETO_FRONT`, `MANUAL`. The chosen ids are recorded on the result
and echoed on every evidence item derived from them. When no variant scores
higher, that is reported rather than silently producing nothing.

## Evidence

Phase 5 reports content-feature differences as a single mean over the whole
comparison. An optimizer needs the difference **during the interval that
underperformed**, which is a different number, so Phase 7 restricts Phase 4's
per-time feature matrices to the interval and computes the delta itself.

Every evidence item is addressable by id and records what was measured, where,
and in which variants — never an interpretation. `association` states the
epistemic status explicitly (`temporal_comparison` or `measured_value`) so a
reader cannot mistake a comparison for a cause.

`consistency` is the fraction of reference variants differing in the same
direction. One reference is an anecdote; three of three is a pattern.

**Overfitting guard.** Twenty-one features scanned across several intervals will
always turn up something. Only differences clearing 15% relative magnitude become
evidence, and the number of features examined is recorded on every run and
surfaced as a warning, so a reader can judge how much selection produced the
reported set.

## Recommendations

Deterministic rules map a specific measured feature difference onto a specific
intervention type. No language model is involved: an LLM could not tell a real
0.17 motion difference from a plausible-sounding one, so it must not be what
decides *what* to recommend.

**Atomic by default.** One intervention per recommendation, because Phase 8 has
to attribute a score change to a cause. A bundle of five edits that improves the
objective tells you almost nothing about which edit mattered. Bundles are offered
only under `EXPLORATORY`, carry `HIGH` risk, and warn that they cannot be
attributed individually.

## Validation — four gates

1. **Evidence grounding.** Every cited id must resolve. An ungrounded
   recommendation is indistinguishable from a guess.
2. **Constraints.** A frozen modality or preserved interval is a rule the user
   set, not a preference to trade away.
3. **Causal language.** Text asserting an outcome is rejected, not softened.
   Rewriting a causal claim into a hedged one would hide that the generator
   produced it; rejections are recorded with a reason so the failure is countable.
4. **Actionability.** No interval, no edit instruction, or a zero-length target
   means Phase 8 could not build it.

The language filter matches **verb forms** of causation, not the noun. That
distinction was found the hard way: the first version rejected the generator's
own disclaimer — *"not a demonstrated cause"* — because the pattern matched the
word. A filter that rejects the text keeping a recommendation honest is worse
than no filter.

## Evidence confidence

Explicitly **not** a success probability, and the schema says so in a `caveat`
field that the UI renders verbatim.

```
0.40 * reference_consistency
+ 0.25 * feature_difference_strength
+ 0.25 * objective_gap_strength
+ 0.10 * min(1, evidence_count / 4)
```

Halved when the interval holds fewer than two samples. Agreement across
references is weighted highest because it is the hardest thing to get by chance.

## Ranking

```
priority = evidence_confidence * information_value / cost_weight
```

`information_value` is 1.0 for an atomic intervention and 0.6 for a bundle,
because a bundle answers a vaguer question. `cost_weight` is 1.0 / 1.6 / 2.6 for
low / medium / high edit cost. Results are then diversified: one per intervention
type first, so a single strong feature cannot fill the list with variations on
itself.

Deduplication is **structural** — same intervention type over the same interval
is one test, whatever the titles say. That catches "move CTA earlier" and "show
CTA sooner" without needing embeddings.

## Human approval

Recommendations never become candidates on their own. A `ProposedVariantSpec` is
built only from `APPROVED` or `MODIFIED` reviews; attempting it without one is a
422. A modified review supplies its own interventions and the original
recommendation is preserved unchanged, so the two are never confused.

Reviews are stored separately from results and **carried across
re-optimization**, because a human decision is not derived data. That was a real
bug: the first implementation did `rmtree` then `replace` and destroyed every
approval on the next run.

## Persistence

```
artifacts/experiments/<experiment_id>/optimization/<request_key>/
    request.json  metadata.json  weak_intervals.json  evidence.json
    recommendations.json  hypotheses.json  traces.json
    reviews.json                 <- human decisions, preserved
    proposed_variants/*.json     <- Phase 8 input, preserved
```

Keyed by a hash of the request, so an identical request lands in one place.

## Performance

Measured at the real shape (30 samples × 20,484 vertices × 148 regions, 21
content features at 4 Hz). Recorded in
`artifacts/benchmarks/phase7_optimization.json`:

| Variants | Optimize | Peak memory | Evidence | Recommendations |
|---|---|---|---|---|
| 2 | 15.7 ms | 1.0 MB | 6 | 0 |
| 10 | 22.1 ms | 0.3 MB | 84 | 5 |
| 25 | 31.3 ms | 0.3 MB | 83 | 5 |

On the two **real** TRIBE runs the whole pipeline takes **7.3 ms**. There is no
LLM and therefore no token cost: the generator is deterministic.

## Three comparable variants now exist

The earlier limitation — "only two real variants, of different lengths and
different stimuli" — is closed. Two edited variants of the **same 10 s base
content** were produced (`sintel_10s_vB`: brighter, more saturated, louder;
`sintel_10s_vC`: darker, higher contrast, quieter) and put through the full
pipeline: TRIBE inference, analytics, content analysis.

All three pass Phase 5's comparability gate — **26 checks, 100% timeline
coverage, content comparable** — so this is a genuine A/B/N of the same content
rather than a comparison of unrelated stimuli.

Measured whole-cortex mean response:

| Variant | Edit | Score |
|---|---|---|
| C_dark | darker, higher contrast, quieter | **+0.04704** |
| A_orig | unmodified | −0.00358 |
| B_bright | brighter, more saturated, louder | −0.00737 |

With two references available, consistency is finally computable **and
discriminating**. Optimizing the lowest-ranked variant:

| Feature | Source | Reference mean | Consistency |
|---|---|---|---|
| brightness | +0.1376 | +0.0090 | 1.0 |
| motion | +0.0129 | +0.0061 | 1.0 |
| motion_structural | +0.0413 | +0.0213 | **0.5** |

The 0.5 entry is the point: the two references disagree on the direction of that
difference, and the recommendation resting on it ranks last (priority 0.242
against 0.285). Under the old behaviour every one of these would have read 1.0.

## Consistency needs something to be consistent across

`consistency` is the fraction of reference variants differing in the same
direction. It is **undefined below two references** and reported as `null`, with
`reference_count` always present beside it.

The first implementation returned 1.0 for a single reference — a reference
agreeing with itself — which made an anecdote indistinguishable from a
three-of-three pattern. The confidence formula now scores undefined consistency
as 0 rather than imputing it, which caps a single-reference recommendation at
**0.60**: the honest ceiling for evidence that could not be corroborated. The
formula string says so.

## One-sample intervals

At TR = 1 s a one-sample interval is one second of predicted response, and a
content change recommended from it rests on a single number. Both feature and
event evidence now require **two samples**; previously they disagreed, so a
one-sample interval could still produce an event-based recommendation while
producing no feature-based one. The interval itself carries a warning saying it
is below the threshold.

## Regional contribution

Phase 6 now decomposes an aggregate target into per-region shares, so a
recommendation can say *where* a gap sits rather than only *when*. Verified on
the real 30 s run: 148 Destrieux regions, shares summing to 1.0, led by superior
temporal and middle temporal regions.

Shares are of the aggregate's **magnitude**, so a strongly negative region is
visible rather than cancelling against a positive one, and `vertex_counts` is
reported so a large share from a small region is not read as a large effect. An
ROI target is not decomposed — it is already one region, and decomposing it
would restate the question. Network targets are not decomposed either, because
mixing the Yeo and Destrieux atlases would produce a meaningless split.

Regional evidence ranks regions by the size of the difference in their
contribution *share*, not by absolute response: a region large in both variants
explains none of the gap.

## Structural markers: HOOK and PRODUCT_REVEAL

Both now exist, and neither is invented.

**HOOK** is the first *measured* scene boundary, not an arbitrary number of
seconds. Its provenance states plainly that no detector for "a hook" exists and
that the label asserts position only, never rhetorical function.

**PRODUCT_REVEAL** is the first appearance of an object from the detector's
product vocabulary that persists at least 0.5 s. A stimulus with no such object
produces no marker and a warning, rather than a guess. On the real Sintel clip
a HOOK is placed at 0.00–4.17 s and no product reveal is emitted, which is
correct.

They live in `structural_markers`, **not** in `events`. That is deliberate:
`ContentAnalysisResult` validates that `events` forms a contiguous partition of
the stimulus so coverage arithmetic holds, and a marker overlaps whatever
interval it sits inside. The first implementation appended them to `events` and
the schema rejected it — the invariant caught the design error. Phase 7 merges
both lists when resolving a temporal scope, which needs matching by type and not
a partition.

## Stale feature arrays

A run directory accumulated one feature array per analysis — three for one run
here. That is not merely wasted space: the arrays differ in *column count*
across analysis versions, so a consumer selecting one by globbing rather than by
reading the manifest silently pairs a current feature-name list with a stale
matrix. That exact failure surfaced as an `IndexError` during Phase 7
development.

Arrays no longer referenced by the manifest are now pruned on write, after the
manifest is durable so a reader is never pointed at a deleted file. Verified: 3
files reduced to 1. `ContentSeries` additionally validates matrix width against
the supplied names, so a mismatched pairing fails with an explanation instead of
an index error deep in a loop.

## Scientific limitations

- **Recommendations are hypotheses.** Nothing is validated until Phase 8 builds
  the candidate, re-runs TRIBE and rescores it.
- **Evidence is temporal co-occurrence.** A reference variant differing in motion
  during an interval where it also scored higher is a comparison, not a cause.
- **TR = 1.0 s.** Intervals are second-granular, and every interval reports its
  sample count.
- **Feature scanning is selection.** 18–21 features per interval will surface
  differences by chance; the count is reported on every run.
- **A HOOK is a position, not a rhetorical function.** The marker says where the
  opening scene is; it makes no claim that the content there hooks anyone.
- **PRODUCT_REVEAL is bounded by the detector's vocabulary.** An unlisted product
  cannot be found, and absence of the marker is not evidence of absence.
