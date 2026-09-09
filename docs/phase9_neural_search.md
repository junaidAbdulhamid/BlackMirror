# Phase 9 — Automated Neural Search

**Status: in progress.** Built so far are the cache measurement, the candidate
materializer, the search foundations (Steps 1–7), candidate genomes with
decoding (Steps 9–10), fitness and the evaluator over Phase 8 (Steps 11–13),
seven search strategies (Steps 14–35, 48–50), the orchestrator with stopping,
tracking and checkpointing (Steps 39–47, 63), guardrails and feasibility
(Steps 61–62), Pareto search (Steps 58–60), the persistent evaluation cache
(Steps 36–38), populations and successive elimination (Steps 22, 32–35), the
scheduler and human override (Steps 51–53, 72), the artifact layout (Step 76),
the API (Step 75) and the interface (Steps 64–71). This document covers what exists and states plainly what does not.

The prior gate document is [`phase9_integration_report.md`](phase9_integration_report.md).

---

## The problem

Phase 8 gave the system an evaluation function. A variant goes in, a score on a
declared objective comes out:

    F(x) = G(TRIBE(x))

Phase 9 turns that into a search:

    x* = argmax F(x)   subject to   x ∈ X,  C(x) valid,  cost ≤ B

The systems fact that shapes everything is that **F is expensive**. A new
candidate for a ten-second clip costs roughly 52 minutes of inference, measured.
So the goal is never "find the highest score"; it is "find the strongest
observed candidate using as few expensive evaluations as possible".

---

## 1. The cache measurement

The question was whether a candidate that changes only audio could reuse the
cached video features, which plausibly dominate that 52 minutes. If so,
realistic budgets would be forty candidates rather than five.

**Answer: no.** From `neuralset.extractors`, the cache item identity is

```python
item_uid=lambda event: f"{event.study_relative_path()}_{event.offset:.2f}_{event.duration:.2f}"
```

and the on-disk cache confirms it, holding one entry per media filename. The key
is **path plus time range, with no content hash**. A candidate written to a new
path misses every extractor, including video, whatever it changed. Budgets must
therefore assume full cost per new candidate.

### The hazard this exposes

Because identity is the path, writing *different* content to a path that was
already processed returns the previous file's cached features. Every candidate
would be scored using the first candidate's features, silently, with no error
and plausible-looking numbers.

The materializer closes this by construction: each output is named after the
plan and the source content, so path identity equals content identity. Identical
plans share a path and legitimately share features; different plans can never
collide. The cache goes from a hazard to a correct optimisation.

---

## 2. Candidate materialization

`src/blackmirror/materialization/` builds the file Phase 8 needs. Phase 8
deliberately refuses to generate media, and that was right for a phase where a
human builds each variant; automated search cannot evaluate what it cannot
build, so this is the join.

**Edits.** Five parametric operations, each one filter with one numeric value:
`audio_gain_db`, `brightness`, `contrast`, `saturation`, `speed`. Every one is a
mechanical transform with recorded parameters. Nothing here can write speech,
render text or invent footage; `insert_speech` and `replace_text` from Phase 7
need generative models and are out of scope.

**Determinism.** ffmpeg is not reproducible by default, so every invocation pins
`-fflags +bitexact`, bitexact stream flags, strips container metadata and forces
single-threaded x264. Verified by test on both paths, audio-only and video
re-encode: the same plan yields byte-identical output.

**Controlled comparison.** Only the edited stream is re-encoded; the other is
copied. Verified on the real corpus: after an audio-gain edit the video stream
hash is unchanged, so the candidate differs from its parent in exactly the way
the plan declares and in no other way.

**Provenance.** A materialized file binds through a distinct Phase 8 adapter,
`ffmpeg-materialized`, with method `deterministic_ffmpeg_edit` and the tool
version recorded. Reusing the `user_supplied` adapter would record human
provenance for machine output.

---

## 3. Search foundations (Steps 1–7)

`src/blackmirror/search/`.

### Search space and parameters

Every parameter names the `EditOperation` it drives, so a space that cannot be
materialized fails at construction rather than hours later. Parameter bounds are
checked against what the operation accepts. Two parameters may not drive one
operation, because they would fight at materialization and ordering would
decide the winner.

Types are `CONTINUOUS`, `INTEGER`, `CATEGORICAL` and `BOOLEAN`. The vocabulary
also names `EVENT_TIMING`, `TEXT_CANDIDATE` and `STRUCTURED_INTERVENTION`, which
are **rejected** with a message naming what is missing, because a space that
accepted them would report a search over parameters that never moved.

Sampling is seeded and reproducible. Values quantize to a declared resolution so
near-duplicates collapse rather than costing an evaluation each. Decoding
refuses an out-of-range value instead of clamping it, so the recorded genome can
never disagree with the produced file.

### Budget

Limits: candidates, TRIBE runs, wall seconds, cost, generations. At least one
finite limit is required; an unbounded search over hour-long evaluations is not
a safe default.

There is no GPU-seconds field. TRIBE runs on CPU here, so such a field would
count something that never happens. Wall-clock is measured, and monetary cost is
derived from an operator-supplied rate and labelled an estimate.

The ledger is immutable and returns a new instance on every spend, so a crash
between spending and checkpointing cannot lose the record and cause a double
spend. A cache hit counts as an evaluated candidate but **not** as a TRIBE run;
conflating them would make the compute figures wrong in the flattering
direction.

### Hard admission

`admit()` returns how many of a batch may proceed, not a yes or no. Five
requested against two remaining admits exactly two. Verified by test that
repeated batches can never exceed the ceiling in total.

### Experiment and config

`SearchExperiment` holds the root, space, objectives, budget and ledger. It
deliberately holds no population or trajectory: those grow per evaluation and
belong in the checkpointed state beside it.

Configuration changes are versioned, keeping superseded revisions, so a report
can say which settings were in force when a candidate was proposed.

### The minimum-improvement default

`minimum_improvement` defaults to **0.0157**, not to zero.

That is the measured nuisance floor: swapping between two mirrors of nominally
identical Llama-3.2-3B weights shifted whole-cortex mean by that much on
average, while the real differences between three content variants were 0.0117
to 0.0241. A search reporting an improvement below it has found nothing
distinguishable from an implementation detail. Configuring a lower threshold is
allowed but produces a warning that the final report must carry.

See [`scientific_limitations.md`](scientific_limitations.md) §6a.

---

## 4. Candidate genomes and decoding (Steps 9-10)

### The genome

A point in the declared space plus its provenance: origin, generation, parents.
The biological word is borrowed and the analogy stops there; there is no
expression, no dominance, no fitness held inside it.

Identity is **canonical**, derived from the values quantized to each
parameter's resolution with names sorted, and deliberately excludes origin,
generation and parent. How a point was reached does not change what it is, so
two strategies arriving at the same point recognise it as one candidate and pay
for one evaluation. Quantization is what keeps a continuous strategy from
proposing an unbounded stream of points that differ below the resolution the
encoder can express.

Lineage is validated: a mutation must name its parent, a crossover needs two,
the root descends from nothing, and nothing is its own parent. Parents are a
tuple rather than a single field because crossover has two, which
`ProposedVariantSpec.parent_variant_id` cannot express.

`genome_distance` normalizes each ranged parameter by the width of its domain
and counts a categorical mismatch as one, then averages. A 3 dB move in a 12 dB
range therefore counts the same as a 0.05 move in a 0.2 range, and spaces of
different sizes stay comparable. It underpins neighbourhood generation and
diversity later.

### Decoding, and the evidence problem

`GenomeDecoder` produces the materialization plan and the
`ProposedVariantSpec`. Reusing the Phase 7 vocabulary rather than handing Phase
8 a bare file path is what keeps a search candidate the same kind of object as
a human-approved one, auditable by the same screens.

One thing could not be reused. `ContentIntervention` requires at least one
evidence id, and every Phase 7 evidence kind records a measured comparison
between variants. A search candidate has no measurement behind it: a strategy
chose a number from a declared range. Citing a measured kind would make a guess
indistinguishable from a finding.

So Phase 9 adds `EvidenceKind.SEARCH_SPACE_SAMPLE` and a third `association`
value, `not_measured`. Such an item carries the setting that was applied in
`source_value` and leaves every measurement-shaped field empty:
`reference_value`, `delta`, `consistency` are `None` and `reference_count` is
zero. The description states in words that the value was selected and nothing
has been observed about its effect, so a reader who never inspects the enum is
not misled either.

The expected direction is taken from the objective's own direction, so a
candidate can never contradict the objective it is aimed at, which Phase 8's
request schema would reject anyway.

Constraints are checked at decode time, before anything is built: a candidate
editing a frozen modality is refused rather than materialized, evaluated, and
only then found inadmissible after an hour of inference. Duration is the
exception, checked against the produced file, because a speed change's exact
effect on container length is not known until it is written.

---

## 5. Fitness, the evaluator, and random search (Steps 11-15, 48-50)

### Fitness is built from raw values

Phase 6 reports raw and normalized, and leads with raw because normalization can
turn a negligible raw difference into a decisive-looking score. A search driven
by the normalized number would chase exactly that amplification. The nuisance
floor is also expressed in raw units, so comparing raw keeps the margin and the
floor in the same currency. The normalized score is carried alongside for
display.

Direction is folded in so higher is always better: a MINIMIZE objective's value
is negated, and a TARGET objective's becomes the negative distance from its
declared target. Every strategy can then take a maximum without knowing what
the objective asked for.

A missing value stays `None` rather than becoming a sentinel, because a
stand-in number would be ranked against real ones and could win.

### Best is the argmax, and ties are reported

This was got wrong first and corrected after running the loop. Promotion was
originally gated on the improvement clearing the noise floor, which made the
recorded best depend on the order candidates happened to be evaluated in: a
stronger candidate arriving second failed to displace a weaker first one, and a
different seed would report a different winner among identical numbers.

The best is now the plain argmax, which is order-independent. Whether its lead
means anything is a separate question, asked by `beats` for stopping decisions
and answered in reports by `tied_with_best`, which lists every candidate the
winner is not measurably ahead of. A run on the demonstration surface reports:

    BEST OBSERVED : c0006  fitness 0.2990
      vs root     : +0.0490  (threshold 0.0157) -> MEANINGFUL
      NOT measurably ahead of 2: c0002@0.2910, c0004@0.2960
      -> the ranking among these is not a finding

Naming a single winner without that second line would overstate what the search
established.

### The evaluator

`CandidateEvaluator` runs genome to spec and plan, materializes the file, builds
the Phase 8 request and hands it to the orchestrator. It re-implements no part
of the pipeline: a second scoring path would eventually disagree with the first
and there would be no way to tell which number was real.

`resimulation.request_builder` is not reused, because it derives the
hypothesis-to-objective map from a stored Phase 7 optimization and a search
candidate has none. The maps come from the decoder instead, and
`ResimulationRequest`'s own validators still enforce every consistency rule.
`optimization_request_key` carries `search-<search_id>`; what actually marks a
candidate as machine-proposed is the binding's adapter id,
`ffmpeg-materialized`, which no human-supplied variant carries.

Every failure path returns a fitness carrying its reason rather than raising
(Step 54). A constraint violation or a neutral genome is rejected before any
compute; a pipeline exception is recorded with the time it consumed. One bad
candidate cannot end a search that has already spent hours.

### Random search

The baseline. It holds no model of the objective and learns nothing from
results, which is the point: it measures how much of any other strategy's
performance comes from the shape of the space rather than from the strategy. On
a space where the signal sits below the noise floor it will match anything
cleverer, and that is a result worth being able to see.

It never proposes the same point twice, never proposes the unedited root, and
reports exhaustion rather than looping when a small space runs out. Its
generator state is exported with `get_state`, so a search interrupted at
evaluation seven resumes proposing the eighth candidate it would have proposed
rather than restarting the stream and paying for earlier points again.

The registry advertises only strategies that exist. Asking for one that is not
implemented raises rather than returning something that silently does nothing.

---

## 6. The remaining strategies (Steps 16-35)

Seven strategies now exist, all behind one interface. Each is documented with
its strengths, failure modes and parameters in
[`search_algorithms.md`](search_algorithms.md); measured results are in
[`search_benchmarks.md`](search_benchmarks.md).

Shared machinery came first. `MutationRegistry` keys an operator to each
parameter type, with magnitude expressed as a fraction of the parameter's own
range so that "small" means the same thing on a 12 dB knob and a 0.4 one. A
mutation always changes something: a child identical to its parent would be a
duplicate consuming an evaluation slot. `neighbours` moves one parameter at a
time, which is what makes a local move attributable. `crossover` is uniform and
takes every value from a parent, so a child can only sit at a combination its
parents already justified.

Two bugs surfaced while testing this and were fixed rather than worked around.
Tournament selection could return the same candidate twice, which produced a
crossover child listing one parent on both sides and failed the genome's own
lineage validation. And local search originally stopped when a neighbourhood was
exhausted, which made it indistinguishable from hill climbing; it now restarts
from a fresh point, which is the difference between the two.

### What the benchmark showed

Local search was strongest on both synthetic surfaces. Hill climbing was worst
on both and demonstrably trapped on the two-peak one, ending at 2.224 against a
true optimum of 6.0 while using four of its forty evaluations. Evolutionary
search underperformed random sampling on that surface at this budget, which is
recorded rather than smoothed over: populations need generations, and
generations cost evaluations this pipeline does not have cheaply.

---

## 7. The orchestrator (Steps 39-47, 63)

The loop is: propose, drop duplicates, admit against budget, evaluate, record,
let the strategy observe, checkpoint. The module contains no search logic, no
budget arithmetic and no scoring; what it owns is the order, and the guarantee
that budget is checked before work is scheduled rather than after.

**Checkpointing after every evaluation**, not every generation, because an
evaluation is 52 minutes and losing a generation to a crash is losing most of a
day. Resume replays completed candidates into the tracker rather than
re-evaluating them.

**A failed candidate is data.** It is recorded, counted against the budget it
consumed, and the loop continues. A search that died because one ffmpeg call
failed would waste every hour spent before it.

**A stop takes effect at the next candidate**, not the next generation, so a
user who asks to stop is not made to wait another hour.

**Stopping reasons are ordered.** A user stop outranks everything, then hard
budget limits, then a target being reached, then patience. A search that hit its
budget and its patience in the same round reports the budget, because that is
the binding constraint.

**Patience counts only believable gains.** Counting any gain at all would keep a
search alive indefinitely on drift smaller than the pipeline's own sensitivity,
spending real hours chasing noise.

The report names the best observed candidate, what it cost to find, and the
caveats: that this is not a global optimum, whether the improvement over the
root clears the measured floor, and which candidates the winner is not
measurably ahead of.

Concurrency is deliberately not used. `max_concurrent_candidates` exists in the
config, but the loop evaluates sequentially, because inference is memory-bound
on this machine and parallel evaluations would contend for the same weights and
finish later in total.

---

## 8. Guardrails and feasibility (Steps 61-62)

Fitness answers how a candidate did on the objective. Feasibility answers
whether it is allowed to win. Keeping them apart matters: a candidate can score
highest and still be unusable because it ran long, breached a secondary measure
the user protected, or edited something locked.

A guardrail is not a penalty term. A penalty trades off, so enough gain on the
primary objective eventually buys a violation. A guardrail does not: if
stability must not fall below a bound, a candidate that breaches it is out
regardless of its score, because that is what "must not" means.

Four kinds exist: an objective floor, an objective ceiling, a maximum relative
drop against the root, and a maximum duration change. Constraints a genome can
violate on its own are caught earlier by the decoder; these are checked after
evaluation because they depend on measurements that do not exist until the
candidate has run.

A guardrail whose objective was not measured is reported as **unchecked**, not
silently passed. Silently passing would let a guardrail protect nothing while
appearing to protect something. Treating unchecked as disqualifying is available
and off by default.

The veto is enforced where it cannot be forgotten: `best_of` selects the highest
*eligible* candidate, not the highest-scoring one. A disqualified leader is
still recorded and still surfaced, because "the top score broke a rule" is
something a user needs told rather than a fact quietly dropped.

---

## 9. Multi-objective search (Steps 58-60)

Weighting several objectives into one number asserts an exchange rate, and
Phase 6 already showed how easily a ranking flips within the range of weights
someone might plausibly choose. The Pareto frontier needs no such assertion: it
removes only candidates beaten on every objective at once and leaves the
trade-off to a person.

Dominance is computed on **directional fitness**, each objective folded so
higher is better, rather than on normalized scores. Normalization can amplify a
negligible raw difference, and a frontier built on amplified noise would present
false trade-offs.

`non_dominated_sort` partitions candidates into successive frontiers, which is
the ranking half of NSGA-II and is tested against a known answer. Crowding
distance and the rest of NSGA-II's machinery are **not** implemented: at budgets
of a handful of candidates it would be elaborate scaffolding around too few
points to mean anything. Within a front nothing dominates anything, so ties
break by candidate id, which is arbitrary but deterministic. Inventing a
preference inside a front would smuggle back the exchange rate the frontier
exists to avoid.

---

## 10. Caching, populations and scheduling

**The evaluation cache** (Steps 36-38) keys on the genome *and* everything else
that could change the number: the root media hash, the objective definition, and
the versions of the space, materializer, scoring engine and model. Leaving any
out would return a number computed under different rules, which is worse than no
cache. It persists, so a second search over the same root pays nothing for
points already evaluated. Failures are deliberately not cached: a transient
failure stored forever becomes a permanent one.

**Successive halving** (Step 22) eliminates between generations on full
evaluations. It does **not** fabricate a low-fidelity score. There is no
shortened TRIBE pass whose ranking has been shown to agree with a full one, so
inventing one would produce confident eliminations based on a number nobody
validated. What halving buys is that the next generation's budget goes to the
survivors, not that any evaluation became cheaper.

**Diversity** (Steps 34-35) is measured as mean pairwise genome distance,
normalized per parameter by its own domain width. It returns null below two
members, where it is undefined, rather than zero, which would read as "fully
collapsed". Novelty only breaks ties unless the caller raises its weight
deliberately, because a population selected mainly for being different stops
optimising.

**The scheduler** (Steps 51-53) orders candidates by an explainable priority
rather than an opaque score: pinned first, then cached, then descendants of
retained candidates, then generation, then proposal order. Each carries the
sentence that explains its position. Cached candidates go early because they
cost nothing and can raise the bar everything else is measured against.

Concurrency is honoured and defaults to one worker. On this machine inference is
memory-bound, so parallel evaluations contend for the same weights and finish
later in total. Concurrent runs restore priority order in their results, so
changing the worker count does not change the record.

---

## 11. API and interface

Routes under `/api/search` list searches and serve state, best, trajectory,
population, budget, events, results, Pareto frontier and the final report, plus
control routes for pause, resume, stop, pin and eliminate. Starting a search
queues it on a single background worker and returns immediately; a search is a
sequence of Phase 8 evaluations and inherits their cost.

One worker globally, not one per search: two searches at once would each be
running TRIBE on a memory-bound machine, and the wall-clock figures in each
ledger would reflect contention with an unrelated run rather than the search
itself.

The `/search` screen is arranged around the claim it has to make honestly. The
headline is the best **observed** candidate, and the line beneath says whether
its lead over the root clears the measured noise floor. When it does not, that
is the first thing a reader sees. Candidates the winner is not measurably ahead
of are named. The tree shows every candidate with its standing, and an
eliminated one carries its reason rather than leaving absence to be interpreted.

The trajectory plot draws individual evaluations as well as the running best,
because a best-so-far line alone hides how much of the budget was spent far from
it. It toggles between evaluations and compute on the x-axis, which is Step 68's
question: how much did this cost.

---

## 12. What is not built

A learned surrogate model, Bayesian optimization over one, reinforcement
learning, preference learning and anything that deploys a winner: all deferred
by Step 95 and none of them started.

Multi-fidelity evaluation has an interface but no implementations, and
deliberately so. There is no validated low-cost approximation of a TRIBE pass,
and adding one without evaluating its agreement with the full pass would make
every elimination it drove untrustworthy.

---

## 13. Measured on the real pipeline

Random search against local search, three candidates each, identical root,
objective, space, seed and budget, on the four-second clip.

| strategy | best | vs root | TRIBE runs | wall clock |
| --- | --- | --- | --- | --- |
| random | -0.023349 | -0.000198 | 3 | 69.2 min |
| local | -0.025313 | -0.002162 | 2 | 45.8 min |

Root: -0.023151. Neither beat the root, neither beat the other, and every
difference was far below the 0.0157 nuisance floor. Both are reported with
`improvement_is_resolvable = false`.

The shared evaluation cache saved a full inference: both strategies seed
identically, so local search's first proposal was already evaluated and came
back in zero minutes, leaving it with two inference runs against three and a 34%
saving in wall clock.

Full detail and the reasons this is not evidence about either strategy are in
[`search_benchmarks.md`](search_benchmarks.md).

---

## 14. Limitations

- The search space is restricted to five parametric edits. It cannot express
  most of what Phase 7 can recommend.
- A new candidate costs roughly 52 minutes. Budgets in single digits are the
  realistic case, and a twenty-five candidate search is a full day of compute.
- The nuisance floor was measured with one variable on three variants of one
  clip. It is a lower bound on the pipeline's sensitivity, not a complete error
  budget.
- The real benchmark found nothing. Six candidates over a four-second clip,
  three per strategy, produced a spread of 0.004 against a nuisance floor of
  0.0157, and neither strategy beat the root or the other by a resolvable
  margin. The machinery works end to end at real cost; the search space and
  budget were too small to produce a finding, and the report says so rather
  than presenting the winner as one.
- Strategy comparison on synthetic surfaces does not transfer. Those are
  smooth, noiseless and two-dimensional, and they do not simulate the nuisance
  floor at all. The real comparison has now been run and was inconclusive, as
  recorded in `search_benchmarks.md`.
- Evaluations run sequentially. Concurrency is exposed in the config and
  ignored by the loop, because it would make this machine slower.
