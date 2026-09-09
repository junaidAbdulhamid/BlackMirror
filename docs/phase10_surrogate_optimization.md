# Phase 10 — Surrogate Modelling and Bayesian Optimization

A cheap learned approximation of the expensive objective, used to decide which
candidates deserve a real evaluation.

The gate document is
[`phase10_integration_report.md`](phase10_integration_report.md). The technique
is explained in [`bayesian_optimization.md`](bayesian_optimization.md), and how
the model is judged in [`surrogate_evaluation.md`](surrogate_evaluation.md).

---

## The problem

The real objective is

    F(x) = G(TRIBE(x))

and costs roughly 23 minutes per candidate on a four-second clip, 52 on a
ten-second one. With ten evaluations available, the question is not "which
candidate is best" but "which ten are worth measuring".

A surrogate learns

    F_hat(x) ≈ F(x)

from candidates already evaluated, scores thousands of untested ones in
milliseconds, and an acquisition function turns those cheap scores into a
decision about where to spend the next expensive one.

**It never replaces the real evaluator.** It is a screening model. Every number
it produces is an estimate, labelled as one in the types, the API and the
interface, and no candidate is reported as good until the real pipeline has
measured it.

---

## 1. The training dataset

Only real evaluated candidates become rows. A model trained on its own
predictions learns its own errors and reports them back with growing confidence,
so `SurrogateDataset` is built through a gate that admits a `CandidateFitness`
only if it produced a real measured value, and records why anything else was
excluded.

**Identity is wide on purpose.** Two labels are comparable only if the root
media, the objective definition, the search space, the materializer, the scorer
and the model fingerprint all match. `DatasetIdentity` composes exactly that
set, reusing the idea already proven in Phase 9's `CacheIdentity` rather than
inventing a parallel scheme. A dataset mixing two identities would be learning a
mixture of two functions and reporting one number, with nothing in the output
saying so.

`dataset_hash()` sorts by candidate id before hashing, so the same observations
in a different order produce the same hash while changing any single label
changes it.

## 2. Feature encoding

`GenomeFeatureEncoder` derives its column layout from the declared search space
once, in sorted name order, and checks it on every encode. A model trained with
`gain` in column 0 and asked to predict with `brightness` there returns a
confident, meaningless number, and nothing would raise.

**Scaling is by declared domain, not fitted statistics.** Each parameter maps
onto [0, 1] using its own bounds. That needs no data, is exactly reproducible at
any sample size, and makes a knob measured in decibels contribute comparably to
one measured in arbitrary units. Standard scaling is available for callers with
enough samples to estimate a variance, and refuses below two genomes rather than
dividing by an unstable estimate.

**Categoricals are one-hot and nothing more.** Step 4 asks about embeddings for
high-cardinality choices and Step 5 about text. The search space cannot express
either: text and structured-intervention parameters are rejected at construction
because nothing can materialize them, and the categorical parameters that exist
hold a handful of numeric choices. Embedding machinery would be scaffolding
around a feature type the system cannot produce.

## 3. Surrogate models

| family | role | uncertainty |
| --- | --- | --- |
| **Gaussian Process** (Matérn 2.5) | **default** | posterior standard deviation |
| Random Forest | baseline | ensemble spread |
| Extra Trees | baseline | ensemble spread |
| Gradient Boosting | available | none |

**The default is a Gaussian Process** because the regime demands it: two to five
dimensions, deterministic observations, tiny sample counts. That is where GPs
are strongest and tree ensembles weakest. §8 measures the real space at *one*
effective dimension rather than two **under a whole-cortex objective**, which
strengthens this choice rather than weakening it — but the figure quoted here
was the declared one, not a measured one, until that measurement was made, and
the measured one turns out to depend on which objective is asking.

**The two uncertainties are different objects.** A GP posterior grows with
distance from observations. A forest's spread is disagreement among correlated
learners, and it *collapses* beyond the training range, exactly where confidence
should be lowest. Every prediction carries `uncertainty_kind` so the two cannot
be confused, and the acquisition registry records which functions assume a real
Gaussian.

**Targets are centred before fitting.** A GP's prior mean is zero and objective
values here sit near -0.023; without centring the posterior is pulled toward
zero everywhere and the model spends its capacity representing an offset.

## 4. Acquisition functions

    UCB(x) = mu + kappa*sigma
    PI(x)  = Phi(z)
    EI(x)  = (mu - f* - xi)*Phi(z) + sigma*phi(z),   z = (mu - f* - xi)/sigma

All four are implemented and verified against closed forms rather than golden
values: UCB with mu 0.5, sigma 0.1 and kappa 2 gives exactly 0.7, and EI at the
incumbent equals sigma times the normal density at zero.

`GREEDY_MEAN` exists as the baseline the others must beat. It is pure
exploitation and will re-sample one region forever, which is the failure the
whole technique is built to avoid.

An acquisition that needs an uncertainty **raises rather than substituting
zero**, because a silent substitution would turn EI into a broken greedy. When
the configured acquisition cannot be computed, the loop falls back to greedy and
labels the round accordingly.

## 5. Candidate pool

Several thousand genomes are sampled per round, mixing uniform draws with
perturbations around the incumbents. Sampling rather than optimizing the
acquisition directly is the right choice at two to five dimensions — and §8
measures the real space at one effective dimension under the objective tested,
where it is more clearly right still. It also avoids an inner optimizer
converging on a sharp artefact of a model fitted to a handful of points.

**Nothing is materialized.** A pool member is a genome; no file is written and
no inference runs until the acquisition has chosen. That is the entire economy
of the phase.

Pool ids are namespaced by round. An incumbent from an earlier pool carries a
pool id too, and a bare counter would eventually reissue it, making a genome its
own parent, which Phase 9's lineage validation correctly refuses.

## 6. The loop

    bootstrap -> train -> pool -> predict -> acquire -> evaluate -> record -> retrain

**Bootstrap first.** Nothing can be fitted to zero observations, and a model
fitted to three points should not be choosing anything. The first several
evaluations are spent covering the space.

**The trust gate decides who chooses.** Below a demonstrated rank correlation on
order-aware validation, the surrogate does not select and a Phase 9 strategy
does. Each round records its mode, so a run is auditable afterwards.

**Budget is Phase 9's.** Surrogate predictions cost nothing and are charged
nothing; `BudgetLedger` already separates a cache hit from an inference run, and
`admit()` already enforces that screening ten thousand candidates cannot
schedule more real evaluations than remain.

**Every round records predicted against actual**, with the error. That is the
diagnostic that says whether to believe the model, and it is what the interface
plots against the identity line.

## 7. Measured on synthetic surfaces

Where signal is known to exist and the optimum is known, so the claim is
falsifiable. Two-dimensional space, 24 evaluations, 6 seeds, 6 bootstrap.

Command:

    python scripts/benchmark_surrogate.py --budget 24 --seeds 6 --bootstrap 6

Space: gain in [-10, 10] and brightness in [-1, 1]. The threshold is what random
search reaches on the full budget, so the question is how quickly each method
gets to a bar random sampling eventually clears. Runs that never reach it count
at budget + 1, because a median taken only over successful runs rewards a method
that succeeds rarely but occasionally early.

### one_peak — a single smooth maximum at (3, 0.4), optimum 0.0

| method | best@5 | best@10 | best@20 | evals to threshold | reached |
| --- | --- | --- | --- | --- | --- |
| random | -6.246 | -1.208 | -1.008 | 20 | 3/6 |
| beam | -3.668 | -1.235 | -0.392 | 14 | 5/6 |
| **bayesian** | -6.246 | **-0.319** | **-0.001** | **10** | **6/6** |

### two_peaks — a broad low peak and a tall narrow one, optimum 6.0

| method | best@5 | best@10 | best@20 | evals to threshold | reached |
| --- | --- | --- | --- | --- | --- |
| random | 3.806 | 5.320 | 5.412 | 17 | 3/6 |
| beam | 3.806 | 5.320 | 5.495 | 14 | 5/6 |
| **bayesian** | 3.806 | **5.727** | **5.823** | **9** | 5/6 |

### Sample efficiency (Step 76)

Reaching the same bar, measured against random search:

| surface | random | bayesian | fewer evaluations |
| --- | --- | --- | --- |
| one_peak | 20 | 10 | **50%** |
| two_peaks | 17 | 9 | **47%** |

At roughly 23 minutes per real evaluation, halving the count is about four hours
saved per search of this size.

### Reading these honestly

**Bayesian wins on every metric here**, including reliability: it reached the
threshold in six runs of six on the first surface where random managed three.

**It is behind at five evaluations, deliberately.** The first six go to
bootstrap, so at `best@5` it is identical to random by construction. The
advantage appears once the model starts choosing, which is exactly the trade
Step 30 describes.

**Beam search is the honest middle.** A Phase 9 strategy with no model beats
random comfortably and loses to the surrogate, which is what "the surrogate adds
something over a good heuristic" should look like when it is true.

**None of this transfers automatically.** These surfaces are smooth, noiseless
and two-dimensional, and evaluation is free. What they establish is that the
mechanism works and is implemented correctly, not that it will help on a
predicted cortical response.

## 8. Measured on the real corpus

The integration report identified one blocker: the labels available to a
surrogate spanned 0.004, a quarter of the 0.0157 nuisance floor. Its own
recommendation was to try the cheaper of the two fixes first — widen the search
space and see whether the effects clear the floor. That measurement has now been
made.

Command:

    python scripts/measure_space_range.py --gain 15 --brightness 0.35

Four corners of a space more than twice the width of the current one
(±15 dB against ±6, ±0.35 brightness against ±0.15), each a real materialization
scored through the real pipeline. About 25 minutes per corner, 94 minutes total.

| gain (dB) | brightness | whole-cortex mean | distance from root |
| --- | --- | --- | --- |
| — (root, neutral) | — | **-0.023151** | — |
| -15 | -0.35 | -0.054594 | 0.031443 |
| +15 | -0.35 | -0.055032 | 0.031882 |
| -15 | +0.35 | -0.039132 | 0.015981 |
| +15 | +0.35 | -0.039260 | 0.016109 |

### The verdict: resolvable, and by a margin that matters only once the root is counted

| range | value | against the 0.0157 floor |
| --- | --- | --- |
| across the four corners | 0.015901 | **1.01x** |
| including the root | 0.031882 | **2.03x** |

The script reports the first number and calls the space resolvable. That is true
and it is nearly worthless: clearing the floor by one percent is not a margin,
it is a coin flip dressed as a result.

The second number is the one to read, and the reason is not presentational. The
root is the neutral genome — a legitimate point in this space, and the baseline
every candidate is scored against. Excluding it measures the range across the
*edges* of the space while ignoring its middle. Counted properly the space spans
twice the floor, and **the widening worked**: this space can produce differences
the pipeline can resolve, where the current one could not.

### The finding that matters more: audio gain is invisible to this objective

Reading the corners as a factorial rather than as a spread:

| effect | holding | change in whole-cortex mean | against the floor |
| --- | --- | --- | --- |
| gain, -15 -> +15 | brightness -0.35 | 0.000439 | 0.03x |
| gain, -15 -> +15 | brightness +0.35 | 0.000128 | 0.01x |
| brightness, -0.35 -> +0.35 | gain -15 | 0.015462 | 0.98x |
| brightness, -0.35 -> +0.35 | gain +15 | 0.015772 | 1.00x |

A 30 dB swing in loudness moves the predicted whole-cortex mean by one to three
percent of the floor. Brightness moves it by ninety-eight to a hundred. The two
gain effects agree with each other, and the two brightness effects agree with
each other, so this is consistent structure rather than one odd corner.

**The obvious conclusion — that loudness does nothing — is wrong**, and the
check that shows it is cheap, because the predictions were already computed:

    python scripts/measure_roi_effects.py --contrast NAME=RUN_A,RUN_B ...

Artifact: `artifacts/benchmarks/roi_effects.json`.

| contrast | whole-cortex mean | mean per-vertex \|change\| | ratio | largest ROI effect |
| --- | --- | --- | --- | --- |
| gain, at brightness +0.35 | 0.000128 | 0.005410 | **42.2x** | 0.026018 |
| gain, at brightness -0.35 | 0.000439 | 0.003406 | **7.8x** | 0.015121 |
| brightness, at gain +15 | 0.015772 | 0.037586 | 2.4x | 0.189144 |
| brightness, at gain -15 | 0.015462 | 0.036973 | 2.4x | 0.189915 |

The last column is the point. **Gain moves individual vertices by up to forty
times what it moves the whole-cortex mean.** The prediction changes; the
average over 20,484 vertices cancels it. Brightness, by contrast, moves the
cortex coherently — a ratio of 2.4 rather than 42 — which is why it survives
averaging.

So the parameter is not inert. **The objective is blind to it.** Those are
different problems with different fixes, and the first one would have led to
discarding a working parameter.

### Where each parameter lands, and how much of that to believe

Largest regional effects, Destrieux 2009, regions of at least 20 vertices:

| brightness (both gain levels agree) | gain (both brightness levels) |
| --- | --- |
| S_oc-temp_med_and_Lingual — lingual | S_occipital_ant |
| G_oc-temp_lat-fusifor — fusiform | S_temporal_transverse — Heschl's |
| S_calcarine — primary visual cortex | G_temp_sup-G_T_transv — Heschl's |
| Pole_occipital | G_temp_sup-Lateral, S_temporal_sup |

**Brightness is clean.** Its four largest regions are lingual, fusiform,
calcarine and occipital pole: primary and early visual cortex, in both
hemispheres, essentially identical at both gain levels, at effects up to 0.19 —
twelve times the whole-cortex floor. A brightness manipulation moving visual
cortex is the result one would predict, and it is what appeared.

**Gain is suggestive and not clean, and the difference matters.** The transverse
temporal gyrus and sulcus — Heschl's gyrus, primary auditory cortex — appear in
both gain contrasts and in neither brightness contrast, along with lateral and
superior temporal regions. That dissociation is real and it is the reason to
keep the parameter.

But gain's *largest* single effect at both brightness levels is `S_occipital_ant`,
and both occipital poles rank high. **A purely auditory manipulation does not
straightforwardly predict its biggest effect in visual cortex.** Possible
readings — a multimodal model leaking across streams, or the audio change
shifting the event segmentation and with it the timing of visual features — are
guesses, and separating them needs work not done here.

**Two limits bound all of this.** These are effects on the order of 0.015 to
0.026, which is one to one-and-a-half times a floor that was measured as a
*whole-cortex* quantity. An ROI mean over 26 to 49 vertices is a far noisier
number than a mean over 20,484, so the ROI-level floor is certainly higher than
0.0157 and **has not been measured**. And there is one clip and one
deterministic run per condition, so there is no scatter to test against: what
carries the argument is anatomical coherence across two independent contrasts,
which is weaker evidence than a statistic and is offered as such.

None of it is a claim about hearing, seeing, or anything a person would
experience. It is a statement about which vertices of a predicted response
moved.

### The root is the best point measured

Every corner scored below the unedited stimulus, and brightness in *either*
direction lowered the objective. The surface has a maximum at or near the
neutral point rather than at an edge.

A search over this space would therefore be expected to conclude that the
unedited stimulus is the best of those tried. That is a real result and a
falsifiable one, and it is worth saying plainly that it is the opposite of the
outcome an optimization phase is usually built hoping for.

### Why the real-pipeline surrogate benchmark is not being run

Steps 74-76 ask for a sample-efficiency comparison on the real pipeline. With
the range now measured, that comparison can be costed, and the cost is not the
reason to decline it.

A benchmark of three methods over a 24-evaluation budget is 72 real evaluations
at about 25 minutes each: **30 hours for a single seed**, and one seed of a
stochastic search measures luck. Several seeds is a week of continuous compute.

What that week would buy is a sample-efficiency number on a **one-dimensional,
apparently single-peaked function whose peak is already known** — it sits at the
neutral point, located by five evaluations. Bayesian optimization exists to
spend a small budget well on a surface nobody can characterise cheaply. Run
against a surface already characterised, a favourable result would demonstrate
that the method can rediscover a known answer, and an unfavourable one would be
attributed to the surface rather than the method. Neither outcome would change
what anyone should believe.

The synthetic benchmark in §7 already establishes what a benchmark can
establish here: the mechanism works, and it beats both random search and a
Phase 9 heuristic where signal is known to exist. The honest gap is not
measurement, it is that **no real search space with demonstrated multi-parameter
structure exists yet to measure on**. Finding one is prior work to a real
benchmark, not something a benchmark substitutes for.

### What this changes

1. **The current space (±6 dB, ±0.15) should not be searched.** It spans 0.004,
   a quarter of the floor, and nothing found inside it could be a finding.
2. **The wider space (±15 dB, ±0.35) is usable, with the root counted.** Its
   resolvable variation is real but modest, at twice the floor.
3. **Audio gain should not be spent as a search dimension under a whole-cortex
   objective** — it contributes 1-3% of the floor there — but it should not be
   removed from the space. Under an objective targeting auditory regions it is
   the one parameter with a plausible claim to being measured. The fix belongs
   to the objective, not the parameter.
4. **A parameter's effect must be measured before it is searched**, and measured
   the way `measure_roi_effects.py` measures it rather than by a single scalar.
   Contrast, saturation and speed are the untested candidates the materializer
   already supports, and any of them could have the same shape as gain: real,
   regional, and invisible to an average.
5. **The ROI-level nuisance floor should be measured next.** Every regional
   claim above is compared against a whole-cortex floor because no better number
   exists, and that comparison is generous to the claims. Re-running the encoder
   swap of `scientific_limitations.md` §6a and recording per-ROI shifts would
   produce the right denominator, and it is the cheapest useful experiment
   remaining.

## 9. Scientific limitations

**A predicted score is not an outcome.** It is a model's estimate of what the
real pipeline would report, and the real pipeline is the only authority. The
wording is enforced in the types and the interface rather than left to
convention.

**Rank skill on synthetic surfaces does not transfer.** Those surfaces are
smooth, noiseless and two-dimensional; a predicted cortical response is none of
those. What the synthetic benchmark establishes is that the mechanism works, not
that it will work here.

**The bootstrap cost is real.** A method that spends six evaluations before it
starts choosing cannot beat random sampling to an easy target reached in three.
Surrogate assistance pays off in final quality and reliability, not in speed to
a low bar, and the benchmark reports both so the trade is visible.

**No global optimum is claimed, ever.** Bayesian optimization is a heuristic for
spending a small budget well. The system says "best observed".

**Feasibility modelling is not implemented.** Step 44 asks for a classifier
predicting whether a candidate satisfies the guardrails. Every candidate in the
corpus was feasible and none failed, so there are zero negative examples and the
step's own condition is not met. The constrained acquisition path exists and is
applied only when a caller supplies a feasibility estimate.

**Multi-fidelity is an interface with one level.** There is no validated
low-cost approximation of a TRIBE pass, and adding one without measuring its
agreement with the full pass would make every elimination it drove
untrustworthy.

## 10. What is not built

Everything Step 94 defers: a learned policy, cross-user modelling, conversion
prediction, demographic targeting, autonomous deployment, causal claims. None
started.

Also not built: TPE and Optuna-style samplers, which exist for mixed conditional
spaces of higher dimensionality than this one has; and surrogate ensembles,
which Step 39 marks optional and which are not justified before the individual
models have data to distinguish them.
