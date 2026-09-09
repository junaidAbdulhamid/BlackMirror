# Phase 10 Integration Report

Written before any Phase 10 code, from an inspection of Phases 1–9 as they
actually stand. Where the code and the brief's assumptions differ, this follows
the code and says so.

**The headline finding is in §8 and it changes what Phase 10 can honestly
claim.** There are five unique evaluated candidates, and the entire spread of
their objective values is a quarter of the pipeline's measured nuisance floor.
A surrogate can be built and validated; what it cannot yet do is demonstrate
that it helps on real data.

---

## 1. Available training data

Every real evaluated candidate in the repository:

| search | candidate | gain | brightness | fitness | origin |
| --- | --- | --- | --- | --- | --- |
| random_search-17 | c0001 | +0.5 | +0.09 | -0.026397 | random sample |
| random_search-17 | c0002 | +5.5 | -0.06 | -0.027350 | random sample |
| random_search-17 | c0003 | +3.0 | +0.06 | -0.023349 | random sample |
| local_search-17 | c0001 | +0.5 | +0.09 | -0.026397 | random sample (cache hit) |
| local_search-17 | c0002 | +0.0 | +0.09 | -0.025313 | neighbour |
| local_search-17 | c0003 | -0.5 | +0.09 | -0.025414 | neighbour |

Six rows, **five unique genomes**, one root, one objective, one search-space
version. The duplicate is the shared cache working correctly, not a second
observation.

They live in two places, both usable:

- `artifacts/search/<search_id>/evaluations/<candidate_id>.json`, the full
  `CandidateFitness` including genome, raw objectives, feasibility and compute.
- `artifacts/search/cache/<prefix>/<key>.json`, the same records keyed by
  candidate identity, which is what makes them reusable across searches.

The Phase 8 re-simulation `phase8-e2e-2` is a real evaluated variant but is
**not** usable training data: it was user-supplied media with no genome, so
there is no feature vector to learn from.

## 2. Genome feature types

`CandidateGenome.values` is `dict[str, float]`. Every parameter, whatever its
declared type, is carried as a float, and the type lives on the
`SearchParameter` in the space rather than on the genome.

The space supports four realizable types: `CONTINUOUS`, `INTEGER`,
`CATEGORICAL`, `BOOLEAN`. Three more are declared and **rejected at
construction**: `EVENT_TIMING`, `TEXT_CANDIDATE`, `STRUCTURED_INTERVENTION`,
because no materializer implements them.

This settles two of the brief's steps before they begin. Step 4's
high-cardinality categorical encoding and Step 5's text embeddings have nothing
to encode: there are no text parameters, and categorical parameters map to a
handful of numeric choices. Building embedding machinery now would be
scaffolding around a feature type the system cannot produce.

## 3. Typical search dimensionality

**Two in practice, five as a hard ceiling.** A space may hold at most one
parameter per `EditOperation`, the constraint exists so two parameters cannot
fight at materialization, and there are exactly five operations: audio gain,
brightness, contrast, saturation, speed.

That is unusually favourable for surrogate modelling. Low dimensionality is
where Gaussian Processes are strongest, and it removes the usual argument for
tree ensembles over GPs on mixed high-dimensional spaces.

## 4. Objective types

`NeuralObjective` carries `MAXIMIZE`, `MINIMIZE` or `TARGET`.
`directional_fitness` already folds all three so higher is better, which is the
form a surrogate should learn: one convention, no per-objective special cases
inside the model.

`TARGET` needs care. Folding it produces the negative distance from the target,
which is a **kinked** function of the raw value. A GP with a smooth kernel will
model that badly near the target. Training on the raw value and folding after
prediction avoids the kink, and is the better default.

## 5. Multi-objective support

Present and reusable. `CandidateFitness.raw_objectives` stores every objective's
value per candidate, so one surrogate per objective (Step 41) needs no schema
change. `search/pareto.py` provides dominance, frontier extraction,
non-dominated sorting and front-wise selection, all on directional fitness.

Step 42's warning is well founded here: with five samples, EHVI or anything of
that sophistication would be fitting a frontier to noise. Random scalarization
(Step 43) is the appropriate ceiling.

## 6. Current number of evaluated samples

**Five unique.** For context on what more would cost: roughly 23 minutes per
candidate on the four-second clip and 52 on the ten-second one. Twenty samples
is about eight hours; fifty is about nineteen.

## 7. Candidate evaluator

`CandidateEvaluator.evaluate(genome) -> CandidateFitness` is the oracle, and it
already does everything Phase 10 needs: decode, materialize, build a validated
Phase 8 request, run the pipeline, fold direction, judge feasibility, and return
failures as data rather than raising. `CachingEvaluator` wraps it so a repeated
genome costs nothing.

Phase 10 should call this and nothing lower. It should not touch Phase 8
directly.

## 8. Important data sparsity risks

**This is the finding that shapes the phase.**

| quantity | value |
| --- | --- |
| unique samples | 5 |
| dimensions | 2 |
| label range | -0.02735 to -0.02335 |
| label spread | 0.00400 |
| label standard deviation | 0.00126 |
| measured nuisance floor | 0.01570 |
| **spread ÷ floor** | **0.25x** |

The labels a surrogate would learn from span a quarter of the interval the
pipeline cannot resolve. Everything the model could learn to separate is
smaller than the amount the whole pipeline moves when an ostensibly equivalent
language-model mirror is swapped.

### The distinction that matters here

Observation noise and the nuisance floor are **not the same thing**, and
conflating them would lead to the wrong modelling choice.

- **Observation noise is zero.** Measured directly: two independent runs of the
  same stimulus under the same configuration produced bit-identical
  predictions. Re-evaluating a genome returns exactly the same number. A GP
  should therefore use a near-zero noise term, not an inflated one.
- **The nuisance floor is sensitivity to configuration**, 0.0157 in whole-cortex
  mean. It bounds whether a *conclusion* survives a change of implementation
  detail. It does not make individual labels noisy.

So a surrogate *can* fit this data exactly, because the function is
deterministic. What it cannot do is demonstrate that it generalizes, and any
improvement it steered toward would be below the floor and therefore not a
finding.

### What follows

1. The surrogate machinery should be built and validated on **synthetic
   functions with known structure**, where signal exists and correctness is
   checkable. That is Steps 77–83 and it is legitimate science.
2. The **trust gate (Step 50) is the feature that handles this situation**, and
   the real corpus is the ideal test of it. A correctly built system should
   refuse to hand over candidate selection to a surrogate that cannot
   demonstrate rank skill, and fall back to Phase 9. Demonstrating that refusal
   on real data is a stronger result than a fabricated success.
3. The real benchmark of Steps 74–76 **cannot yet show a sample-efficiency
   gain**, and should not be presented as if it might. Reporting "Bayesian
   optimization reached the threshold in 7 evaluations instead of 18" requires
   a threshold that means something, and no threshold in this corpus does.

### What would change it

Either a wider search space whose effects exceed the floor, or a longer
budget. From the corpus re-run, deliberate brightness and loudness edits
produced whole-cortex means spanning about 0.015, comparable to the floor,
whereas the current space (±6 dB, ±0.15 brightness) spans 0.004. A space with
larger permitted edits is the cheaper of the two fixes and should be tried
first.

### It was tried, and this is what it found

Four corners of a space at ±15 dB and ±0.35 brightness, each a real
materialization through the real pipeline, 94 minutes of compute. Recorded in
full in [`phase10_surrogate_optimization.md`](phase10_surrogate_optimization.md)
§8; the artifact is `artifacts/benchmarks/space_range.json`.

**The widening works, but only just, and only when counted correctly.** Across
the four corners the range is 0.0159, which clears the floor by one percent —
not a margin. Including the root, which is a legitimate point in the space and
the baseline every candidate is compared against, the range is 0.0319, or
**2.03x the floor**. That is genuinely resolvable, where the current space at
0.25x was not.

**The more consequential finding was not the one being looked for.** Read as a
factorial rather than as a spread, a 30 dB swing in loudness moves the predicted
whole-cortex mean by 0.0001 to 0.0004 — **1 to 3% of the floor** — while the
same swing in brightness moves it by 0.0155, or 98 to 100%. The paired
measurements agree closely within each parameter, so this is structure and not a
single odd corner.

**The natural conclusion from that table is wrong, and a cheap check caught it.**
Taking the same contrasts per region rather than as a single average
(`scripts/measure_roi_effects.py`, artifact
`artifacts/benchmarks/roi_effects.json`): gain moves individual vertices by up to
**42 times** what it moves the whole-cortex mean, where brightness moves them by
2.4 times. Gain is not inert — it is *cancelling*, and a mean over 20,484
vertices is the wrong instrument for it. Among the regions it moves are the
transverse temporal gyrus and sulcus (Heschl's gyrus, primary auditory cortex),
which appear in both gain contrasts and neither brightness contrast; brightness
lands squarely on calcarine, lingual and fusiform cortex at twelve times the
floor. The dissociation is the reason to keep the parameter.

It is a lead rather than a result: gain's largest single effect is in an
occipital region, which a purely auditory manipulation does not predict; the
magnitudes are near a floor measured as a whole-cortex quantity, so the relevant
ROI-level floor is higher and unmeasured; and one clip with one deterministic
run per condition supports no significance test.

The space is therefore nominally two-dimensional and, **under this objective**,
effectively one-dimensional. That corrects a premise this report was written on:
§9's case for a Gaussian Process cited "two to five dimensions", which described
the declared space rather than the one this objective responds to. The
conclusion survives — a GP is, if anything, more clearly right at one effective
dimension — but it was reached from a number that was wrong.

**And the root is the best point measured.** Every corner scored below the
unedited stimulus, in both brightness directions. A search over this space would
be expected to report that the unedited stimulus wins.

**Consequences for this phase**, all of which are now recorded rather than
pending:

1. The current space should not be searched at all: 0.25x the floor.
2. The wider space is usable, at 2.03x, with the root counted.
3. Audio gain should not be spent as a search dimension *under a whole-cortex
   objective*, where it returns 1-3% of the floor — but it should stay in the
   space. The defect is in the objective, which averages the effect away, not in
   the parameter. Under an ROI-targeted objective it is the better candidate of
   the two.
3b. The ROI-level nuisance floor is now the cheapest useful experiment
   outstanding, because every regional claim above is being compared against a
   whole-cortex floor for want of a better denominator.
4. The real-pipeline sample-efficiency benchmark of Steps 74-76 is **declined
   with a reason rather than deferred**. It would cost about 30 hours per seed
   to measure a method built for uncharacterised surfaces against a
   one-dimensional one whose peak five evaluations already located. §8 of the
   surrogate document sets out the arithmetic.

## 9. Suggested surrogate families

Available without new dependencies: **scikit-learn 1.8.0** and **scipy 1.18.1**,
both currently transitive rather than declared, so Phase 10 must add them to
`pyproject.toml` explicitly. No optuna, botorch, gpytorch or skopt, and the
brief's warning against blindly adding dependencies applies.

| family | verdict |
| --- | --- |
| **Gaussian Process** (RBF / Matérn) | **Default.** Two to five dimensions, deterministic observations and tiny n is precisely where GPs are strongest, and the posterior variance is the only genuinely principled uncertainty available. |
| **Random Forest / Extra Trees** | Baseline and comparison. Ensemble spread is a usable exploration signal but is **not** a posterior and must not be documented as one. |
| **Gradient Boosting** | Available, likely to overfit at this n. Include in the registry, do not default to it. |
| **TPE / Optuna** | Not justified. It exists for mixed conditional spaces of higher dimensionality, and adopting a dependency to solve a problem this space does not have would be the wrong trade. |

Expected Improvement and UCB both need a predictive standard deviation. The GP
supplies one properly; the forest supplies a proxy. Both should be implemented,
with the difference documented rather than smoothed over.

## 10. Other integration facts

**Versioning already exists and should be reused rather than duplicated.**
`ContentSearchSpace.search_space_version`, `NeuralObjective`'s definition hash,
`MATERIALIZATION_VERSION`, the scoring version and the model fingerprint are all
present, and `CacheIdentity` already composes exactly those into one identity.
Steps 54–55, 69 and 70 want the same composition; the surrogate dataset should
key on it rather than inventing a parallel scheme.

**Budget integration is straightforward.** `BudgetLedger.with_evaluation`
already separates a cache hit from a TRIBE run, so surrogate predictions
naturally cost nothing against the budget. `admit()` already enforces Step 87's
requirement that scoring ten thousand candidates cheaply cannot schedule more
real evaluations than remain.

**Feasibility data for Step 44 does not exist.** Every one of the five
candidates was feasible and none failed, so there are zero negative examples. A
feasibility classifier cannot be trained, and the step's own condition, "only if
enough failed/valid examples exist", is not met.

## 11. Recommended order

The brief's order is sound. One emphasis: its own CRITICAL note says to prove
the encoder-to-surrogate-to-prediction chain against real observations before
building the orchestrator, and given §8 that check will fail on the real corpus.
It should still be run, and its failure recorded, because the trust gate that
follows is the correct response to it.

I would add one step before the benchmark: **widen the search space and measure
whether the label spread clears the floor**. If it does not, no amount of
surrogate quality will produce a reportable sample-efficiency result, and that
is worth knowing before spending eight hours generating training data.
