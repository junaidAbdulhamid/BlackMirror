# Evaluating the surrogate

Step 92. How model quality is measured, which number decides whether the
surrogate may choose candidates, and why the thresholds are what they are.

---

## The metric that decides

**Spearman rank correlation on order-aware validation.**

The surrogate's job is to decide which candidate receives the next expensive
evaluation. That is a ranking decision, not a prediction task. A model with a
large constant bias but perfect ordering picks the right candidate every time; a
model with tiny error but shuffled ordering picks badly and costs an hour doing
it. So rank correlation gates, and error metrics report.

## Every metric collected

| metric | what it says | why it is here |
| --- | --- | --- |
| MAE | mean absolute error | whether the numbers shown to a person are believable |
| RMSE | root mean squared error | as MAE, but punishes large misses |
| R² | variance explained | `None` when labels are constant, where it is undefined |
| **Spearman** | rank correlation | **the gate**: can it order candidates at all |
| Kendall | rank correlation, pair-based | a second opinion, less sensitive to outliers |
| **Top-K recall** | of the true best K, how many it also ranked top K | the most direct statement of the job |
| Uncertainty error correlation | does uncertainty track actual error | whether the uncertainty means anything |
| Coverage at one sigma | fraction inside the ±1σ band | whether the uncertainty is scaled right |
| OOD distance | distance to nearest training point | whether the model is extrapolating |

`None` is used deliberately throughout. Rank correlation over constant labels is
undefined, not zero, and returning zero would read as "the model cannot rank"
when the truth is "there is nothing to rank".

## Two validation schemes, and which to believe

**Random K-fold** predicts a held-out point using observations that, in the real
search, had not happened yet. That is leakage from the future. It reads higher
than what the model will actually do.

**Order-aware (prefix) validation** predicts each observation from only the ones
that preceded it. That is exactly what happens online. It is the honest number
and it is what the trust gate reads.

Both are computed and both are reported. Where they disagree, the prefix number
is the one that matters; the interface labels the random-fold scheme with a
warning when it has to fall back to it.

K-fold switches to leave-one-out automatically below twice the fold count,
because a five-fold split of eight points leaves folds too small to say
anything, and an arbitrary split would look like a measurement without being
one.

## The trust gate

Configured by `TrustPolicy`:

| setting | default | reason |
| --- | --- | --- |
| `min_samples` | 8 | below this, cross-validation produces a number but not a measurement; one fold's luck moves it entirely |
| `min_spearman` | 0.2 | a weak but real ordering signal; below it the model is not adding information over random |
| `min_top_k_recall` | none | available, off by default, because at small K it is very coarse |
| `retrain_every` | 1 | at these sizes every observation is a large fraction of what is known |

Failing the gate is not an error state. The surrogate keeps training and keeps
reporting; it simply does not get to choose candidates, and selection falls back
to a Phase 9 strategy. Each round records which mode chose it, so a run is
auditable afterwards.

**The threshold is deliberately low.** 0.2 is a weak correlation. The gate is
not asserting that the model is good, only that it is better than nothing at
ordering. A higher bar would be defensible and would simply mean more rounds
spent in fallback.

## Two kinds of uncertainty

These are different objects and the code refuses to blur them.

**Posterior standard deviation** (Gaussian Process) comes from an explicit model
of the function. It is small near observations and grows with distance from
them. Expected improvement and probability of improvement have their intended
meaning under it.

**Ensemble spread** (random forest, extra trees) is disagreement among
correlated learners fitted to the same data. It behaves roughly like uncertainty
inside the data and badly outside it: extrapolate past the training range and
every tree returns the same boundary value, so the spread *collapses* exactly
where confidence should be lowest.

Every prediction carries `uncertainty_kind` so a reader cannot mistake the
second for the first, and the interface prints which one it is looking at.

## Out-of-distribution distance

Model uncertainty can fail silently, so distance to the nearest training point
is computed independently, in the encoded feature space where every parameter
has been mapped to a comparable range.

It is a property of the data rather than of the model, which is what makes it
catch the failure mode ensemble spread misses. A candidate far from everything
observed should be treated cautiously whatever the model claims.

## What good looks like, measured

From the synthetic surfaces, where signal is known to exist and the optimum is
known. `rho_k` is five-fold (leave-one-out below ten samples); `rho_pfx` is the
order-aware number the gate reads; `top3` is recall of the true best three.

### one_peak — a single smooth maximum

| n | model | MAE | RMSE | R² | rho_k | rho_pfx | top3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | GP | 28.70 | 36.12 | 0.480 | 0.595 | 0.700 | 0.67 |
| 12 | GP | 12.26 | 18.29 | 0.856 | 0.881 | 0.683 | 0.67 |
| 20 | GP | 5.53 | 9.58 | 0.961 | 0.964 | 0.792 | 1.00 |
| 30 | GP | **0.60** | **1.07** | **1.000** | **1.000** | **0.856** | **1.00** |
| 8 | RF | 18.05 | 22.25 | 0.803 | 0.643 | 0.800 | 0.67 |
| 12 | RF | 14.92 | 18.68 | 0.850 | 0.839 | 0.833 | 0.67 |
| 20 | RF | 16.66 | 23.16 | 0.774 | 0.664 | 0.824 | 0.00 |
| 30 | RF | 8.33 | 13.04 | 0.932 | 0.965 | 0.882 | 0.67 |

### two_peaks — a broad low peak and a tall narrow one

| n | model | MAE | RMSE | R² | rho_k | rho_pfx | top3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | GP | 0.725 | 0.829 | 0.006 | 0.286 | 0.400 | 0.33 |
| 12 | GP | 0.503 | 0.582 | 0.506 | 0.811 | 0.533 | 1.00 |
| 20 | GP | 0.491 | 0.647 | 0.635 | 0.651 | 0.632 | 0.33 |
| 30 | GP | **0.212** | **0.350** | **0.931** | **0.899** | **0.786** | 0.67 |
| 8 | RF | 0.743 | 0.835 | **-0.009** | 0.452 | **-0.100** | 0.67 |
| 12 | RF | 0.589 | 0.769 | 0.137 | 0.559 | 0.533 | 0.67 |
| 20 | RF | 0.833 | 1.115 | **-0.085** | 0.605 | 0.632 | 0.00 |
| 30 | RF | 0.563 | 0.809 | 0.630 | 0.816 | 0.799 | 0.67 |

Four things in these numbers are worth reading deliberately.

**The Gaussian Process wins on accuracy, decisively.** At 30 samples its error
is an order of magnitude below the forest's on the smooth surface (MAE 0.60
against 8.33) and less than half on the harder one, with R² 1.000 against 0.932
and 0.931 against 0.630. That is the regime argument in §3 of the phase document
holding up rather than being asserted.

**But not on the metric that gates it, and that deserves saying plainly.** At
n=30 the forest's *prefix* rank correlation is slightly higher on both surfaces:
0.882 against 0.856 on one_peak, 0.799 against 0.786 on two_peaks. Two cells,
both narrow, and the GP leads at every smaller sample size and on every other
measure — but the honest summary is that this evidence does not establish the GP
as the better *ranker* at 30 samples, only as the far better *estimator*. The
default stands on the accuracy, the uncertainty being a real posterior, and the
forest's extrapolation collapse described above; it does not stand on a prefix
rho advantage that was not observed here.

**The forest's R² goes negative twice** — at n=8 and n=20 on two_peaks it
predicts worse than the mean of the training labels would. Its rank correlation
stays positive in the same cells, which is the whole reason ranking gates and
error reports.

**Prefix validation reads lower than k-fold, consistently.** The clearest case
is the GP on one_peak at n=30: 1.000 under k-fold against 0.856 order-aware.
That gap is the leakage described above, measured rather than argued, and it is
why the pessimistic number is the one the gate reads.

**Eight samples is not enough on a harder surface.** The GP's R² at n=8 on
two_peaks is 0.006 — it explains nothing — with rank correlation 0.286. The
`min_samples = 8` default is a floor below which nothing is attempted, not a
promise that eight suffices.

Uncertainty near observed points is also verified to be smaller than uncertainty
far from them, directly rather than by assumption.

## What bad looks like, also measured

Given a flat objective, the trainer reports `spearman = None`, the gate refuses,
the state becomes `DEGRADED`, and the loop runs entirely in fallback. That path
is tested, because it is the one the real corpus currently takes.

## A note on the kernel

The Gaussian Process uses a Matérn kernel with `nu = 2.5` rather than an RBF.
RBF assumes the function is infinitely differentiable, which is a strong claim
about a response surface nobody has characterised; Matérn 2.5 is twice
differentiable and is the standard choice in Bayesian optimization for that
reason.

On smooth surfaces the amplitude and length scale are only weakly identified
together: a long length scale needs a large amplitude to produce the same
variance, so the optimizer walks to a bound and scikit-learn reports
non-convergence. Measured across sample sizes and bound choices, rank
correlation stayed at 1.0 throughout, so this is a degeneracy in the
parameterisation rather than a failure to fit. The warning is silenced and the
fact recorded instead: `hyperparameters_at_bounds` appears in the model
metadata, so a genuinely stuck fit is still visible.

## Observation noise

Measured, not assumed. Two independent runs of the same stimulus under the same
configuration produce **bit-identical** predictions, so re-evaluating a genome
returns exactly the same number. The GP's noise term therefore reflects
numerical tolerance, not measurement scatter.

This is distinct from the **nuisance floor** of 0.0157, which is the pipeline's
sensitivity to a configuration change. That bounds whether a *conclusion*
survives, not whether a label is noisy, and inflating the noise term with it
would smooth away real structure. See
[`scientific_limitations.md`](scientific_limitations.md) §6a.
