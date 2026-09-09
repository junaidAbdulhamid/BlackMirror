# Bayesian optimization, plainly

Step 91. What the technique is, why it fits this problem, and what it does not
promise.

---

## The problem it solves

Some functions are expensive to evaluate. Ours costs roughly 23 minutes for a
four-second clip and 52 for a ten-second one, because a single evaluation
materializes media, runs TRIBE inference, then analytics, content analysis,
scoring and comparison.

With a cheap function you would simply try thousands of inputs. With an
expensive one you get maybe ten tries, and the whole question becomes *which
ten*. Bayesian optimization is a way of answering that with a model instead of a
guess.

## The idea in three parts

**1. Fit a cheap model to what you already measured.** After a handful of real
evaluations you have pairs of (candidate, score). A surrogate learns an
approximation from them:

    F_hat(x) ≈ F(x)

Predicting with it takes microseconds, so you can score thousands of untested
candidates for nothing.

**2. Ask the model not just what it predicts, but how sure it is.** A Gaussian
Process returns both: a mean `mu(x)` and a standard deviation `sigma(x)`. Near
points you have measured, `sigma` is small. Far from them it grows, because the
model genuinely does not know.

**3. Choose using both.** This is the part that distinguishes Bayesian
optimization from just maximising the model. An *acquisition function* scores
each candidate by how worthwhile evaluating it would be:

    UCB(x) = mu(x) + kappa * sigma(x)

High `mu` is **exploitation**: the model thinks this is good. High `sigma` is
**exploration**: the model does not know, and finding out would teach it
something. `kappa` sets the balance.

Then you spend one real evaluation on the winner, add the answer to your data,
refit, and repeat.

## Why not just maximise the model

Because the model is wrong, and it is most wrong where it has no data. Picking
the highest predicted score sends every evaluation into the region the model
already understands, so it never learns anything new and converges on whatever
its first few points happened to suggest. That failure has a name in the
codebase: it is the `GREEDY_MEAN` acquisition, kept precisely so the
uncertainty-aware ones have a baseline to beat.

## The acquisition functions

Let `f*` be the best real observation so far and `Phi`, `phi` the standard
normal CDF and density.

| name | formula | reads as |
| --- | --- | --- |
| Greedy mean | `mu(x)` | "best guess wins" |
| Upper confidence bound | `mu + kappa*sigma` | "optimistic estimate wins" |
| Probability of improvement | `Phi(z)` | "most likely to beat the incumbent" |
| Expected improvement | `(mu - f* - xi)*Phi(z) + sigma*phi(z)` | "largest expected gain" |

with `z = (mu - f* - xi) / sigma`.

**Expected improvement is the usual default** and is the default here. It values
a candidate by how much better it is expected to be, which naturally prefers a
moderately good but uncertain candidate over a slightly better but certain one.

**A caveat the code enforces rather than mentions.** Expected improvement and
probability of improvement are exact only if the predictive distribution really
is Gaussian. That holds for a Gaussian Process. It does not hold for the spread
of a random forest's trees, which is disagreement among correlated learners, not
a posterior. UCB makes no distributional claim and degrades gracefully; the
others produce a number that looks like a probability and is not one. The
registry records which acquisitions assume a Gaussian, and the orchestrator
falls back to greedy rather than computing EI on a quantity that cannot support
it.

## Exploration and exploitation, concretely

Suppose the best real score so far is 0.40, and two candidates are on offer:

- **A**: predicted 0.44, uncertainty 0.01
- **B**: predicted 0.42, uncertainty 0.20

Greedy picks A. Expected improvement usually picks B, because A can only be
about 0.04 better while B might be much better or much worse, and finding out
which is worth an evaluation. If B disappoints, the model learns that region is
poor and stops proposing it; that is not a wasted evaluation, it is the one that
stopped a whole region being retried.

## Why this problem suits it

Bayesian optimization is at its best when evaluations are expensive, dimensions
are few, and observations are quiet. All three hold here:

- **Expensive**: 23 to 52 minutes per candidate.
- **Low dimensional**: the search space holds at most one parameter per edit
  operation, and there are five operations, so two to five *declared*
  dimensions. Measured, it is lower still: across a ±15 dB by ±0.35 brightness
  grid, loudness moved the *whole-cortex mean* by 1-3% of the nuisance floor
  while brightness moved it by 98-100%, making that two-parameter space
  effectively one-dimensional **under that objective**. Per region the picture
  differs — loudness moves individual vertices by up to 42x what it moves their
  average, so it is cancelling rather than absent, and a different objective
  would see it. The condition holds more strongly than it was claimed to, but
  the declared count was not the measured one, and the measured one depends on
  the objective. See
  [`phase10_surrogate_optimization.md`](phase10_surrogate_optimization.md) §8.
- **Quiet**: measured directly, two runs of the same stimulus under the same
  configuration give bit-identical predictions. Observation noise is zero, so
  the Gaussian Process uses a noise term reflecting numerical tolerance rather
  than measurement scatter.

## What it does not promise

**Not a global optimum.** Bayesian optimization has no guarantee of finding the
best point in the space. It is a heuristic for spending a small budget well. The
system says "best observed", and means it.

**Not a replacement for the real evaluator.** The surrogate screens; TRIBE
decides. Every number a surrogate produces is labelled an estimate, and no
candidate is reported as good until the real pipeline has measured it.

**Not automatic.** A surrogate that cannot rank candidates is worse than none,
because it steers every expensive evaluation confidently into a region it has
misunderstood. The trust gate exists for that: below a demonstrated rank
correlation on order-aware validation, the surrogate is not allowed to choose
and selection falls back to a Phase 9 strategy.

**Not free of a cold start.** Nothing can be fitted to zero observations. The
first several evaluations are spent covering the space, and only then does the
model begin to steer. That bootstrap cost is real and shows up in the
benchmarks: a method that spends six evaluations before it starts choosing
cannot beat random sampling on an easy target reached in three.

## Further reading in this repository

- [`phase10_surrogate_optimization.md`](phase10_surrogate_optimization.md) for
  what was built and what it measured.
- [`surrogate_evaluation.md`](surrogate_evaluation.md) for how model quality is
  judged and where the trust thresholds come from.
