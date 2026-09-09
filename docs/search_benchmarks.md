# Search benchmarks

Step 94. Recorded results, with the configuration needed to reproduce them.

## Synthetic surfaces (Step 87)

Command:

    python scripts/benchmark_search_strategies.py --budget 40 --seeds 10

Space: two continuous parameters, gain in [-10, 10] at resolution 0.1 and
brightness in [-1, 1] at resolution 0.01. Ten seeds, 0 through 9, per strategy.
Budget of 40 evaluations each. Evaluation is free here, so the budget is a
proxy for the 52 minutes each would cost on the real pipeline.

### one_peak

A single smooth maximum at (3, 0.4). Optimum 0.0.

| strategy | median best | gap to optimum | evaluations used | evaluations to best |
| --- | --- | --- | --- | --- |
| local | -0.043 | 0.043 | 40 | 28 |
| beam | -0.235 | 0.235 | 40 | 30 |
| evolutionary | -0.242 | 0.242 | 40 | 30 |
| epsilon_greedy | -0.266 | 0.266 | 40 | 33 |
| random | -0.528 | 0.528 | 40 | 18 |
| grid | -1.400 | 1.400 | 36 | 23 |
| hill_climbing | -10.550 | 10.550 | 4 | 3 |

### two_peaks

A broad low peak near (-6, 0) reaching 2.0, and a tall narrow one near (6, 0.5)
reaching 6.0. Designed so a climber starting in the wrong basin stays there.

| strategy | median best | gap to optimum | evaluations used | evaluations to best |
| --- | --- | --- | --- | --- |
| grid | 5.940 | 0.060 | 36 | 29 |
| local | 5.936 | 0.064 | 40 | 27 |
| epsilon_greedy | 5.852 | 0.148 | 40 | 26 |
| beam | 5.843 | 0.157 | 40 | 22 |
| random | 5.663 | 0.337 | 40 | 24 |
| evolutionary | 3.967 | 2.033 | 40 | 22 |
| hill_climbing | 2.224 | 3.776 | 4 | 2 |

## What these show

**Local search was strongest on both surfaces.** Its axis-aligned neighbourhood
refines efficiently, and the restart on exhaustion lets it leave a basin.

**Hill climbing was worst on both, and visibly trapped on the second.** It
reached 2.224 against a true optimum of 6.0, which is the broad low peak almost
exactly. It also used only 4 of its 40 evaluations, because it stops the moment
a round brings no improvement. That is correct behaviour and a poor use of a
budget; Step 19's limitation is not theoretical.

**Evolutionary search underperformed random sampling on two_peaks.** Populations
need generations to pay off, and generations cost evaluations. At a budget of 40
it had not had enough of them. This is worth stating plainly rather than
presenting the algorithm as strictly better because it is more sophisticated.

**Grid search topped two_peaks and was near the bottom on one_peak.** Its
accuracy depends on whether the optimum sits near a lattice point, which is luck
rather than search.

## What these do not show

Nothing here supports a claim that any strategy is best in general. These are
two smooth, noiseless, two-dimensional surfaces evaluated for free. A predicted
cortical response is high-dimensional, expensive, and carries a measured
nuisance floor of 0.0157 that these surfaces do not simulate at all.

The comparison that would count is Step 88: random search against one
intelligent strategy on the real pipeline, identical root, objective, space and
budget. That has **not** been run. At roughly 52 minutes per evaluation, a
ten-evaluation comparison of two strategies is about 18 hours.

## Real pipeline benchmark (Step 88)

Command:

    python scripts/benchmark_search_pipeline.py --budget 3 --seed 17

**Configuration.** Root run `20260906T225022Z-630b9b86`, the four-second Sintel
speech clip on the gated text encoder, with analytics and content analysis
already present. Objective: whole-cortex mean response, maximize, stored under
experiment `search-bench`. Search space: audio gain in [-6, 6] dB at 0.5
resolution and brightness in [-0.15, 0.15] at 0.01. Budget of three candidates
per strategy, seed 17, batch size one. Random search against local search, with
everything else held identical, which is the only way the comparison means
anything.

The evaluation cache is shared between the two strategies deliberately: if both
propose the same point, charging one of them twice for it would distort the
compute figures for no reason.

### Results

| strategy | best fitness | vs root | evaluations | to best | TRIBE runs | wall clock | cache hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random search | -0.023349 | -0.000198 | 3 | 3 | 3 | 69.2 min | 0% |
| local search | -0.025313 | -0.002162 | 3 | 2 | 2 | 45.8 min | 25% |

Root fitness: **-0.023151**. Both searches stopped on `max_candidates`. No
candidate failed.

### What actually happened

**Neither strategy beat the root.** Random search's best was 0.000198 below it;
local search's was 0.002162 below. Both are reported as `resolvable=False`,
because both are far smaller than the measured nuisance floor of 0.0157. The
honest reading is that six evaluations over this space found nothing
distinguishable from the starting point, in either direction.

**Neither strategy beat the other, either.** The gap between them is 0.00196,
an order of magnitude below the floor. Reporting local search as "worse" here
would be reading noise. What can be said is that within this budget neither
demonstrated an advantage.

**The whole spread was inside the noise.** Every one of the six candidates fell
between -0.0273 and -0.0233, a range of 0.004. The pipeline's own sensitivity to
swapping an ostensibly equivalent language-model mirror is four times that. Each
strategy also reports two candidates tied with its own best.

**The shared cache paid for itself immediately.** Both strategies seed from the
same seed, so local search's first proposal was the same genome random search
had already evaluated. It was served from cache in 0.0 minutes, and local search
finished with 2 inference runs against random search's 3 and 45.8 minutes of
wall clock against 69.2. That is a 34% saving on a benchmark of six candidates,
and it is why the cache keys on candidate identity rather than on the search.

### What this does not show

It does not show that random search is better than local search, or that this
search space contains nothing worth finding. Three evaluations per strategy on
one four-second clip is far too few to support either claim. What it does show
is that the machinery runs end to end on real media at real cost, and that the
reporting refuses to dress up a sub-noise-floor difference as a result.

The obvious follow-up is a larger budget on a longer clip with a search space
whose effects plausibly exceed the noise floor. At roughly 23 minutes per
candidate for a four-second clip and 52 for a ten-second one, a twenty-candidate
comparison of two strategies is between 15 and 35 hours of compute.
