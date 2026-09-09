# Search algorithms

Seven strategies, all implementing the same interface so the orchestrator never
asks which one it holds. Each entry says how it works, when it is the right
choice, what it costs, and where it fails.

The cost column that matters is **evaluations**, not runtime. One evaluation of
a ten-second clip is roughly 52 minutes of inference, so an algorithm that needs
twice as many evaluations takes twice as many hours.

---

## Random search

**How.** Samples points uniformly from the declared space, filtering duplicates
and the unedited root. Learns nothing from results.

**When.** Always, at least once. It is the control: any other strategy is only
worth its complexity if it beats random sampling on the same budget, space,
objective and seed.

**Strengths.** No assumptions about the surface. Cannot get trapped. Trivially
parallel. Its performance is a direct measure of how much of the problem is the
shape of the space rather than the algorithm.

**Weaknesses.** Ignores everything it has learned. On a surface with real
structure it wastes most of its budget far from anything good.

**Parameters.** `max_resample_attempts` bounds re-drawing before the space is
declared exhausted.

---

## Grid search

**How.** Enumerates an evenly spaced lattice. Refuses to start if the lattice
exceeds a configured size.

**When.** Small discrete spaces, or when full coverage matters more than
efficiency, for example when reporting a response surface rather than finding a
maximum.

**Strengths.** Exhaustive, reproducible, and trivially explainable. Coverage is
guaranteed rather than probabilistic.

**Weaknesses.** Combinatorial. Four parameters at five steps is 625 points,
which at 52 minutes each is three weeks. The size check exists because that
would otherwise be discovered by starting it. Its accuracy also depends on
whether the optimum happens to sit near a lattice point, which is luck rather
than search.

**Parameters.** `steps` per parameter, `max_points` as a hard refusal.

---

## Local search

**How.** Generates axis-aligned neighbours of the best point found so far: one
parameter moves at a time, by a fixed fraction of its range. When a
neighbourhood is exhausted it restarts from a fresh random point.

**When.** The default choice on a continuous space when the budget is small.

**Strengths.** Moves are attributable, because exactly one parameter differs
between a neighbour and its parent, so a better neighbour says which change
helped. The restart is what lets it leave a basin, which is the whole difference
from hill climbing. It was the strongest strategy on both synthetic surfaces.

**Weaknesses.** Deterministic neighbourhoods can step over a narrow peak
entirely. It has no memory of regions already explored beyond the duplicate
filter.

**Parameters.** `magnitude` sets the step as a share of each parameter's range.

---

## Hill climbing

**How.** Local search that stops when no neighbour improves on the current best
by more than a threshold.

**When.** When a quick answer near a known-good starting point is enough, and
the budget is better spent elsewhere once local progress halts.

**Strengths.** Cheapest by far. In the benchmark it used four evaluations of a
forty-evaluation budget, because it stops the moment it stops improving.

**Weaknesses.** It gets trapped, and this is not theoretical. On the two-peak
surface it ended at 2.224 against a true optimum of 6.0: it climbed the broad
low peak and never saw the tall narrow one. Nothing in its output distinguishes
that answer from a global best, which is exactly why the other strategies exist.
It also leaves most of the budget unspent, so "cheap" is not the same as
"efficient".

**Parameters.** `magnitude`, `minimum_improvement` for what counts as progress.

---

## Beam search

**How.** Keeps the top K candidates and expands all of them, round-robin, by
mutation. Records why each pruned candidate was dropped.

**When.** When a single retained point is too fragile and full enumeration is
unaffordable, which is most of the time.

**Strengths.** A second promising region survives pruning, so one bad basin does
not doom the run. Pruning decisions are explainable rather than inferred from
absence.

**Weaknesses.** Cost scales with the width. A wide beam approaches random
sampling; a beam of one is hill climbing without the stop.

**Parameters.** `beam_width`, `magnitude`.

---

## Epsilon-greedy

**How.** With probability 1-e, mutates the best point found so far. With
probability e, samples somewhere unrelated. The rate decays by generation and
never falls below a floor.

**When.** When the budget is large enough to afford some deliberate waste, and
the surface is suspected to have more than one basin.

**Strengths.** The explore/exploit balance is explicit and tunable, and every
decision is recorded, so a report can say why each candidate exists. The floor
guarantees the search never becomes unable to leave a basin it has reached.

**Weaknesses.** The rate is a guess. Too high and it is random search with extra
steps; too low and it is hill climbing.

**Parameters.** `exploration_rate`, `decay`, `minimum_rate`, `magnitude`.

---

## Evolutionary search

**How.** Maintains a population, selects parents by binary tournament,
recombines by uniform crossover or mutates, and keeps the fittest as the next
population. Elites survive unchanged.

**When.** Larger budgets and spaces where parameter combinations interact, so
that recombining two good candidates can plausibly produce a better one.

**Strengths.** Population diversity resists local optima. Elitism guarantees the
best answer is never lost to chance.

**Weaknesses.** The most parameters and the least interpretable path from root
to winner. It underperformed random sampling on the two-peak surface at a
forty-evaluation budget, which is the honest result: populations need
generations, and generations cost evaluations this pipeline does not have
cheaply. Binary tournament also needs two distinct parents, and with a nearly
converged population it falls back to mutation.

**Parameters.** `population_size`, `elite_count`, `crossover_rate`, `magnitude`.

---

## Choosing

On the evidence available, which is two synthetic surfaces at a forty-evaluation
budget:

- **Local search** was strongest on both.
- **Hill climbing** was worst on both, and demonstrably trapped on one.
- **Evolutionary search** did not repay its complexity at this budget.
- **Random search** was mid-table, which is the point of having it.

None of that transfers automatically to a real objective. These surfaces are
smooth, two-dimensional and noiseless; a predicted cortical response is none of
those. Run random search alongside whatever else you choose, and treat the
comparison on the real pipeline as the only one that counts.
