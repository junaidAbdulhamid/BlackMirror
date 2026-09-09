"""Steps 16-35: the search algorithms beyond random sampling.

WHAT THEY SHARE
    All of them implement the same protocol, so the orchestrator never asks
    which one it is holding. They differ only in `propose` and `observe`: where
    the next points come from, and what is remembered from the results.

WHY THEY ALL TRACK WHAT THEY HAVE PROPOSED
    An evaluation costs roughly 52 minutes. Re-proposing a point already
    evaluated is not a small inefficiency, it is most of an hour spent
    recomputing a known number. Every strategy here filters its own proposals
    against a canonical fingerprint before returning them.

WHY "BEST" HAS NO THRESHOLD INSIDE A STRATEGY
    A strategy climbing towards higher fitness uses a plain comparison, because
    gating its internal notion of best on the noise floor would make its path
    depend on the order results arrived. Whether an improvement is large enough
    to *believe* is a question for stopping and for the final report, not for
    deciding which point to explore from next.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from blackmirror.search.fitness import CandidateFitness, best_of
from blackmirror.search.genome import CandidateGenome, GenomeOrigin
from blackmirror.search.mutation import (
    MutationMagnitude,
    MutationRegistry,
    crossover,
    neighbours,
)
from blackmirror.search.schemas import SearchStrategyName
from blackmirror.search.space import ContentSearchSpace


class _Base:
    """Bookkeeping every strategy needs: space, seed, ids, and what it has seen."""

    name: SearchStrategyName

    def __init__(self) -> None:
        self._space: ContentSearchSpace | None = None
        self._rng = Random(0)
        self._seed = 0
        self._counter = 0
        self._seen: set[str] = set()
        self._results: list[CandidateFitness] = []
        self._stop_reason = ""

    # --- lifecycle --------------------------------------------------------

    def initialize(self, space: ContentSearchSpace, seed: int) -> None:
        self._space = space
        self._seed = seed
        self._rng = Random(seed)
        self._counter = 0
        self._seen = set()
        self._results = []
        self._stop_reason = ""

    @property
    def space(self) -> ContentSearchSpace:
        if self._space is None:
            raise RuntimeError("strategy used before initialize()")
        return self._space

    def _next_id(self) -> str:
        self._counter += 1
        return f"c{self._counter:04d}"

    def _accept(self, genome: CandidateGenome) -> CandidateGenome | None:
        """Return the genome unless it is neutral or already proposed."""
        space = self.space
        if space.is_neutral(genome.values):
            return None
        fingerprint = genome.fingerprint(space)
        if fingerprint in self._seen:
            return None
        self._seen.add(fingerprint)
        return genome

    def _from_values(
        self,
        values: dict[str, float],
        *,
        origin: GenomeOrigin,
        generation: int,
        parents: tuple[str, ...] = (),
    ) -> CandidateGenome | None:
        genome = CandidateGenome(
            genome_id=self._next_id(),
            values=self.space.clamp(values),
            origin=origin,
            generation=generation,
            parent_genome_ids=parents,
            proposed_by=self.name.value,
        )
        accepted = self._accept(genome)
        if accepted is None:
            # The id was consumed by a rejected proposal; that is harmless and
            # keeps ids strictly increasing, which makes a log readable.
            return None
        return accepted

    # --- observing --------------------------------------------------------

    def observe(self, results: Sequence[CandidateFitness]) -> None:
        self._results.extend(results)

    @property
    def best(self) -> CandidateFitness | None:
        return best_of(self._results)

    def should_stop(self) -> tuple[bool, str]:
        return (True, self._stop_reason) if self._stop_reason else (False, "")

    # --- state ------------------------------------------------------------

    def get_state(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "seed": self._seed,
            "counter": self._counter,
            "seen": sorted(self._seen),
            "stop_reason": self._stop_reason,
            "rng_state": _encode_rng(self._rng),
        }

    def set_state(self, state: dict[str, object]) -> None:
        self._seed = _as_int(state.get("seed"), 0)
        self._counter = _as_int(state.get("counter"), 0)
        seen = state.get("seen")
        self._seen = {str(item) for item in seen} if isinstance(seen, list) else set()
        self._stop_reason = str(state.get("stop_reason") or "")
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(_decode_rng(rng_state))


class GridSearch(_Base):
    """Step 16. Exhaustive over a small discrete space, and only a small one.

    The size is computed before anything is enumerated and refused above a
    configured ceiling. A grid of 4 x 3 x 5 is sixty points, which at roughly
    52 minutes each is over two days; without the check that would be discovered
    by starting it.
    """

    name = SearchStrategyName.GRID_SEARCH

    def __init__(self, *, steps: int = 5, max_points: int = 64) -> None:
        super().__init__()
        self.steps = steps
        self.max_points = max_points
        self._queue: list[dict[str, float]] = []

    def initialize(self, space: ContentSearchSpace, seed: int) -> None:
        super().initialize(space, seed)
        size = space.cardinality(self.steps)
        if size > self.max_points:
            raise ValueError(
                f"this grid would have {size:.0f} points at {self.steps} steps per "
                f"parameter, above the limit of {self.max_points}. At roughly an hour "
                f"per evaluation that is {size / 24:.1f} days of compute. Reduce the "
                f"steps, shrink the space, or choose a sampling strategy."
            )
        self._queue = [point for point in space.grid(self.steps) if not space.is_neutral(point)]

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        out: list[CandidateGenome] = []
        while self._queue and len(out) < count:
            genome = self._from_values(
                self._queue.pop(0), origin=GenomeOrigin.GRID_POINT, generation=generation
            )
            if genome is not None:
                out.append(genome)
        if not self._queue:
            self._stop_reason = "every point in the grid has been proposed"
        return tuple(out)

    def get_state(self) -> dict[str, object]:
        return {**super().get_state(), "queue": self._queue}

    def set_state(self, state: dict[str, object]) -> None:
        super().set_state(state)
        queue = state.get("queue")
        if isinstance(queue, list):
            self._queue = [dict(item) for item in queue]


class LocalSearch(_Base):
    """Step 17. Explore around the best point found so far.

    Neighbours are axis-aligned and deterministic: one parameter moves at a
    time, so when a neighbour scores better exactly one thing differs and the
    move can be attributed. Varying everything at once would produce points
    nobody could explain.

    When a neighbourhood is exhausted it restarts from a fresh random point
    rather than stopping. That is the difference from hill climbing, and it is
    what lets it leave a local optimum: the algorithm keeps refining locally but
    is not confined to the basin it started in. Stopping is left to the
    orchestrator through patience and budget.
    """

    name = SearchStrategyName.LOCAL_SEARCH

    def __init__(self, *, magnitude: MutationMagnitude = MutationMagnitude.SMALL) -> None:
        super().__init__()
        self.magnitude = magnitude
        self._queue: list[dict[str, float]] = []
        #: Set when a round had to restart rather than refine. Hill climbing
        #: reads it, because for that algorithm it means a local optimum.
        self._neighbourhood_exhausted = False

    def _seed_point(self, generation: int) -> CandidateGenome | None:
        """Where to start when nothing has been evaluated yet."""
        for _ in range(32):
            genome = self._from_values(
                self.space.sample(self._rng),
                origin=GenomeOrigin.RANDOM_SAMPLE,
                generation=generation,
            )
            if genome is not None:
                return genome
        return None

    def _refill(self) -> None:
        best = self.best
        if best is None:
            return
        self._queue = [
            point
            for point in neighbours(best.genome, self.space, magnitude=self.magnitude)
            if not self.space.is_neutral(point)
        ]

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        out: list[CandidateGenome] = []
        if self.best is None:
            genome = self._seed_point(generation)
            if genome is not None:
                out.append(genome)
            if len(out) >= count:
                return tuple(out)

        attempts = 0
        exhausted = False
        while len(out) < count and attempts < 8:
            if not self._queue:
                self._refill()
                attempts += 1
                if not self._queue:
                    exhausted = True
                    break
            best = self.best
            parents = (best.genome.genome_id,) if best is not None else ()
            genome = self._from_values(
                self._queue.pop(0),
                origin=GenomeOrigin.NEIGHBOUR,
                generation=generation,
                parents=parents,
            )
            if genome is not None:
                out.append(genome)
            elif not self._queue:
                exhausted = True

        if len(out) < count and exhausted:
            self._neighbourhood_exhausted = True
            while len(out) < count:
                restart = self._seed_point(generation)
                if restart is None:
                    self._stop_reason = (
                        "the neighbourhood is exhausted and no fresh point remains "
                        "to restart from"
                    )
                    break
                out.append(restart)
        return tuple(out)

    def observe(self, results: Sequence[CandidateFitness]) -> None:
        previous = self.best
        super().observe(results)
        # A new best moves the neighbourhood, so anything queued around the old
        # one is stale and would explore a region already superseded.
        if self.best is not None and (
            previous is None or self.best.candidate_id != previous.candidate_id
        ):
            self._queue = []


class HillClimbing(LocalSearch):
    """Step 18. Local search that stops when a round brings no improvement.

    Step 19's limitation is real and is the reason the other strategies exist:
    hill climbing goes uphill from wherever it started and cannot leave a local
    optimum. On a multi-modal surface it will confidently return a point that is
    only the best in its own neighbourhood, and nothing in its output would
    distinguish that from a global best.
    """

    name = SearchStrategyName.HILL_CLIMBING

    def __init__(
        self,
        *,
        magnitude: MutationMagnitude = MutationMagnitude.SMALL,
        minimum_improvement: float = 0.0,
    ) -> None:
        super().__init__(magnitude=magnitude)
        self.minimum_improvement = minimum_improvement
        self._plateau = False

    def observe(self, results: Sequence[CandidateFitness]) -> None:
        previous = self.best
        super().observe(results)
        current = self.best
        if previous is None or current is None:
            return
        improved = current.improvement_over(previous)
        if current.candidate_id == previous.candidate_id or (
            improved is not None and improved <= self.minimum_improvement
        ):
            self._plateau = True

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        """Never restart: being unable to improve locally is the stop condition."""
        if self._plateau or self._neighbourhood_exhausted:
            return ()
        return super().propose(count, generation)

    def should_stop(self) -> tuple[bool, str]:
        """Exhausting the neighbourhood and plateauing are the same thing here.

        Both mean no nearby point improves on the current one, which is exactly
        what reaching a local optimum is. Reporting the mechanical reason
        instead would hide the thing a reader needs to know.
        """
        if self._plateau or self._neighbourhood_exhausted or self._stop_reason:
            return True, (
                f"no neighbour improved on the current best by more than "
                f"{self.minimum_improvement:g}; hill climbing has reached a local "
                f"optimum, which may not be the global one"
            )
        return False, ""

    def get_state(self) -> dict[str, object]:
        return {**super().get_state(), "plateau": self._plateau}

    def set_state(self, state: dict[str, object]) -> None:
        super().set_state(state)
        self._plateau = bool(state.get("plateau", False))


class BeamSearch(_Base):
    """Steps 20-21. Keep the top K and expand all of them.

    One retained point is hill climbing and cannot escape a local optimum; every
    point retained is exhaustive and unaffordable. A beam is the middle: enough
    diversity that a second promising region survives pruning, at a cost that
    stays proportional to the width rather than to the space.

    Pruning decisions are recorded so a report can say why a candidate was
    dropped rather than leaving it to be inferred from absence.
    """

    name = SearchStrategyName.BEAM_SEARCH

    def __init__(
        self,
        *,
        beam_width: int = 3,
        magnitude: MutationMagnitude = MutationMagnitude.MEDIUM,
        registry: MutationRegistry | None = None,
    ) -> None:
        super().__init__()
        if beam_width < 1:
            raise ValueError("beam width must be at least 1")
        self.beam_width = beam_width
        self.magnitude = magnitude
        self.registry = registry or MutationRegistry()
        self._beam: list[CandidateFitness] = []
        self.pruned: list[tuple[str, str]] = []

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        out: list[CandidateGenome] = []
        if not self._beam:
            for _ in range(count):
                genome = self._from_values(
                    self.space.sample(self._rng),
                    origin=GenomeOrigin.RANDOM_SAMPLE,
                    generation=generation,
                )
                if genome is not None:
                    out.append(genome)
            return tuple(out)

        # Expand the beam round-robin so every retained point gets a descendant
        # before any gets a second one.
        index = 0
        attempts = 0
        while len(out) < count and attempts < count * 8:
            attempts += 1
            parent = self._beam[index % len(self._beam)]
            index += 1
            child = self.registry.mutate(
                parent.genome,
                self.space,
                self._rng,
                genome_id=self._next_id(),
                magnitude=self.magnitude,
                proposed_by=self.name.value,
            )
            if self._accept(child) is not None:
                out.append(child)
        return tuple(out)

    def observe(self, results: Sequence[CandidateFitness]) -> None:
        super().observe(results)
        pool = [item for item in [*self._beam, *results] if item.is_usable]
        unique: dict[str, CandidateFitness] = {}
        for item in pool:
            unique.setdefault(item.candidate_id, item)
        ranked = sorted(
            unique.values(), key=lambda item: item.scalar_fitness or float("-inf"), reverse=True
        )
        kept = ranked[: self.beam_width]
        kept_ids = {item.candidate_id for item in kept}
        for position, item in enumerate(ranked[self.beam_width :], start=self.beam_width + 1):
            if item.candidate_id not in kept_ids:
                self.pruned.append(
                    (
                        item.candidate_id,
                        f"rank {position} of {len(ranked)}; beam width {self.beam_width}",
                    )
                )
        self._beam = kept

    @property
    def beam(self) -> tuple[CandidateFitness, ...]:
        return tuple(self._beam)

    def get_state(self) -> dict[str, object]:
        return {
            **super().get_state(),
            "beam_ids": [item.candidate_id for item in self._beam],
            "pruned": [list(item) for item in self.pruned],
        }


class EpsilonGreedy(_Base):
    """Steps 24-27. Exploit near the best, explore elsewhere, on a schedule.

    With probability 1-e it perturbs the best point found so far; with
    probability e it samples somewhere unrelated. The rate decays by generation
    so the search widens early and narrows later, and never falls below a floor,
    because a search that stops exploring entirely cannot leave a local optimum
    it has already reached.

    Every draw comes from the seeded generator, so the sequence of explore and
    exploit decisions is reproducible.
    """

    name = SearchStrategyName.EPSILON_GREEDY

    def __init__(
        self,
        *,
        exploration_rate: float = 0.3,
        decay: float = 1.0,
        minimum_rate: float = 0.05,
        magnitude: MutationMagnitude = MutationMagnitude.MEDIUM,
        registry: MutationRegistry | None = None,
    ) -> None:
        super().__init__()
        self.exploration_rate = exploration_rate
        self.decay = decay
        self.minimum_rate = minimum_rate
        self.magnitude = magnitude
        self.registry = registry or MutationRegistry()
        self.decisions: list[str] = []

    def rate_at(self, generation: int) -> float:
        return max(self.minimum_rate, self.exploration_rate * (self.decay**generation))

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        rate = self.rate_at(generation)
        out: list[CandidateGenome] = []
        attempts = 0
        while len(out) < count and attempts < count * 8:
            attempts += 1
            best = self.best
            explore = best is None or self._rng.random() < rate
            if explore:
                genome = self._from_values(
                    self.space.sample(self._rng),
                    origin=GenomeOrigin.RANDOM_SAMPLE,
                    generation=generation,
                )
                decision = "explore"
            else:
                assert best is not None
                child = self.registry.mutate(
                    best.genome,
                    self.space,
                    self._rng,
                    genome_id=self._next_id(),
                    magnitude=self.magnitude,
                    proposed_by=self.name.value,
                )
                genome = self._accept(child)
                decision = "exploit"
            if genome is not None:
                out.append(genome)
                self.decisions.append(f"{genome.genome_id}:{decision}@e={rate:.3f}")
        return tuple(out)

    def get_state(self) -> dict[str, object]:
        return {**super().get_state(), "decisions": list(self.decisions)}


class EvolutionarySearch(_Base):
    """Steps 31-35. A population that is selected, recombined and mutated.

    Built last and deliberately: it has the most knobs and the least
    interpretable path, so it is only worth running once the simpler strategies
    have shown what the surface looks like. Elitism keeps the best point
    unchanged across generations, because a population that can lose its best
    answer to chance is worse than one that cannot.

    Diversity is checked but does not override fitness unless configured to,
    since a population selected mainly for novelty stops optimising.
    """

    name = SearchStrategyName.EVOLUTIONARY_SEARCH

    def __init__(
        self,
        *,
        population_size: int = 6,
        elite_count: int = 1,
        crossover_rate: float = 0.5,
        magnitude: MutationMagnitude = MutationMagnitude.MEDIUM,
        registry: MutationRegistry | None = None,
    ) -> None:
        super().__init__()
        if elite_count >= population_size:
            raise ValueError("elite_count must leave room for at least one new candidate")
        self.population_size = population_size
        self.elite_count = elite_count
        self.crossover_rate = crossover_rate
        self.magnitude = magnitude
        self.registry = registry or MutationRegistry()
        self._population: list[CandidateFitness] = []

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        out: list[CandidateGenome] = []
        if len(self._population) < 2:
            for _ in range(count):
                genome = self._from_values(
                    self.space.sample(self._rng),
                    origin=GenomeOrigin.RANDOM_SAMPLE,
                    generation=generation,
                )
                if genome is not None:
                    out.append(genome)
            return tuple(out)

        attempts = 0
        while len(out) < count and attempts < count * 8:
            attempts += 1
            left, right = self._select_two()
            recombine = (
                self._rng.random() < self.crossover_rate
                and left.candidate_id != right.candidate_id
            )
            if recombine:
                child = crossover(
                    left.genome,
                    right.genome,
                    self.space,
                    self._rng,
                    genome_id=self._next_id(),
                    proposed_by=self.name.value,
                )
            else:
                child = self.registry.mutate(
                    left.genome,
                    self.space,
                    self._rng,
                    genome_id=self._next_id(),
                    magnitude=self.magnitude,
                    proposed_by=self.name.value,
                )
            if self._accept(child) is not None:
                out.append(child)
        return tuple(out)

    def _select_two(self) -> tuple[CandidateFitness, CandidateFitness]:
        """Binary tournament: pick two at random, keep the fitter.

        Rank-free and scale-free, so it behaves the same whether fitness values
        are near zero or in the thousands, and it cannot be dominated by one
        outlier the way roulette selection can.
        """
        def tournament() -> CandidateFitness:
            left, right = self._rng.choice(self._population), self._rng.choice(self._population)
            return left if left.is_better_than(right) else right

        first = tournament()
        # Crossover needs two distinct parents; a genome cannot list the same
        # parent twice, and a child of one parent is a mutation, not a
        # recombination.
        for _ in range(8):
            second = tournament()
            if second.candidate_id != first.candidate_id:
                return first, second
        alternatives = [
            item for item in self._population if item.candidate_id != first.candidate_id
        ]
        return (first, self._rng.choice(alternatives)) if alternatives else (first, first)

    def observe(self, results: Sequence[CandidateFitness]) -> None:
        super().observe(results)
        pool = [item for item in [*self._population, *results] if item.is_usable]
        unique: dict[str, CandidateFitness] = {}
        for item in pool:
            unique.setdefault(item.candidate_id, item)
        ranked = sorted(
            unique.values(), key=lambda item: item.scalar_fitness or float("-inf"), reverse=True
        )
        # Elitism first, then the rest by fitness, up to the population size.
        self._population = ranked[: self.population_size]

    @property
    def population(self) -> tuple[CandidateFitness, ...]:
        return tuple(self._population)

    @property
    def elites(self) -> tuple[CandidateFitness, ...]:
        return tuple(self._population[: self.elite_count])

    def get_state(self) -> dict[str, object]:
        return {
            **super().get_state(),
            "population_ids": [item.candidate_id for item in self._population],
        }


def _as_int(value: object, default: int) -> int:
    return int(value) if isinstance(value, int | float | str) else default


def _encode_rng(rng: Random) -> list[object]:
    state = rng.getstate()
    return [state[0], list(state[1]), state[2]]


def _decode_rng(state: object) -> tuple[int, tuple[int, ...], float | None]:
    if not isinstance(state, list | tuple) or len(state) != 3:
        raise ValueError("unreadable random generator state")
    version, internal, gauss = state
    if not isinstance(internal, list | tuple):
        raise ValueError("unreadable random generator state")
    return (int(version), tuple(int(v) for v in internal), gauss)


__all__ = [
    "BeamSearch",
    "EpsilonGreedy",
    "EvolutionarySearch",
    "GridSearch",
    "HillClimbing",
    "LocalSearch",
]
