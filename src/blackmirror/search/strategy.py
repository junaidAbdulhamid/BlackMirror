"""Steps 14-15, 48-49: the strategy interface, a registry, and random search.

WHY AN INTERFACE FIRST
    Every algorithm in this phase does the same four things: set up, propose the
    next candidates, look at what came back, and say whether it is finished.
    Naming that explicitly is what lets the orchestrator be written once and the
    algorithms be swapped, compared and benchmarked against each other. The
    alternative, a growing conditional inside the loop, would make the
    comparison in Step 87 impossible to run fairly.

WHY RANDOM SEARCH IS FIRST
    It is the baseline. A smarter strategy is only worth its complexity if it
    beats random sampling on the same budget, space and objective, and without
    random search built there is nothing to make that claim against. It is also
    the honest control: on a space where the objective is flat or the signal is
    below the noise floor, random search will match anything cleverer, and that
    is a result worth being able to see.

WHY PROPOSALS ARE SEEDED AND STATEFUL
    A search must be reproducible and resumable. The generator lives on the
    strategy and its state is exported with `get_state`, so a run interrupted at
    evaluation seven resumes proposing the eighth candidate it would have
    proposed, not a fresh one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from random import Random
from typing import Protocol, runtime_checkable

from blackmirror.search.fitness import CandidateFitness
from blackmirror.search.genome import CandidateGenome, GenomeOrigin
from blackmirror.search.schemas import SearchStrategyName
from blackmirror.search.space import ContentSearchSpace


@runtime_checkable
class SearchStrategy(Protocol):
    """What every search algorithm must provide (Step 48)."""

    name: SearchStrategyName

    def initialize(self, space: ContentSearchSpace, seed: int) -> None:
        """Bind to a space and a seed. Called once before any proposal."""
        ...

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        """Up to `count` candidates. May return fewer, never more."""
        ...

    def observe(self, results: Sequence[CandidateFitness]) -> None:
        """Take in evaluated candidates, including failures."""
        ...

    def should_stop(self) -> tuple[bool, str]:
        """Whether the strategy itself is finished, and why."""
        ...

    def get_state(self) -> dict[str, object]:
        """Everything needed to resume this strategy exactly."""
        ...

    def set_state(self, state: dict[str, object]) -> None:
        """Restore from `get_state`."""
        ...


class RandomSearch:
    """Samples valid points uniformly from the space (Steps 14-15).

    Holds no model of the objective and learns nothing from results, which is
    the point: it measures how much of any other strategy's performance comes
    from the shape of the space rather than from the strategy.
    """

    name = SearchStrategyName.RANDOM_SEARCH

    def __init__(self, *, max_resample_attempts: int = 32) -> None:
        self._space: ContentSearchSpace | None = None
        self._rng = Random(0)
        self._seed = 0
        self._proposed = 0
        self._seen: set[str] = set()
        self._observed = 0
        self._exhausted = False
        # A space can be small enough that fresh points run out. Re-drawing
        # forever would hang the search, so give up after a bounded number of
        # attempts and report the space as exhausted.
        self.max_resample_attempts = max_resample_attempts

    # --- lifecycle --------------------------------------------------------

    def initialize(self, space: ContentSearchSpace, seed: int) -> None:
        self._space = space
        self._seed = seed
        self._rng = Random(seed)
        self._proposed = 0
        self._seen = set()
        self._observed = 0
        self._exhausted = False

    @property
    def space(self) -> ContentSearchSpace:
        if self._space is None:
            raise RuntimeError("strategy used before initialize()")
        return self._space

    # --- proposing --------------------------------------------------------

    def propose(self, count: int, generation: int) -> tuple[CandidateGenome, ...]:
        out: list[CandidateGenome] = []
        for _ in range(max(0, count)):
            genome = self._draw(generation)
            if genome is None:
                self._exhausted = True
                break
            out.append(genome)
        return tuple(out)

    def _draw(self, generation: int) -> CandidateGenome | None:
        """One point not already proposed, or None if the space is used up.

        Duplicates are filtered here rather than after evaluation, because the
        whole cost of a duplicate is the evaluation it would trigger.
        """
        space = self.space
        for _ in range(self.max_resample_attempts):
            values = space.sample(self._rng)
            # A point at every neutral value would rebuild the root, whose
            # score is already known.
            if space.is_neutral(values):
                continue
            genome = CandidateGenome(
                genome_id=f"c{self._proposed + 1:04d}",
                values=values,
                origin=GenomeOrigin.RANDOM_SAMPLE,
                generation=generation,
                proposed_by=self.name.value,
            )
            fingerprint = genome.fingerprint(space)
            if fingerprint in self._seen:
                continue
            self._seen.add(fingerprint)
            self._proposed += 1
            return genome
        return None

    # --- observing --------------------------------------------------------

    def observe(self, results: Sequence[CandidateFitness]) -> None:
        """Random search learns nothing; it only counts what came back."""
        self._observed += len(results)

    def should_stop(self) -> tuple[bool, str]:
        if self._exhausted:
            return True, (
                "the space has no further distinct points to sample within the "
                "resample limit"
            )
        return False, ""

    # --- state ------------------------------------------------------------

    def get_state(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "seed": self._seed,
            "proposed": self._proposed,
            "observed": self._observed,
            "exhausted": self._exhausted,
            "seen": sorted(self._seen),
            # The generator's own state, so resuming continues the same stream
            # rather than restarting it and re-proposing earlier points.
            "rng_state": _encode_rng(self._rng),
        }

    def set_state(self, state: dict[str, object]) -> None:
        self._seed = _as_int(state.get("seed"), 0)
        self._proposed = _as_int(state.get("proposed"), 0)
        self._observed = _as_int(state.get("observed"), 0)
        self._exhausted = bool(state.get("exhausted", False))
        seen = state.get("seen")
        self._seen = {str(item) for item in seen} if isinstance(seen, list) else set()
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(_decode_rng(rng_state))


def _as_int(value: object, default: int) -> int:
    return int(value) if isinstance(value, int | float | str) else default


def _encode_rng(rng: Random) -> list[object]:
    """Random.getstate returns a tuple containing a tuple; JSON needs lists."""
    state = rng.getstate()
    return [state[0], list(state[1]), state[2]]


def _decode_rng(state: object) -> tuple[int, tuple[int, ...], float | None]:
    if not isinstance(state, list | tuple) or len(state) != 3:
        raise ValueError("unreadable random generator state")
    version, internal, gauss = state
    if not isinstance(internal, list | tuple):
        raise ValueError("unreadable random generator state")
    return (int(version), tuple(int(v) for v in internal), gauss)


class SearchStrategyRegistry:
    """Step 49. Named lookup, so the orchestrator holds no conditional chain."""

    def __init__(self, factories: dict[SearchStrategyName, object] | None = None) -> None:
        self._factories: dict[SearchStrategyName, object] = dict(
            factories or {SearchStrategyName.RANDOM_SEARCH: RandomSearch}
        )

    def register(self, name: SearchStrategyName, factory: object) -> None:
        self._factories[name] = factory

    def create(self, name: SearchStrategyName) -> SearchStrategy:
        factory = self._factories.get(name)
        if factory is None:
            available = ", ".join(sorted(item.value for item in self._factories))
            raise ValueError(
                f"search strategy {name.value!r} is not implemented yet; "
                f"available: {available or 'none'}"
            )
        return factory()  # type: ignore[operator]

    def available(self) -> tuple[SearchStrategyName, ...]:
        return tuple(sorted(self._factories, key=lambda item: item.value))

    def __contains__(self, name: object) -> bool:
        return name in self._factories


def default_registry() -> SearchStrategyRegistry:
    """Every strategy that actually exists.

    A registry advertising a planned-but-unbuilt algorithm would let a search be
    configured with one that silently does nothing, so entries are added here
    only once the algorithm behind them works.

    Imported inside the function because the concrete strategies import this
    module for the protocol.
    """
    from blackmirror.search.strategies import (
        BeamSearch,
        EpsilonGreedy,
        EvolutionarySearch,
        GridSearch,
        HillClimbing,
        LocalSearch,
    )

    return SearchStrategyRegistry(
        {
            SearchStrategyName.RANDOM_SEARCH: RandomSearch,
            SearchStrategyName.GRID_SEARCH: GridSearch,
            SearchStrategyName.LOCAL_SEARCH: LocalSearch,
            SearchStrategyName.HILL_CLIMBING: HillClimbing,
            SearchStrategyName.BEAM_SEARCH: BeamSearch,
            SearchStrategyName.EPSILON_GREEDY: EpsilonGreedy,
            SearchStrategyName.EVOLUTIONARY_SEARCH: EvolutionarySearch,
        }
    )


def unique_genomes(
    genomes: Iterable[CandidateGenome], space: ContentSearchSpace
) -> tuple[CandidateGenome, ...]:
    """Drop points already present in the batch, keeping the first of each."""
    seen: set[str] = set()
    out: list[CandidateGenome] = []
    for genome in genomes:
        fingerprint = genome.fingerprint(space)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(genome)
    return tuple(out)


__all__ = [
    "RandomSearch",
    "SearchStrategy",
    "SearchStrategyRegistry",
    "default_registry",
    "unique_genomes",
]
