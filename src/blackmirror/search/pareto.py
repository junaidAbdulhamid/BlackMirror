"""Steps 58-60: search over several objectives without collapsing them to one.

WHY NOT JUST WEIGHT THEM
    Weighting turns a multi-objective problem into a single-objective one by
    asserting an exchange rate: so much stability is worth so much response.
    That rate is a value judgement, it is usually invented rather than measured,
    and Phase 6 already showed how easily a ranking flips within the range of
    weights a user might plausibly have chosen.

    The Pareto frontier needs no such assertion. It answers "which candidates
    are worth considering at all" by removing only those that are beaten on
    every objective at once, and leaves the trade-off to a person.

WHY DOMINANCE IS COMPUTED ON DIRECTIONAL FITNESS
    Phase 6's `analyze_pareto` compares normalized scores where higher is
    better. Here each objective's raw value is put through the same direction
    folding used for scalar fitness, so a MINIMIZE objective's dominance
    behaves correctly without depending on normalization, which can amplify a
    negligible raw difference.

WHY THE RANKING IS SHALLOW
    Non-dominated sorting into fronts, as NSGA-II does, is implemented here and
    tested against a known frontier. What is not implemented is crowding
    distance and the rest of NSGA-II's selection machinery: at budgets of a
    handful of candidates it would be elaborate scaffolding around too few
    points to mean anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.scoring.schemas import NeuralObjective
from blackmirror.search.fitness import CandidateFitness, directional_fitness


class ParetoModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ParetoFront(ParetoModel):
    """The non-dominated set, and what it excluded."""

    objective_ids: tuple[str, ...]
    non_dominated: tuple[str, ...] = ()
    dominated: tuple[str, ...] = ()
    #: candidate -> the candidates that dominate it.
    dominated_by: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    #: Candidates left out because an objective had no value for them.
    excluded: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.non_dominated)


def directional_vector(
    fitness: CandidateFitness, objectives: Sequence[NeuralObjective]
) -> tuple[float, ...] | None:
    """Each objective folded so higher is better, or None if any is missing."""
    values: list[float] = []
    for objective in objectives:
        folded = directional_fitness(
            fitness.raw_objectives.get(objective.objective_id), objective
        )
        if folded is None:
            return None
        values.append(folded)
    return tuple(values)


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    """True when `left` is at least as good everywhere and better somewhere."""
    if len(left) != len(right):
        raise ValueError("cannot compare objective vectors of different lengths")
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def pareto_front(
    results: Iterable[CandidateFitness], objectives: Sequence[NeuralObjective]
) -> ParetoFront:
    """The non-dominated set among eligible candidates.

    Infeasible candidates are excluded rather than ranked: a guardrail breach
    disqualifies a candidate from the answer, and a frontier is an answer.
    """
    objective_ids = tuple(item.objective_id for item in objectives)
    materialized = list(results)
    vectors: dict[str, tuple[float, ...]] = {}
    excluded: list[str] = []

    for item in materialized:
        if not item.is_eligible:
            excluded.append(item.candidate_id)
            continue
        vector = directional_vector(item, objectives)
        if vector is None:
            excluded.append(item.candidate_id)
            continue
        vectors[item.candidate_id] = vector

    warnings: list[str] = []
    if excluded:
        warnings.append(
            f"{len(excluded)} candidate(s) were excluded from the frontier because "
            f"they were infeasible or lacked a value on at least one objective"
        )

    dominated_by: dict[str, tuple[str, ...]] = {}
    for candidate, vector in vectors.items():
        dominators = tuple(
            other
            for other, other_vector in vectors.items()
            if other != candidate and dominates(other_vector, vector)
        )
        if dominators:
            dominated_by[candidate] = dominators

    return ParetoFront(
        objective_ids=objective_ids,
        non_dominated=tuple(sorted(c for c in vectors if c not in dominated_by)),
        dominated=tuple(sorted(dominated_by)),
        dominated_by=dominated_by,
        excluded=tuple(sorted(excluded)),
        warnings=tuple(warnings),
    )


def non_dominated_sort(
    results: Sequence[CandidateFitness], objectives: Sequence[NeuralObjective]
) -> tuple[tuple[str, ...], ...]:
    """Partition candidates into successive frontiers.

    Front 0 is the non-dominated set; front 1 is what becomes non-dominated
    once front 0 is removed, and so on. This is the ranking half of NSGA-II and
    is what a multi-objective selection uses in place of a scalar sort.
    """
    remaining = {
        item.candidate_id: vector
        for item in results
        if item.is_eligible and (vector := directional_vector(item, objectives)) is not None
    }
    fronts: list[tuple[str, ...]] = []
    while remaining:
        current = tuple(
            sorted(
                candidate
                for candidate, vector in remaining.items()
                if not any(
                    other != candidate and dominates(other_vector, vector)
                    for other, other_vector in remaining.items()
                )
            )
        )
        if not current:  # pragma: no cover - cycles are impossible under dominance
            break
        fronts.append(current)
        for candidate in current:
            del remaining[candidate]
    return tuple(fronts)


def select_by_pareto_rank(
    results: Sequence[CandidateFitness],
    objectives: Sequence[NeuralObjective],
    *,
    count: int,
) -> tuple[CandidateFitness, ...]:
    """Take the best `count`, front by front (Step 60).

    Within a front nothing dominates anything, so there is no principled order
    and ties are broken by candidate id. That is arbitrary but deterministic,
    which is the honest choice: inventing a preference inside a front would
    reintroduce the exchange rate the frontier exists to avoid.
    """
    by_id = {item.candidate_id: item for item in results}
    chosen: list[CandidateFitness] = []
    for front in non_dominated_sort(results, objectives):
        for candidate_id in front:
            if len(chosen) >= count:
                return tuple(chosen)
            chosen.append(by_id[candidate_id])
    return tuple(chosen)


__all__ = [
    "ParetoFront",
    "directional_vector",
    "dominates",
    "non_dominated_sort",
    "pareto_front",
    "select_by_pareto_rank",
]
