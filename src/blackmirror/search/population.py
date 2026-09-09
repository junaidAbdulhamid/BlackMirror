"""Steps 22-23, 32-35: populations, diversity, and successive elimination.

WHY SUCCESSIVE HALVING IS ELIMINATION, NOT PARTIAL EVALUATION
    The textbook version saves compute by evaluating many candidates cheaply and
    only the survivors expensively. That requires a valid low-fidelity signal,
    and this pipeline has none: there is no shortened TRIBE pass whose ranking
    has been shown to agree with a full one. Inventing one would produce
    confident eliminations based on a number nobody validated.

    So halving here eliminates *between* generations on full evaluations. That
    is honest and still useful, because it concentrates the remaining budget on
    the survivors rather than spreading it evenly. It saves nothing on the
    candidates already run, and the docstring says so rather than implying a
    speed-up that does not exist.

WHY DIVERSITY IS MEASURED BUT DOES NOT RULE
    A population that collapses onto one point stops searching and starts
    re-measuring. Tracking spread makes that visible. Letting novelty outrank
    fitness, though, optimises for being different rather than for the
    objective, so diversity only breaks ties unless the caller explicitly asks
    for more.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.search.fitness import CandidateFitness
from blackmirror.search.genome import genome_distance
from blackmirror.search.space import ContentSearchSpace


class EliminationReason(StrEnum):
    OUTSIDE_BEAM = "outside_beam"
    HALVED = "halved"
    GUARDRAIL = "guardrail"
    NOT_MEASURABLE = "not_measurable"
    FAILED = "failed"


class PopulationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Elimination(PopulationModel):
    """Why one candidate did not survive (Step 71)."""

    candidate_id: str
    reason: EliminationReason
    detail: str
    rank: int | None = None
    of: int | None = None


class SearchPopulation(PopulationModel):
    """One generation's membership and what happened to it (Step 32)."""

    generation: int = Field(ge=0)
    candidate_ids: tuple[str, ...] = ()
    survivors: tuple[str, ...] = ()
    eliminated: tuple[Elimination, ...] = ()
    #: Mean pairwise genome distance, on [0, 1]. None below two members.
    diversity: float | None = None
    best_candidate_id: str | None = None
    best_fitness: float | None = None
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def has_collapsed(self) -> bool:
        """Whether the population has converged onto effectively one point."""
        return self.diversity is not None and self.diversity < 0.01


def population_diversity(
    members: list[CandidateFitness], space: ContentSearchSpace
) -> float | None:
    """Mean pairwise distance. `None` below two members, where it is undefined.

    Returning 0 for a single member would read as "fully collapsed", which is a
    claim about a population that does not exist yet.
    """
    if len(members) < 2:
        return None
    total = 0.0
    pairs = 0
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            total += genome_distance(left.genome, right.genome, space)
            pairs += 1
    return total / pairs if pairs else None


def rank_candidates(members: list[CandidateFitness]) -> list[CandidateFitness]:
    """Eligible candidates, best first. Ineligible ones are not ranked."""
    return sorted(
        (item for item in members if item.is_eligible),
        key=lambda item: item.scalar_fitness or float("-inf"),
        reverse=True,
    )


def build_population(
    generation: int,
    members: list[CandidateFitness],
    space: ContentSearchSpace,
    *,
    keep: int,
    reason: EliminationReason = EliminationReason.OUTSIDE_BEAM,
) -> SearchPopulation:
    """Rank, keep the top `keep`, and record why each other was dropped."""
    ranked = rank_candidates(members)
    survivors = ranked[:keep]
    survivor_ids = {item.candidate_id for item in survivors}
    eliminated: list[Elimination] = []

    for position, item in enumerate(ranked, start=1):
        if item.candidate_id in survivor_ids:
            continue
        eliminated.append(
            Elimination(
                candidate_id=item.candidate_id,
                reason=reason,
                detail=(
                    f"rank {position} of {len(ranked)}; {keep} retained"
                ),
                rank=position,
                of=len(ranked),
            )
        )

    for item in members:
        if item.is_eligible:
            continue
        if not item.is_usable:
            eliminated.append(
                Elimination(
                    candidate_id=item.candidate_id,
                    reason=(
                        EliminationReason.FAILED
                        if item.status.value == "evaluation_failed"
                        else EliminationReason.NOT_MEASURABLE
                    ),
                    detail=item.reason,
                )
            )
        else:
            eliminated.append(
                Elimination(
                    candidate_id=item.candidate_id,
                    reason=EliminationReason.GUARDRAIL,
                    detail=item.feasibility.summary,
                )
            )

    best = survivors[0] if survivors else None
    return SearchPopulation(
        generation=generation,
        candidate_ids=tuple(item.candidate_id for item in members),
        survivors=tuple(item.candidate_id for item in survivors),
        eliminated=tuple(eliminated),
        diversity=population_diversity(survivors, space),
        best_candidate_id=None if best is None else best.candidate_id,
        best_fitness=None if best is None else best.scalar_fitness,
    )


def successive_halving(
    members: list[CandidateFitness],
    space: ContentSearchSpace,
    *,
    generation: int,
    keep_fraction: float = 0.5,
    minimum: int = 1,
) -> SearchPopulation:
    """Keep the better half (Step 22).

    This eliminates on full evaluations already paid for. It does not make those
    evaluations cheaper, and no partial-fidelity score is fabricated to pretend
    otherwise. What it buys is that the *next* generation's budget goes to the
    survivors instead of being spread across everything.
    """
    if not 0 < keep_fraction < 1:
        raise ValueError("keep_fraction must be strictly between 0 and 1")
    eligible = [item for item in members if item.is_eligible]
    keep = max(minimum, int(len(eligible) * keep_fraction))
    return build_population(
        generation, members, space, keep=keep, reason=EliminationReason.HALVED
    )


def diversity_aware_selection(
    members: list[CandidateFitness],
    space: ContentSearchSpace,
    *,
    count: int,
    novelty_weight: float = 0.0,
) -> list[CandidateFitness]:
    """Pick `count` candidates by fitness, optionally spread out (Step 35).

    With `novelty_weight` at zero this is a plain fitness ranking. Above zero it
    greedily prefers candidates far from those already chosen, scaled so that
    fitness still dominates unless the caller deliberately raises the weight.
    Novelty is normalized against the fitness spread of the pool so the two
    terms are commensurable rather than being added in different units.
    """
    if not 0.0 <= novelty_weight <= 1.0:
        raise ValueError("novelty_weight is a fraction between 0 and 1")
    ranked = rank_candidates(members)
    if novelty_weight == 0.0 or len(ranked) <= 1:
        return ranked[:count]

    scores = [item.scalar_fitness or 0.0 for item in ranked]
    spread = max(scores) - min(scores)
    chosen: list[CandidateFitness] = [ranked[0]]
    pool = ranked[1:]

    while pool and len(chosen) < count:
        def combined(item: CandidateFitness) -> float:
            fitness = item.scalar_fitness or 0.0
            novelty = min(
                genome_distance(item.genome, picked.genome, space) for picked in chosen
            )
            # Novelty is on [0, 1]; put fitness on the same scale before mixing.
            scaled = 0.0 if spread == 0 else (fitness - min(scores)) / spread
            return (1.0 - novelty_weight) * scaled + novelty_weight * novelty

        best = max(pool, key=combined)
        chosen.append(best)
        pool.remove(best)
    return chosen


__all__ = [
    "Elimination",
    "EliminationReason",
    "SearchPopulation",
    "build_population",
    "diversity_aware_selection",
    "population_diversity",
    "rank_candidates",
    "successive_halving",
]
