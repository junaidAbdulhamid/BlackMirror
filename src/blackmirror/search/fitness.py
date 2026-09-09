"""Step 12: what one evaluated candidate is worth.

WHY FITNESS IS BUILT FROM RAW VALUES, NOT NORMALIZED SCORES
    Phase 6 reports both, and leads with raw for a reason: normalization can
    turn a negligible raw difference into a decisive-looking score. A search
    driven by the normalized number would chase exactly that amplification, and
    would happily rank candidates by a rescaling of noise.

    The improvement threshold is also defined in raw units. The measured
    nuisance floor of 0.0157 is a shift in whole-cortex mean, so a search that
    compared normalized scores could not tell whether a margin cleared it. Raw
    values keep the comparison and the floor in the same units.

    The normalized score is carried alongside, because it is what the Phase 6
    screens display and a reader will want both to agree.

WHY DIRECTION IS FOLDED IN
    Strategies hill-climb, rank and take maxima. All of that needs one
    convention: higher is better. So a MINIMIZE objective's raw value is
    negated and a TARGET objective's becomes the negative distance from its
    declared target. `scalar_fitness` is therefore always maximised, whatever
    the objective asked for, and `raw_objectives` keeps the untransformed
    measurement for reporting.

WHY A FAILURE IS A RESULT
    An evaluation that crashes must not end the search (Step 54). It returns a
    fitness with a status and a reason instead of raising, so the budget records
    what was spent, the trajectory records the attempt, and the strategy learns
    that this region is expensive rather than silently never hearing back.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from blackmirror.scoring.schemas import NeuralObjective, ObjectiveDirection
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.guardrails import FeasibilityResult


class FitnessStatus(StrEnum):
    """Whether this candidate produced a usable number, and if not why."""

    EVALUATED = "evaluated"
    #: The pipeline ran but the objective had no valid value on one side.
    NOT_MEASURABLE = "not_measurable"
    #: Materialization, inference, or a later stage failed.
    EVALUATION_FAILED = "evaluation_failed"
    #: Refused before any compute: constraints, duplicate, or budget.
    REJECTED = "rejected"


def directional_fitness(
    raw_value: float | None, objective: NeuralObjective
) -> float | None:
    """Fold the objective's direction in so higher is always better.

    Returns `None` rather than a sentinel when there is no value, because a
    stand-in number would be ranked against real ones and could win.
    """
    if raw_value is None:
        return None
    if objective.direction is ObjectiveDirection.MAXIMIZE:
        return raw_value
    if objective.direction is ObjectiveDirection.MINIMIZE:
        return -raw_value
    target = objective.normalization.target_value
    if target is None:  # pragma: no cover - Phase 6 requires it for TARGET
        return None
    return -abs(raw_value - target)


class ComputeCost(BaseModel):
    """What this evaluation actually consumed."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    wall_seconds: float = Field(default=0.0, ge=0)
    tribe_runs: int = Field(default=0, ge=0)
    served_from_cache: bool = False


class CandidateFitness(BaseModel):
    """One evaluated candidate: what was measured and what it cost.

    Deliberately not called a quality, a rating or a score of the content. It is
    the value of one explicitly declared function of model-predicted cortical
    responses, and nothing else.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    candidate_id: str = Field(min_length=1)
    genome: CandidateGenome
    status: FitnessStatus
    #: Objective id -> the value as measured, direction untouched.
    raw_objectives: dict[str, float | None] = Field(default_factory=dict)
    #: Objective id -> Phase 6's normalized score, for display alongside.
    normalized_objectives: dict[str, float | None] = Field(default_factory=dict)
    #: Primary objective's value with direction folded in. Always maximised.
    scalar_fitness: float | None = None
    primary_objective_id: str | None = None
    #: Run produced by Phase 8, so a candidate can be opened in the Phase 2 UI.
    candidate_run_id: str | None = None
    resimulation_id: str | None = None
    variant_path: str | None = None
    variant_sha256: str | None = None
    duration_change_seconds: float = 0.0
    compute: ComputeCost = ComputeCost()
    #: Whether the candidate satisfied every declared guardrail. An infeasible
    #: candidate keeps its fitness but can never be selected as best.
    feasibility: FeasibilityResult = FeasibilityResult(feasible=True)
    #: Why it is not usable, when it is not. Never empty for a failure.
    reason: str = ""
    evaluated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @model_validator(mode="after")
    def _coherent(self) -> CandidateFitness:
        if self.status is FitnessStatus.EVALUATED:
            if self.scalar_fitness is None:
                raise ValueError(
                    "an evaluated candidate needs a scalar fitness; if the objective "
                    "could not be measured the status is not_measurable"
                )
            if self.primary_objective_id is None:
                raise ValueError("an evaluated candidate must name the objective it scored")
        elif not self.reason:
            raise ValueError(f"status {self.status.value} must carry a reason")
        return self

    @property
    def is_usable(self) -> bool:
        """Whether this candidate produced a number that can be compared."""
        return self.status is FitnessStatus.EVALUATED and self.scalar_fitness is not None

    @property
    def is_feasible(self) -> bool:
        return self.feasibility.feasible

    @property
    def is_eligible(self) -> bool:
        """Whether this candidate may be *selected* as best.

        Separate from `is_usable` because an infeasible candidate is still a
        real measurement worth recording and learning from; it simply cannot
        win. Conflating the two would either discard the measurement or let a
        candidate that breached a guardrail be reported as the result.
        """
        return self.is_usable and self.is_feasible

    def improvement_over(self, other: CandidateFitness | None) -> float | None:
        """Directional gain against a reference. `None` if either is unusable."""
        if not self.is_usable or other is None or not other.is_usable:
            return None
        assert self.scalar_fitness is not None and other.scalar_fitness is not None
        return self.scalar_fitness - other.scalar_fitness

    def is_better_than(self, other: CandidateFitness | None) -> bool:
        """Plain comparison, used to track which candidate is best.

        Deliberately has no threshold. Gating *promotion* on the noise floor
        makes the recorded best depend on the order candidates happened to be
        evaluated in: a strong candidate arriving second would fail to displace
        a weaker first one, and a different seed would report a different
        winner among identical numbers. The best is therefore the plain
        argmax, and the question of whether its lead means anything is asked
        separately, by `beats` and `within_noise_of`.
        """
        if not self.is_eligible:
            return False
        if other is None or not other.is_eligible:
            return True
        assert self.scalar_fitness is not None and other.scalar_fitness is not None
        return self.scalar_fitness > other.scalar_fitness

    def beats(self, other: CandidateFitness | None, *, minimum_improvement: float) -> bool:
        """Whether this is better than `other` by a margin worth believing.

        This is the question that drives stopping and patience, and the one a
        report must answer before claiming an improvement. A candidate ahead by
        less than the pipeline's own sensitivity to nuisance choices has not
        been shown to be better at all.
        """
        if not self.is_usable:
            return False
        if other is None or not other.is_usable:
            return True
        gain = self.improvement_over(other)
        return gain is not None and gain > minimum_improvement

    def within_noise_of(
        self, other: CandidateFitness | None, *, minimum_improvement: float
    ) -> bool:
        """Whether these two are too close to tell apart."""
        if not self.is_usable or other is None or not other.is_usable:
            return False
        gain = self.improvement_over(other)
        return gain is not None and abs(gain) <= minimum_improvement


def best_of(results: Iterable[CandidateFitness]) -> CandidateFitness | None:
    """The highest-scoring *eligible* candidate, or None if there is none.

    Eligibility, not merely a number: a candidate that breached a hard
    guardrail cannot be the answer however well it scored (Steps 61-62). It
    stays in the record and the strategy still learns from it.

    Order-independent by construction, so re-running a search with the same
    seed and the same results reports the same winner regardless of the order
    they came back in.
    """
    eligible = [item for item in results if item.is_eligible]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item.scalar_fitness or float("-inf"))


def best_infeasible(results: Iterable[CandidateFitness]) -> CandidateFitness | None:
    """The strongest candidate that was disqualified, for reporting.

    Worth surfacing: "the highest score belonged to a candidate that broke a
    guardrail" is something a user needs told, not silently dropped.
    """
    blocked = [item for item in results if item.is_usable and not item.is_feasible]
    if not blocked:
        return None
    return max(blocked, key=lambda item: item.scalar_fitness or float("-inf"))


def tied_with_best(
    results: Iterable[CandidateFitness], *, minimum_improvement: float
) -> tuple[CandidateFitness, ...]:
    """Candidates the best is not measurably ahead of, best excluded.

    A non-empty result is the honest caveat on any claim of a winner: those
    candidates scored within the pipeline's own sensitivity to nuisance
    choices, so the ranking between them is not a finding. A report that named
    a single best without this would overstate what the search established.
    """
    materialized = list(results)
    best = best_of(materialized)
    if best is None:
        return ()
    return tuple(
        item
        for item in materialized
        if item.is_usable
        and item.candidate_id != best.candidate_id
        and best.within_noise_of(item, minimum_improvement=minimum_improvement)
    )


def rejected(
    genome: CandidateGenome, reason: str, *, candidate_id: str | None = None
) -> CandidateFitness:
    """A candidate refused before any compute was spent on it."""
    return CandidateFitness(
        candidate_id=candidate_id or genome.genome_id,
        genome=genome,
        status=FitnessStatus.REJECTED,
        reason=reason,
    )


def failed(
    genome: CandidateGenome,
    reason: str,
    *,
    candidate_id: str | None = None,
    compute: ComputeCost | None = None,
    resimulation_id: str | None = None,
) -> CandidateFitness:
    """A candidate whose evaluation started and did not produce a value."""
    return CandidateFitness(
        candidate_id=candidate_id or genome.genome_id,
        genome=genome,
        status=FitnessStatus.EVALUATION_FAILED,
        reason=reason,
        compute=compute or ComputeCost(),
        resimulation_id=resimulation_id,
    )


__all__ = [
    "CandidateFitness",
    "ComputeCost",
    "FitnessStatus",
    "best_infeasible",
    "best_of",
    "directional_fitness",
    "failed",
    "rejected",
    "tied_with_best",
]
