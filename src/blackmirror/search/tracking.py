"""Steps 39-46 and 63: best-so-far, trajectory, stopping, and the event log.

WHY THE TRAJECTORY IS THE HEADLINE ARTIFACT
    The interesting question about a search is not what it found but how much it
    cost to find it. A best-so-far curve against evaluation count is the plot
    that answers it, and it is the only honest basis for saying one strategy is
    more sample-efficient than another on this problem.

WHY STOPPING IS SEPARATE FROM THE STRATEGY
    A strategy knows when it has run out of ideas. It does not know the budget,
    the wall clock, or whether the user has asked it to stop. Keeping those
    apart means a strategy cannot accidentally overrule a limit, and the reason
    a search ended is recorded in one place rather than inferred.

WHY IMPROVEMENT IS JUDGED AGAINST A THRESHOLD HERE
    Patience counts generations without a *believable* gain. Counting any gain
    at all would keep a search alive indefinitely on drift smaller than the
    pipeline's own sensitivity, spending real hours chasing noise.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.search.budget import BudgetLedger, BudgetLimit, SearchBudget
from blackmirror.search.fitness import CandidateFitness, best_of
from blackmirror.search.schemas import SearchConfig, StoppingReason


class TrackingModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class SearchEventType(StrEnum):
    """Step 63. What happened, in the order it happened."""

    SEARCH_STARTED = "search_started"
    CANDIDATE_PROPOSED = "candidate_proposed"
    CANDIDATE_EVALUATED = "candidate_evaluated"
    CANDIDATE_REJECTED = "candidate_rejected"
    CANDIDATE_ELIMINATED = "candidate_eliminated"
    CANDIDATE_FAILED = "candidate_failed"
    NEW_BEST_FOUND = "new_best_found"
    GENERATION_COMPLETED = "generation_completed"
    BUDGET_WARNING = "budget_warning"
    STOPPING_CONDITION_MET = "stopping_condition_met"
    SEARCH_COMPLETED = "search_completed"


class SearchEvent(TrackingModel):
    type: SearchEventType
    message: str
    candidate_id: str | None = None
    generation: int = Field(default=0, ge=0)
    evaluation_number: int = Field(default=0, ge=0)
    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class TrajectoryPoint(TrackingModel):
    """One row of the best-so-far curve (Step 40)."""

    evaluation_number: int = Field(ge=0)
    generation: int = Field(ge=0)
    candidate_id: str
    fitness: float | None
    best_fitness: float
    best_candidate_id: str
    wall_seconds: float = Field(ge=0)
    tribe_runs: int = Field(ge=0)
    is_new_best: bool = False


class BestCandidateRecord(TrackingModel):
    """Step 39. The best so far, and what it had cost when it was found."""

    candidate_id: str
    fitness: float
    generation: int = Field(ge=0)
    evaluation_number: int = Field(ge=0)
    wall_seconds_at_discovery: float = Field(ge=0)
    tribe_runs_at_discovery: int = Field(ge=0)
    improvement_over_root: float | None = None


class SearchEfficiencyMetrics(TrackingModel):
    """Step 42. What the search cost, and whether it found anything real."""

    best_fitness: float | None = None
    root_fitness: float | None = None
    absolute_improvement: float | None = None
    evaluations: int = Field(default=0, ge=0)
    evaluations_to_best: int | None = None
    wall_seconds: float = Field(default=0.0, ge=0)
    wall_seconds_to_best: float | None = None
    tribe_runs: int = Field(default=0, ge=0)
    tribe_runs_to_best: int | None = None
    cache_hit_rate: float | None = None
    failure_rate: float | None = None
    #: Whether the winner's lead over the root clears the measured floor.
    improvement_is_resolvable: bool | None = None
    #: Candidates the winner is not measurably ahead of.
    tied_candidate_ids: tuple[str, ...] = ()


class SearchTracker:
    """Accumulates the trajectory, the best record and the event log."""

    def __init__(self, *, root_fitness: float | None = None) -> None:
        self.root_fitness = root_fitness
        self.results: list[CandidateFitness] = []
        self.trajectory: list[TrajectoryPoint] = []
        self.events: list[SearchEvent] = []
        self.best: BestCandidateRecord | None = None
        self._evaluations = 0

    def event(
        self,
        event_type: SearchEventType,
        message: str,
        *,
        candidate_id: str | None = None,
        generation: int = 0,
    ) -> SearchEvent:
        entry = SearchEvent(
            type=event_type,
            message=message,
            candidate_id=candidate_id,
            generation=generation,
            evaluation_number=self._evaluations,
        )
        self.events.append(entry)
        return entry

    def record(
        self, fitness: CandidateFitness, *, generation: int, ledger: BudgetLedger
    ) -> None:
        """Take one evaluated candidate into the record.

        The best is recomputed as an argmax over everything seen, not by
        comparing against the incumbent, so the winner never depends on the
        order results arrived in.
        """
        self.results.append(fitness)
        self._evaluations += 1

        if fitness.status.value == "evaluation_failed":
            self.event(
                SearchEventType.CANDIDATE_FAILED,
                f"{fitness.candidate_id}: {fitness.reason}",
                candidate_id=fitness.candidate_id,
                generation=generation,
            )
        elif fitness.status.value == "rejected":
            self.event(
                SearchEventType.CANDIDATE_REJECTED,
                f"{fitness.candidate_id}: {fitness.reason}",
                candidate_id=fitness.candidate_id,
                generation=generation,
            )
        else:
            self.event(
                SearchEventType.CANDIDATE_EVALUATED,
                f"{fitness.candidate_id} scored {fitness.scalar_fitness}",
                candidate_id=fitness.candidate_id,
                generation=generation,
            )

        winner = best_of(self.results)
        if winner is None:
            return
        assert winner.scalar_fitness is not None
        is_new_best = self.best is None or winner.candidate_id != self.best.candidate_id
        if is_new_best:
            self.best = BestCandidateRecord(
                candidate_id=winner.candidate_id,
                fitness=winner.scalar_fitness,
                generation=generation,
                evaluation_number=self._evaluations,
                wall_seconds_at_discovery=ledger.wall_seconds,
                tribe_runs_at_discovery=ledger.tribe_runs,
                improvement_over_root=(
                    None
                    if self.root_fitness is None
                    else winner.scalar_fitness - self.root_fitness
                ),
            )
            self.event(
                SearchEventType.NEW_BEST_FOUND,
                f"{winner.candidate_id} at {winner.scalar_fitness}",
                candidate_id=winner.candidate_id,
                generation=generation,
            )

        self.trajectory.append(
            TrajectoryPoint(
                evaluation_number=self._evaluations,
                generation=generation,
                candidate_id=fitness.candidate_id,
                fitness=fitness.scalar_fitness,
                best_fitness=winner.scalar_fitness,
                best_candidate_id=winner.candidate_id,
                wall_seconds=ledger.wall_seconds,
                tribe_runs=ledger.tribe_runs,
                is_new_best=is_new_best,
            )
        )

    @property
    def best_fitness(self) -> CandidateFitness | None:
        return best_of(self.results)

    def metrics(
        self, ledger: BudgetLedger, *, minimum_improvement: float
    ) -> SearchEfficiencyMetrics:
        from blackmirror.search.fitness import tied_with_best

        winner = self.best_fitness
        usable = [item for item in self.results if item.is_usable]
        failures = [item for item in self.results if not item.is_usable]
        improvement = (
            None
            if winner is None or self.root_fitness is None or winner.scalar_fitness is None
            else winner.scalar_fitness - self.root_fitness
        )
        return SearchEfficiencyMetrics(
            best_fitness=None if winner is None else winner.scalar_fitness,
            root_fitness=self.root_fitness,
            absolute_improvement=improvement,
            evaluations=len(self.results),
            evaluations_to_best=None if self.best is None else self.best.evaluation_number,
            wall_seconds=ledger.wall_seconds,
            wall_seconds_to_best=(
                None if self.best is None else self.best.wall_seconds_at_discovery
            ),
            tribe_runs=ledger.tribe_runs,
            tribe_runs_to_best=(
                None if self.best is None else self.best.tribe_runs_at_discovery
            ),
            cache_hit_rate=ledger.cache_hit_rate,
            failure_rate=(
                None if not self.results else len(failures) / len(self.results)
            ),
            improvement_is_resolvable=(
                None if improvement is None else improvement > minimum_improvement
            ),
            tied_candidate_ids=tuple(
                item.candidate_id
                for item in tied_with_best(usable, minimum_improvement=minimum_improvement)
            ),
        )


class StoppingPolicy:
    """Steps 43-46. Every reason a search may end, checked in one place."""

    def __init__(
        self,
        config: SearchConfig,
        budget: SearchBudget,
        *,
        started_at: dt.datetime | None = None,
    ) -> None:
        self.config = config
        self.budget = budget
        self.started_at = started_at or dt.datetime.now(dt.UTC)
        self._generations_without_gain = 0
        self._last_believed_best: float | None = None
        self.user_stopped = False

    def note_generation(self, best: CandidateFitness | None) -> None:
        """Advance patience, counting only gains large enough to believe.

        Counting any gain would keep a search alive on drift smaller than the
        pipeline's sensitivity, which is real hours spent chasing noise.
        """
        if best is None or best.scalar_fitness is None:
            self._generations_without_gain += 1
            return
        current = best.scalar_fitness
        if (
            self._last_believed_best is None
            or current - self._last_believed_best > self.config.minimum_improvement
        ):
            self._last_believed_best = current
            self._generations_without_gain = 0
        else:
            self._generations_without_gain += 1

    @property
    def generations_without_gain(self) -> int:
        return self._generations_without_gain

    def request_stop(self) -> None:
        self.user_stopped = True

    def check(
        self,
        ledger: BudgetLedger,
        best: CandidateFitness | None,
        *,
        strategy_stop: tuple[bool, str] = (False, ""),
    ) -> tuple[StoppingReason, str] | None:
        """The reason to stop, or None to continue.

        Order matters: an explicit user stop outranks everything, then hard
        budget limits, then a target being reached, then patience. A search that
        hit its budget and its patience in the same round should report the
        budget, because that is the binding constraint.
        """
        if self.user_stopped:
            return StoppingReason.USER_STOPPED, "a stop was requested"

        exhausted = ledger.exhausted(self.budget)
        for limit, reason in (
            (BudgetLimit.MAX_CANDIDATES, StoppingReason.MAX_CANDIDATES),
            (BudgetLimit.MAX_TRIBE_RUNS, StoppingReason.MAX_TRIBE_RUNS),
            (BudgetLimit.MAX_WALL_SECONDS, StoppingReason.MAX_WALL_TIME),
            (BudgetLimit.MAX_COST, StoppingReason.MAX_COST),
            (BudgetLimit.MAX_GENERATIONS, StoppingReason.MAX_GENERATIONS),
        ):
            if limit in exhausted:
                ceiling = self.budget.limit_for(limit)
                return reason, f"{limit.value} reached its limit of {ceiling}"

        if (
            self.config.target_fitness is not None
            and best is not None
            and best.scalar_fitness is not None
            and best.scalar_fitness >= self.config.target_fitness
        ):
            return (
                StoppingReason.TARGET_REACHED,
                f"best fitness {best.scalar_fitness:g} reached the target "
                f"{self.config.target_fitness:g}",
            )

        if self._generations_without_gain >= self.config.patience:
            return (
                StoppingReason.NO_IMPROVEMENT,
                f"no improvement above {self.config.minimum_improvement:g} for "
                f"{self._generations_without_gain} generations",
            )

        stop, why = strategy_stop
        if stop:
            return StoppingReason.SPACE_EXHAUSTED, why
        return None


def budget_warnings(
    ledger: BudgetLedger, budget: SearchBudget, *, threshold: float = 0.2
) -> tuple[str, ...]:
    """Limits with less than `threshold` of their allowance left."""
    out: list[str] = []
    for limit, left in ledger.remaining(budget).items():
        ceiling = budget.limit_for(limit)
        if left is None or ceiling is None or ceiling <= 0 or left <= 0:
            continue
        if left / ceiling <= threshold:
            out.append(f"{limit.value}: {left:g} of {ceiling:g} remaining")
    return tuple(out)


def summarize(points: Sequence[TrajectoryPoint]) -> tuple[str, ...]:
    """The best-so-far curve as readable rows."""
    return tuple(
        f"eval {point.evaluation_number:>3}  gen {point.generation}  "
        f"{point.candidate_id}  best={point.best_fitness:.4f}"
        + ("  <- new best" if point.is_new_best else "")
        for point in points
    )


def root_fitness_from(results: Iterable[CandidateFitness]) -> float | None:  # pragma: no cover
    """Convenience for callers holding a root evaluation among their results."""
    winner = best_of(results)
    return None if winner is None else winner.scalar_fitness


__all__ = [
    "BestCandidateRecord",
    "SearchEfficiencyMetrics",
    "SearchEvent",
    "SearchEventType",
    "SearchTracker",
    "StoppingPolicy",
    "TrajectoryPoint",
    "budget_warnings",
    "summarize",
]
