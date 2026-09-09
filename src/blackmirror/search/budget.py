"""Steps 3-5: what a search may spend, what it has spent, and hard admission.

WHY THIS IS NOT AN AFTERTHOUGHT
    One evaluation of a ten-second clip costs roughly 52 minutes of inference on
    this machine, measured. A search that overshoots its budget by five
    candidates has overshot by four hours. Budget is therefore checked *before*
    work is scheduled, never after, and the check returns how many items may
    proceed rather than a yes/no, so a batch of five against two remaining runs
    admits exactly two instead of failing or overspending.

WHY WALL-CLOCK AND NOT GPU SECONDS
    TRIBE v2 runs on CPU here, and `device="auto"` has no MPS path. A field
    named `max_gpu_seconds` would be counting something that never happens.
    Wall-clock is what actually elapses and what a person waits for, so that is
    what is measured. Monetary cost is optional and derived from an operator's
    own rate, labelled an estimate because it is one.

WHY THE LEDGER IS IMMUTABLE
    Search state is checkpointed after every evaluation so a killed process can
    resume. A ledger that mutated in place would let a crash between the spend
    and the checkpoint lose the record of work already paid for, and the search
    would spend it again.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BudgetLimit(StrEnum):
    """The dimensions a search can exhaust."""

    MAX_CANDIDATES = "max_candidates"
    MAX_TRIBE_RUNS = "max_tribe_runs"
    MAX_WALL_SECONDS = "max_wall_seconds"
    MAX_COST = "max_cost"
    MAX_GENERATIONS = "max_generations"


class BudgetModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class SearchBudget(BudgetModel):
    """Hard ceilings. Every field is optional, but not all of them.

    At least one finite limit is required (Step 74). A search with no ceiling
    on a 52-minute evaluation is not an experiment, it is an open-ended
    commitment of the machine, and the default must never be that.
    """

    max_candidates: int | None = Field(default=25, ge=1, le=10_000)
    max_tribe_runs: int | None = Field(default=None, ge=1, le=10_000)
    max_wall_seconds: float | None = Field(default=None, gt=0)
    max_cost: float | None = Field(default=None, gt=0)
    max_generations: int | None = Field(default=10, ge=1, le=1_000)
    #: Optional operator rate used only to turn elapsed time into an estimate.
    cost_per_wall_hour: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _at_least_one_finite_limit(self) -> SearchBudget:
        if not any(
            value is not None
            for value in (
                self.max_candidates,
                self.max_tribe_runs,
                self.max_wall_seconds,
                self.max_cost,
                self.max_generations,
            )
        ):
            raise ValueError(
                "a search needs at least one finite limit; an unbounded search "
                "over evaluations that take roughly an hour each is not a "
                "safe default"
            )
        if self.max_cost is not None and self.cost_per_wall_hour <= 0:
            raise ValueError(
                "max_cost is meaningless without cost_per_wall_hour, because "
                "nothing would ever accrue against it"
            )
        return self

    def limit_for(self, limit: BudgetLimit) -> float | None:
        return {
            BudgetLimit.MAX_CANDIDATES: self.max_candidates,
            BudgetLimit.MAX_TRIBE_RUNS: self.max_tribe_runs,
            BudgetLimit.MAX_WALL_SECONDS: self.max_wall_seconds,
            BudgetLimit.MAX_COST: self.max_cost,
            BudgetLimit.MAX_GENERATIONS: self.max_generations,
        }[limit]


class BudgetLedger(BudgetModel):
    """What has actually been spent. Updated by returning a new ledger."""

    candidates_generated: int = Field(default=0, ge=0)
    candidates_evaluated: int = Field(default=0, ge=0)
    tribe_runs: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    evaluation_failures: int = Field(default=0, ge=0)
    generations_completed: int = Field(default=0, ge=0)
    wall_seconds: float = Field(default=0.0, ge=0)
    started_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def estimated_cost(self, budget: SearchBudget) -> float:
        return self.wall_seconds / 3600.0 * budget.cost_per_wall_hour

    def spent_on(self, limit: BudgetLimit, budget: SearchBudget) -> float:
        return {
            BudgetLimit.MAX_CANDIDATES: float(self.candidates_generated),
            BudgetLimit.MAX_TRIBE_RUNS: float(self.tribe_runs),
            BudgetLimit.MAX_WALL_SECONDS: self.wall_seconds,
            BudgetLimit.MAX_COST: self.estimated_cost(budget),
            BudgetLimit.MAX_GENERATIONS: float(self.generations_completed),
        }[limit]

    def remaining(self, budget: SearchBudget) -> dict[BudgetLimit, float | None]:
        """How much is left on each dimension. `None` means unlimited."""
        out: dict[BudgetLimit, float | None] = {}
        for limit in BudgetLimit:
            ceiling = budget.limit_for(limit)
            out[limit] = (
                None if ceiling is None else max(0.0, ceiling - self.spent_on(limit, budget))
            )
        return out

    def exhausted(self, budget: SearchBudget) -> tuple[BudgetLimit, ...]:
        return tuple(
            limit
            for limit, left in self.remaining(budget).items()
            if left is not None and left <= 0
        )

    @property
    def cache_hit_rate(self) -> float | None:
        """`None` rather than zero when nothing has been attempted yet."""
        attempts = self.candidates_evaluated + self.cache_hits
        return None if attempts == 0 else self.cache_hits / attempts

    # --- spending ---------------------------------------------------------

    def with_proposed(self, count: int) -> BudgetLedger:
        return self._update(candidates_generated=self.candidates_generated + count)

    def with_evaluation(
        self, *, wall_seconds: float, cached: bool = False, failed: bool = False
    ) -> BudgetLedger:
        """Record one finished evaluation.

        A cache hit still counts as an evaluated candidate, because the search
        learned its fitness, but it does not count as a TRIBE run, because no
        inference happened. Conflating the two would make the compute figures in
        the final report wrong in the flattering direction.
        """
        return self._update(
            candidates_evaluated=self.candidates_evaluated + 1,
            tribe_runs=self.tribe_runs + (0 if cached else 1),
            cache_hits=self.cache_hits + (1 if cached else 0),
            evaluation_failures=self.evaluation_failures + (1 if failed else 0),
            wall_seconds=self.wall_seconds + max(0.0, wall_seconds),
        )

    def with_generation_completed(self) -> BudgetLedger:
        return self._update(generations_completed=self.generations_completed + 1)

    def _update(self, **fields: object) -> BudgetLedger:
        return self.model_copy(update={**fields, "updated_at": dt.datetime.now(dt.UTC)})


class AdmissionDecision(BudgetModel):
    """How many of a requested batch may proceed, and why not the rest."""

    requested: int = Field(ge=0)
    admitted: int = Field(ge=0)
    binding_limit: BudgetLimit | None = None
    reason: str = ""

    @property
    def rejected(self) -> int:
        return self.requested - self.admitted

    @property
    def is_exhausted(self) -> bool:
        return self.admitted == 0 and self.requested > 0


def admit(
    requested: int, ledger: BudgetLedger, budget: SearchBudget, *, cost_per_candidate: int = 1
) -> AdmissionDecision:
    """Decide how many candidates may be scheduled. Step 5's hard enforcement.

    Returns a partial admission rather than refusing the batch, because the
    useful answer to "five requested, two affordable" is two. Only the limits
    that a *new candidate* consumes are counted here; wall-clock and cost are
    checked as stopping conditions instead, since their spend is not known until
    the work has run.
    """
    if requested <= 0:
        return AdmissionDecision(
            requested=max(0, requested), admitted=0, reason="nothing requested"
        )

    remaining = ledger.remaining(budget)
    allowance = requested
    binding: BudgetLimit | None = None
    for limit in (BudgetLimit.MAX_CANDIDATES, BudgetLimit.MAX_TRIBE_RUNS):
        left = remaining[limit]
        if left is None:
            continue
        affordable = int(left // cost_per_candidate)
        if affordable < allowance:
            allowance = max(0, affordable)
            binding = limit

    already_spent = list(ledger.exhausted(budget))
    if allowance == 0:
        names = ", ".join(limit.value for limit in already_spent) or (
            binding.value if binding else "budget"
        )
        return AdmissionDecision(
            requested=requested, admitted=0, binding_limit=binding,
            reason=f"exhausted: {names}",
        )
    if allowance < requested:
        return AdmissionDecision(
            requested=requested, admitted=allowance, binding_limit=binding,
            reason=(
                f"{binding.value if binding else 'budget'} allows only {allowance} "
                f"of {requested}"
            ),
        )
    return AdmissionDecision(requested=requested, admitted=requested, reason="within budget")


__all__ = [
    "AdmissionDecision",
    "BudgetLedger",
    "BudgetLimit",
    "SearchBudget",
    "admit",
]
