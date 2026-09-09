"""Steps 61-62: what a candidate must satisfy before it may be called best.

WHY FEASIBILITY IS SEPARATE FROM FITNESS
    A candidate can score highest on the objective and still be unusable: it ran
    long, it edited a locked segment, it wrecked a secondary measure the user
    said to protect. Folding that into the fitness number would hide it, and a
    search that optimises a number which silently mixes "good" with "allowed"
    will eventually return something inadmissible with a flattering score.

    So fitness answers "how did it do on the objective" and feasibility answers
    "is it allowed to win". Both are recorded; only the second can veto.

WHY A GUARDRAIL IS A SEPARATE OBJECTIVE, NOT A PENALTY
    A penalty term trades off: enough gain on the primary objective buys a
    violation. A guardrail does not trade off. If the user says stability must
    not fall below a bound, a candidate that breaches it is out regardless of
    how well it scores elsewhere, because that is what "must not" means.

WHAT IS CHECKED WHERE
    Constraints a genome can violate on its own, such as editing a frozen
    modality, are caught by the decoder before anything is built. Guardrails
    here are checked after evaluation, because they depend on measurements that
    do not exist until the candidate has been run.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GuardrailKind(StrEnum):
    OBJECTIVE_FLOOR = "objective_floor"
    OBJECTIVE_CEILING = "objective_ceiling"
    MAX_DURATION_CHANGE = "max_duration_change"
    MAX_RELATIVE_DROP = "max_relative_drop"


class GuardrailModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Guardrail(GuardrailModel):
    """One condition a candidate must satisfy to be eligible to win."""

    guardrail_id: str = Field(min_length=1, max_length=64)
    kind: GuardrailKind
    #: Which objective this watches. Unused by duration guardrails.
    objective_id: str | None = None
    bound: float
    description: str = ""

    @model_validator(mode="after")
    def _coherent(self) -> Guardrail:
        needs_objective = self.kind in (
            GuardrailKind.OBJECTIVE_FLOOR,
            GuardrailKind.OBJECTIVE_CEILING,
            GuardrailKind.MAX_RELATIVE_DROP,
        )
        if needs_objective and not self.objective_id:
            raise ValueError(f"{self.kind.value} needs the objective it watches")
        if self.kind is GuardrailKind.MAX_DURATION_CHANGE and self.bound < 0:
            raise ValueError("a duration allowance cannot be negative")
        if self.kind is GuardrailKind.MAX_RELATIVE_DROP and not 0 <= self.bound <= 1:
            raise ValueError("a relative drop is a fraction between 0 and 1")
        return self

    def describe(self) -> str:
        if self.description:
            return self.description
        if self.kind is GuardrailKind.OBJECTIVE_FLOOR:
            return f"{self.objective_id} must stay at or above {self.bound:g}"
        if self.kind is GuardrailKind.OBJECTIVE_CEILING:
            return f"{self.objective_id} must stay at or below {self.bound:g}"
        if self.kind is GuardrailKind.MAX_RELATIVE_DROP:
            return (
                f"{self.objective_id} must not fall more than "
                f"{self.bound:.0%} below the root"
            )
        return f"duration must not change by more than {self.bound:g}s"


class GuardrailViolation(GuardrailModel):
    guardrail_id: str
    kind: GuardrailKind
    objective_id: str | None = None
    observed: float | None = None
    bound: float
    message: str


class FeasibilityResult(GuardrailModel):
    """Whether a candidate may be called best, and why not if it may not."""

    feasible: bool
    violations: tuple[GuardrailViolation, ...] = ()
    #: Guardrails that could not be evaluated because a value was missing.
    unchecked: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.feasible and not self.unchecked:
            return "all guardrails satisfied"
        if self.feasible:
            return f"satisfied, {len(self.unchecked)} unchecked"
        return "; ".join(item.message for item in self.violations)


class GuardrailPolicy:
    """Evaluates a candidate's measurements against the declared guardrails."""

    def __init__(
        self,
        guardrails: tuple[Guardrail, ...] = (),
        *,
        root_objectives: dict[str, float] | None = None,
        treat_unchecked_as_violation: bool = False,
    ) -> None:
        ids = [item.guardrail_id for item in guardrails]
        if len(set(ids)) != len(ids):
            raise ValueError("guardrail ids must be unique")
        self.guardrails = guardrails
        self.root_objectives = root_objectives or {}
        # Off by default: a guardrail whose objective was not measured is
        # reported as unchecked rather than silently passed or silently failed.
        # Turning it on makes a missing measurement disqualifying, which is the
        # right setting when the guardrail protects something that matters more
        # than finding a winner at all.
        self.treat_unchecked_as_violation = treat_unchecked_as_violation

    def check(
        self, raw_objectives: dict[str, float | None], *, duration_change: float = 0.0
    ) -> FeasibilityResult:
        violations: list[GuardrailViolation] = []
        unchecked: list[str] = []

        for guardrail in self.guardrails:
            if guardrail.kind is GuardrailKind.MAX_DURATION_CHANGE:
                if abs(duration_change) > guardrail.bound + 1e-9:
                    violations.append(
                        GuardrailViolation(
                            guardrail_id=guardrail.guardrail_id,
                            kind=guardrail.kind,
                            observed=duration_change,
                            bound=guardrail.bound,
                            message=(
                                f"duration changed by {duration_change:+.3f}s, beyond "
                                f"the {guardrail.bound:g}s allowed"
                            ),
                        )
                    )
                continue

            assert guardrail.objective_id is not None
            observed = raw_objectives.get(guardrail.objective_id)
            if observed is None:
                unchecked.append(guardrail.guardrail_id)
                continue

            if guardrail.kind is GuardrailKind.OBJECTIVE_FLOOR and observed < guardrail.bound:
                violations.append(
                    self._violation(
                        guardrail, observed,
                        f"{guardrail.objective_id} was {observed:.4f}, below the "
                        f"floor of {guardrail.bound:g}",
                    )
                )
            elif (
                guardrail.kind is GuardrailKind.OBJECTIVE_CEILING
                and observed > guardrail.bound
            ):
                violations.append(
                    self._violation(
                        guardrail, observed,
                        f"{guardrail.objective_id} was {observed:.4f}, above the "
                        f"ceiling of {guardrail.bound:g}",
                    )
                )
            elif guardrail.kind is GuardrailKind.MAX_RELATIVE_DROP:
                root = self.root_objectives.get(guardrail.objective_id)
                if root is None or root == 0:
                    unchecked.append(guardrail.guardrail_id)
                    continue
                drop = (root - observed) / abs(root)
                if drop > guardrail.bound + 1e-12:
                    violations.append(
                        self._violation(
                            guardrail, observed,
                            f"{guardrail.objective_id} fell {drop:.1%} below the root, "
                            f"beyond the {guardrail.bound:.0%} allowed",
                        )
                    )

        feasible = not violations and not (
            unchecked and self.treat_unchecked_as_violation
        )
        return FeasibilityResult(
            feasible=feasible,
            violations=tuple(violations),
            unchecked=tuple(unchecked),
        )

    @staticmethod
    def _violation(
        guardrail: Guardrail, observed: float, message: str
    ) -> GuardrailViolation:
        return GuardrailViolation(
            guardrail_id=guardrail.guardrail_id,
            kind=guardrail.kind,
            objective_id=guardrail.objective_id,
            observed=observed,
            bound=guardrail.bound,
            message=message,
        )

    def describe(self) -> tuple[str, ...]:
        return tuple(item.describe() for item in self.guardrails)


__all__ = [
    "FeasibilityResult",
    "Guardrail",
    "GuardrailKind",
    "GuardrailPolicy",
    "GuardrailViolation",
]
