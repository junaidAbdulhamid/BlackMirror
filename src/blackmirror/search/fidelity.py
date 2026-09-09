"""Step 23: the multi-fidelity interface, with one level implemented.

WHY THE ENUM EXISTS AND THE APPROXIMATIONS DO NOT
    Successive halving and most sample-efficiency techniques assume a cheap,
    lower-fidelity evaluation whose ranking broadly agrees with the expensive
    one. This pipeline has no such thing. Reduced temporal sampling, a cortical
    subset, a shortened stimulus window and a learned surrogate are all
    plausible sources, and not one of them has been evaluated for whether its
    ranking matches a full pass.

    Adding one anyway would be the most damaging shortcut available here: every
    elimination it drove would be confident, cheap, and based on a number nobody
    validated. The step that asks for this architecture says so explicitly, and
    the honest implementation is a declared interface with a single level.

WHAT WOULD BE NEEDED TO ADD ONE
    A measured agreement study: run N candidates at both fidelities, report rank
    correlation and the rate at which the cheap level would have eliminated the
    eventual winner. Until that exists, `FULL` is the only level a search may
    use, and `is_implemented` says which are real.
"""

from __future__ import annotations

from enum import StrEnum


class FidelityLevel(StrEnum):
    """Declared levels. Only FULL is implemented; the rest are placeholders."""

    LOW = "low"
    MEDIUM = "medium"
    FULL = "full"

    @property
    def is_implemented(self) -> bool:
        """Whether a search may actually evaluate at this level."""
        return self is FidelityLevel.FULL

    @property
    def unavailable_reason(self) -> str:
        if self.is_implemented:
            return ""
        return (
            f"fidelity level {self.value!r} has no implementation: no reduced-cost "
            f"approximation of a TRIBE pass has been shown to rank candidates the "
            f"same way a full pass does, so eliminating on one would be a confident "
            f"decision based on an unvalidated number"
        )


def require_implemented(level: FidelityLevel) -> FidelityLevel:
    """Guard for anything about to evaluate at a given fidelity."""
    if not level.is_implemented:
        raise NotImplementedError(level.unavailable_reason)
    return level


__all__ = ["FidelityLevel", "require_implemented"]
