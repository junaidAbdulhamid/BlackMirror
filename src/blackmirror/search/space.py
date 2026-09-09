"""Steps 6-7: what a search is allowed to vary, and how each knob is typed.

WHY THE SPACE IS EXPLICIT
    The alternative is letting something decide for itself what to change about
    a stimulus. That produces candidates nobody authorized, differing from the
    parent in ways nobody recorded, which makes the resulting comparison
    uninterpretable: if two things changed and the score moved, the experiment
    says nothing about either. A declared space is what makes each candidate a
    controlled comparison rather than a new piece of content.

WHY EVERY PARAMETER NAMES AN EDIT OPERATION
    A parameter that cannot be turned into a file cannot be evaluated. Binding
    each parameter to a `materialization.EditOperation` at construction means an
    unrealizable space fails immediately, with a message naming what is missing,
    instead of failing later after a strategy has already spent budget planning
    around it.

WHAT THIS BUILD CANNOT SEARCH OVER
    `EVENT_TIMING`, `TEXT_CANDIDATE` and `STRUCTURED_INTERVENTION` are part of
    the type vocabulary because Phase 7 can express them, but no materializer
    implements them: they need cut-and-reassemble editing or generative models.
    They are rejected rather than silently ignored, because a space that
    accepted them would report a search over parameters that never moved.
"""

from __future__ import annotations

import math
from enum import StrEnum
from itertools import product
from random import Random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from blackmirror.materialization.schemas import (
    OPERATION_BOUNDS,
    OPERATION_MODALITY,
    EditOperation,
    EditStep,
)

SEARCH_SPACE_VERSION = "1.0"

#: A value this close to another is the same candidate for search purposes.
#: Below it, two genomes differ by less than the edit can express in the
#: encoded file, so evaluating both would spend an inference to learn nothing.
DEFAULT_RESOLUTION = 1e-3


class SearchSpaceError(ValueError):
    """The declared space cannot be searched as written.

    Raised directly by the decoding helpers. Note that when it is raised inside
    a pydantic validator the framework re-wraps it as a `ValidationError`, so
    callers guarding construction should catch `ValueError`, which covers both.
    """


class ParameterType(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    EVENT_TIMING = "event_timing"
    TEXT_CANDIDATE = "text_candidate"
    STRUCTURED_INTERVENTION = "structured_intervention"


#: Types this build can turn into media. The rest are declared so a space can
#: describe them and be refused precisely.
REALIZABLE_TYPES = frozenset(
    {
        ParameterType.CONTINUOUS,
        ParameterType.INTEGER,
        ParameterType.CATEGORICAL,
        ParameterType.BOOLEAN,
    }
)

_UNREALIZABLE_REASON: dict[ParameterType, str] = {
    ParameterType.EVENT_TIMING: (
        "moving an event requires cut-and-reassemble editing, which no "
        "materializer implements"
    ),
    ParameterType.TEXT_CANDIDATE: (
        "swapping text requires rendering or generation, which is out of scope"
    ),
    ParameterType.STRUCTURED_INTERVENTION: (
        "a structured intervention has no single parametric form to sample"
    ),
}


class SearchModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class SearchParameter(SearchModel):
    """One knob: a name, a type, the edit it drives, and its legal values."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    type: ParameterType
    operation: EditOperation
    low: float | None = None
    high: float | None = None
    #: Allowed values for CATEGORICAL, and the (false, true) pair for BOOLEAN.
    choices: tuple[float, ...] = ()
    #: Differences smaller than this are not distinguishable candidates.
    resolution: float = Field(default=DEFAULT_RESOLUTION, gt=0)

    @model_validator(mode="after")
    def _is_searchable(self) -> SearchParameter:
        if self.type not in REALIZABLE_TYPES:
            raise SearchSpaceError(
                f"parameter {self.name!r} has type {self.type.value}, which cannot be "
                f"materialized: {_UNREALIZABLE_REASON[self.type]}"
            )
        op_low, op_high, _ = OPERATION_BOUNDS[self.operation]

        if self.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
            if self.low is None or self.high is None:
                raise SearchSpaceError(f"parameter {self.name!r} needs low and high bounds")
            if self.choices:
                raise SearchSpaceError(f"parameter {self.name!r} is ranged, not categorical")
            if self.low >= self.high:
                raise SearchSpaceError(f"parameter {self.name!r} needs low < high")
            if self.type is ParameterType.INTEGER and (
                self.low != int(self.low) or self.high != int(self.high)
            ):
                raise SearchSpaceError(f"integer parameter {self.name!r} needs integer bounds")
            values = (self.low, self.high)
        else:
            if not self.choices:
                raise SearchSpaceError(f"parameter {self.name!r} needs choices")
            if self.low is not None or self.high is not None:
                raise SearchSpaceError(f"parameter {self.name!r} is categorical, not ranged")
            if len(set(self.choices)) != len(self.choices):
                raise SearchSpaceError(f"parameter {self.name!r} has duplicate choices")
            if self.type is ParameterType.BOOLEAN and len(self.choices) != 2:
                raise SearchSpaceError(
                    f"boolean parameter {self.name!r} needs exactly two choices, "
                    f"read as (off, on)"
                )
            values = tuple(self.choices)  # type: ignore[assignment]

        # A range wider than the operation supports would sample values ffmpeg
        # rejects, and only at materialization time.
        for value in values:
            if not op_low <= value <= op_high:
                raise SearchSpaceError(
                    f"parameter {self.name!r} allows {value:g}, outside the range "
                    f"[{op_low:g}, {op_high:g}] that {self.operation.value} accepts"
                )
        return self

    @property
    def modality(self) -> str:
        return OPERATION_MODALITY[self.operation]

    @property
    def is_ranged(self) -> bool:
        return self.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER)

    def sample(self, rng: Random) -> float:
        """One legal value, drawn from the caller's seeded generator."""
        if self.type is ParameterType.CONTINUOUS:
            assert self.low is not None and self.high is not None
            return self.quantize(rng.uniform(self.low, self.high))
        if self.type is ParameterType.INTEGER:
            assert self.low is not None and self.high is not None
            return float(rng.randint(int(self.low), int(self.high)))
        return rng.choice(list(self.choices))

    def quantize(self, value: float) -> float:
        """Snap to the parameter's resolution, so near-duplicates collapse."""
        if not self.is_ranged:
            return value
        if self.type is ParameterType.INTEGER:
            return float(round(value))
        steps = round(value / self.resolution)
        return round(steps * self.resolution, 12)

    def clamp(self, value: float) -> float:
        """Bring a proposed value back inside the declared domain.

        Strategies perturb values and will overshoot. Clamping is correct here
        because the bound is a constraint the user set, not a preference: a
        neighbour just outside it is not a candidate the search may consider.
        """
        if self.is_ranged:
            assert self.low is not None and self.high is not None
            return self.quantize(min(max(value, self.low), self.high))
        return min(self.choices, key=lambda choice: abs(choice - value))

    def contains(self, value: float) -> bool:
        if self.is_ranged:
            assert self.low is not None and self.high is not None
            return self.low - 1e-12 <= value <= self.high + 1e-12
        return any(abs(value - choice) <= 1e-12 for choice in self.choices)

    def grid_values(self, steps: int) -> tuple[float, ...]:
        """Evenly spaced values for exhaustive enumeration."""
        if not self.is_ranged:
            return tuple(self.choices)
        assert self.low is not None and self.high is not None
        if self.type is ParameterType.INTEGER:
            return tuple(float(v) for v in range(int(self.low), int(self.high) + 1))
        if steps < 2:
            return (self.quantize((self.low + self.high) / 2),)
        span = self.high - self.low
        return tuple(
            self.quantize(self.low + span * index / (steps - 1)) for index in range(steps)
        )

    def cardinality(self, steps: int) -> float:
        """How many distinct values this contributes to a grid."""
        if not self.is_ranged:
            return float(len(self.choices))
        if self.type is ParameterType.INTEGER:
            assert self.low is not None and self.high is not None
            return float(int(self.high) - int(self.low) + 1)
        return float(max(steps, 1))


class ContentSearchSpace(SearchModel):
    """The complete set of knobs, and the only things a search may change."""

    parameters: tuple[SearchParameter, ...] = Field(min_length=1)
    search_space_version: str = SEARCH_SPACE_VERSION

    @model_validator(mode="after")
    def _distinct(self) -> ContentSearchSpace:
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise SearchSpaceError("parameter names must be unique")
        operations = [parameter.operation for parameter in self.parameters]
        if len(set(operations)) != len(operations):
            # Two parameters driving one operation would fight at materialization
            # and the winner would depend on ordering.
            raise SearchSpaceError(
                "two parameters drive the same edit operation; each operation may "
                "be controlled by at most one parameter"
            )
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def modalities(self) -> frozenset[str]:
        return frozenset(parameter.modality for parameter in self.parameters)

    def get(self, name: str) -> SearchParameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"no parameter named {name!r} in this search space")

    def sample(self, rng: Random) -> dict[str, float]:
        """One point in the space, using the caller's seeded generator."""
        return {parameter.name: parameter.sample(rng) for parameter in self.parameters}

    def clamp(self, values: dict[str, float]) -> dict[str, float]:
        return {
            parameter.name: parameter.clamp(values[parameter.name])
            for parameter in self.parameters
            if parameter.name in values
        }

    def contains(self, values: dict[str, float]) -> bool:
        if set(values) != set(self.names):
            return False
        return all(
            self.get(name).contains(value) for name, value in values.items()
        )

    def cardinality(self, steps: int = 5) -> float:
        """Size of the grid this space would produce.

        Returned before enumerating anything so a caller can refuse a grid that
        would cost more inference hours than exist.
        """
        total = 1.0
        for parameter in self.parameters:
            total *= parameter.cardinality(steps)
            if math.isinf(total):  # pragma: no cover - defensive
                break
        return total

    def grid(self, steps: int = 5) -> tuple[dict[str, float], ...]:
        axes = [parameter.grid_values(steps) for parameter in self.parameters]
        return tuple(
            dict(zip(self.names, combination, strict=True))
            for combination in product(*axes)
        )

    def to_edit_steps(self, values: dict[str, float]) -> tuple[EditStep, ...]:
        """Turn a point in the space into the edits that build the file.

        Values outside the space are rejected rather than clamped: a caller that
        reached here with an illegal value has a bug, and quietly correcting it
        would make the recorded genome disagree with the produced media.
        """
        missing = set(self.names) - set(values)
        if missing:
            raise SearchSpaceError(f"missing values for: {', '.join(sorted(missing))}")
        unknown = set(values) - set(self.names)
        if unknown:
            raise SearchSpaceError(f"values for unknown parameters: {', '.join(sorted(unknown))}")
        steps: list[EditStep] = []
        for parameter in self.parameters:
            value = values[parameter.name]
            if not parameter.contains(value):
                raise SearchSpaceError(
                    f"{parameter.name}={value:g} is outside its declared domain"
                )
            steps.append(EditStep(operation=parameter.operation, value=value))
        return tuple(steps)

    def is_neutral(self, values: dict[str, float]) -> bool:
        """Whether this point would reproduce the source unchanged."""
        return all(
            values[parameter.name] == OPERATION_BOUNDS[parameter.operation][2]
            for parameter in self.parameters
            if parameter.name in values
        )

    def describe(self) -> tuple[str, ...]:
        lines = []
        for parameter in self.parameters:
            domain = (
                f"[{parameter.low:g}, {parameter.high:g}]"
                if parameter.is_ranged
                else "{" + ", ".join(f"{c:g}" for c in parameter.choices) + "}"
            )
            lines.append(
                f"{parameter.name} ({parameter.type.value}) -> "
                f"{parameter.operation.value} in {domain}"
            )
        return tuple(lines)


def continuous(
    name: str, operation: EditOperation, low: float, high: float, **kwargs: Any
) -> SearchParameter:
    return SearchParameter(
        name=name, type=ParameterType.CONTINUOUS, operation=operation,
        low=low, high=high, **kwargs,
    )


def categorical(
    name: str, operation: EditOperation, choices: tuple[float, ...], **kwargs: Any
) -> SearchParameter:
    return SearchParameter(
        name=name, type=ParameterType.CATEGORICAL, operation=operation,
        choices=choices, **kwargs,
    )


__all__ = [
    "DEFAULT_RESOLUTION",
    "REALIZABLE_TYPES",
    "SEARCH_SPACE_VERSION",
    "ContentSearchSpace",
    "ParameterType",
    "SearchParameter",
    "SearchSpaceError",
    "categorical",
    "continuous",
]
