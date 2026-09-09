"""Steps 28-29: how a candidate is perturbed, and by how much.

WHY A REGISTRY RATHER THAN A FUNCTION
    Each parameter type moves differently. A continuous value slides, an integer
    steps, a categorical swaps, a boolean flips. Writing that as one function
    with a type switch inside means every new parameter type edits the same
    block; a registry keyed on type lets the space grow without the perturbation
    logic being rewritten each time.

WHY MAGNITUDE IS A FRACTION OF THE DECLARED RANGE
    A 3 dB move means something quite different in a 12 dB range than in a 60 dB
    one. Expressing step size as a fraction of each parameter's own domain makes
    "small" mean the same thing everywhere, so a search tuned on one space
    behaves comparably on another.

WHAT IS NOT HERE
    The brief names timing, text-candidate and segment-order mutations. None of
    those parameter types exist in this build, because nothing can materialize
    them. Implementing operators for knobs that cannot turn would produce a
    search that reports moves it never made.
"""

from __future__ import annotations

from enum import StrEnum
from random import Random
from typing import Protocol

from blackmirror.search.genome import CandidateGenome, GenomeOrigin
from blackmirror.search.space import ContentSearchSpace, ParameterType, SearchParameter


class MutationMagnitude(StrEnum):
    """How far a perturbation reaches, as a share of the parameter's range."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


#: Fraction of a parameter's declared domain used as the perturbation scale.
MAGNITUDE_SCALE: dict[MutationMagnitude, float] = {
    MutationMagnitude.SMALL: 0.05,
    MutationMagnitude.MEDIUM: 0.15,
    MutationMagnitude.LARGE: 0.35,
}


class MutationOperator(Protocol):
    """Moves one parameter's value. Never returns anything out of domain."""

    def mutate(
        self, parameter: SearchParameter, value: float, rng: Random, scale: float
    ) -> float: ...


class ContinuousPerturbation:
    """Gaussian step, clamped and quantized.

    Gaussian rather than uniform so most proposals land near the current value
    and the occasional one reaches further, which is the behaviour local search
    wants: refine mostly, but do not be unable to escape.
    """

    def mutate(
        self, parameter: SearchParameter, value: float, rng: Random, scale: float
    ) -> float:
        assert parameter.low is not None and parameter.high is not None
        sigma = (parameter.high - parameter.low) * scale
        return parameter.clamp(rng.gauss(value, sigma) if sigma > 0 else value)


class IntegerStep:
    """Whole-number step of at least one, so a mutation always moves."""

    def mutate(
        self, parameter: SearchParameter, value: float, rng: Random, scale: float
    ) -> float:
        assert parameter.low is not None and parameter.high is not None
        span = parameter.high - parameter.low
        reach = max(1, round(span * scale))
        step = rng.randint(1, reach) * rng.choice((-1, 1))
        return parameter.clamp(value + step)


class CategoricalSwap:
    """Pick a different choice, never the current one.

    Returning the same value would count as a mutation that produced a
    duplicate, which costs a proposal slot and yields nothing.
    """

    def mutate(
        self, parameter: SearchParameter, value: float, rng: Random, scale: float
    ) -> float:
        alternatives = [choice for choice in parameter.choices if choice != value]
        return rng.choice(alternatives) if alternatives else value


class BooleanToggle:
    """Flip between the declared off and on values."""

    def mutate(
        self, parameter: SearchParameter, value: float, rng: Random, scale: float
    ) -> float:
        off, on = parameter.choices
        return on if value == off else off


class MutationRegistry:
    """Which operator handles which parameter type."""

    def __init__(self, operators: dict[ParameterType, MutationOperator] | None = None) -> None:
        self._operators: dict[ParameterType, MutationOperator] = dict(
            operators
            or {
                ParameterType.CONTINUOUS: ContinuousPerturbation(),
                ParameterType.INTEGER: IntegerStep(),
                ParameterType.CATEGORICAL: CategoricalSwap(),
                ParameterType.BOOLEAN: BooleanToggle(),
            }
        )

    def register(self, parameter_type: ParameterType, operator: MutationOperator) -> None:
        self._operators[parameter_type] = operator

    def get(self, parameter_type: ParameterType) -> MutationOperator:
        operator = self._operators.get(parameter_type)
        if operator is None:
            raise ValueError(
                f"no mutation operator for parameter type {parameter_type.value!r}"
            )
        return operator

    def mutate_value(
        self,
        parameter: SearchParameter,
        value: float,
        rng: Random,
        magnitude: MutationMagnitude = MutationMagnitude.MEDIUM,
    ) -> float:
        return self.get(parameter.type).mutate(
            parameter, value, rng, MAGNITUDE_SCALE[magnitude]
        )

    def mutate(
        self,
        genome: CandidateGenome,
        space: ContentSearchSpace,
        rng: Random,
        *,
        genome_id: str,
        magnitude: MutationMagnitude = MutationMagnitude.MEDIUM,
        rate: float = 1.0,
        proposed_by: str = "",
    ) -> CandidateGenome:
        """A child differing from `genome` in one or more parameters.

        `rate` is the chance each parameter is touched. If chance leaves every
        parameter untouched, one is forced, because a mutation that changed
        nothing would be a duplicate of its parent and would waste an
        evaluation slot on a known answer.
        """
        genome.validate_against(space)
        values = dict(genome.values)
        touched: list[str] = []
        for parameter in space.parameters:
            if rng.random() <= rate:
                values[parameter.name] = self.mutate_value(
                    parameter, values[parameter.name], rng, magnitude
                )
                touched.append(parameter.name)
        if not touched:
            parameter = rng.choice(list(space.parameters))
            values[parameter.name] = self.mutate_value(
                parameter, values[parameter.name], rng, magnitude
            )
        return genome.with_values(
            values,
            genome_id=genome_id,
            origin=GenomeOrigin.MUTATION,
            proposed_by=proposed_by or genome.proposed_by,
        )


def neighbours(
    genome: CandidateGenome,
    space: ContentSearchSpace,
    *,
    magnitude: MutationMagnitude = MutationMagnitude.SMALL,
    steps: tuple[float, ...] = (-1.0, 1.0),
) -> tuple[dict[str, float], ...]:
    """Axis-aligned points around a genome, deterministically.

    One parameter moves at a time. That is what makes a local search's result
    interpretable: when a neighbour scores better, exactly one thing differs
    between it and its parent, so the move can be attributed. A neighbourhood
    that varied everything at once would score points nobody could explain.

    Points identical to the parent, or outside the domain after clamping, are
    dropped rather than returned as candidates.
    """
    genome.validate_against(space)
    scale = MAGNITUDE_SCALE[magnitude]
    out: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    current = tuple(sorted(genome.values.items()))
    seen.add(current)
    for parameter in space.parameters:
        base = genome.values[parameter.name]
        for step in steps:
            if parameter.is_ranged:
                assert parameter.low is not None and parameter.high is not None
                delta = (parameter.high - parameter.low) * scale * step
                moved = parameter.clamp(base + delta)
            else:
                choices = list(parameter.choices)
                index = choices.index(base) if base in choices else 0
                moved = choices[(index + int(step)) % len(choices)]
            if moved == base:
                continue
            candidate = {**genome.values, parameter.name: moved}
            key = tuple(sorted(candidate.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return tuple(out)


def crossover(
    left: CandidateGenome,
    right: CandidateGenome,
    space: ContentSearchSpace,
    rng: Random,
    *,
    genome_id: str,
    proposed_by: str = "",
) -> CandidateGenome:
    """Step 30. Take each parameter from one parent or the other.

    Uniform rather than single-point, because the parameters here have no
    meaningful ordering: there is no reason a cut between the second and third
    knob should preserve anything. Every value comes from a parent, so a child
    can only ever sit at a combination its parents already justified.
    """
    left.validate_against(space)
    right.validate_against(space)
    values = {
        parameter.name: (
            left.values[parameter.name]
            if rng.random() < 0.5
            else right.values[parameter.name]
        )
        for parameter in space.parameters
    }
    return CandidateGenome(
        genome_id=genome_id,
        values=values,
        origin=GenomeOrigin.CROSSOVER,
        generation=max(left.generation, right.generation) + 1,
        parent_genome_ids=(left.genome_id, right.genome_id),
        proposed_by=proposed_by or left.proposed_by,
    )


__all__ = [
    "MAGNITUDE_SCALE",
    "BooleanToggle",
    "CategoricalSwap",
    "ContinuousPerturbation",
    "IntegerStep",
    "MutationMagnitude",
    "MutationOperator",
    "MutationRegistry",
    "crossover",
    "neighbours",
]
