"""Step 9: the structured representation of one candidate.

WHAT A GENOME IS
    A point in the declared search space, plus where it came from. Nothing more.
    It is not a description of media and not an edit plan; it is the small piece
    of state a strategy reasons about, mutates and remembers, and everything
    downstream is derived from it.

    The biological word is borrowed and the analogy stops immediately. There is
    no expression, no dominance, no fitness baked in. It is a named point.

WHY IDENTITY IS CANONICAL RATHER THAN INCIDENTAL
    Two genomes that would build the same file must be recognised as the same
    candidate, or the search pays a full inference to learn something it already
    knows. So identity comes from the *values*, quantized by the space's
    declared resolution, with parameter names sorted. It deliberately excludes
    the origin, the generation and the parent: how a point was reached does not
    change what it is.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from blackmirror.search.space import ContentSearchSpace


class GenomeOrigin(StrEnum):
    """How a candidate came to be proposed. Recorded for explainability.

    Step 70 asks why a candidate was selected. Half of that answer is what
    produced it, and reconstructing it later from the tree shape alone would be
    guesswork.
    """

    ROOT = "root"
    RANDOM_SAMPLE = "random_sample"
    GRID_POINT = "grid_point"
    NEIGHBOUR = "neighbour"
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    MANUAL = "manual"


class CandidateGenome(BaseModel):
    """One point in the search space, with its provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    genome_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    values: dict[str, float] = Field(min_length=1)
    origin: GenomeOrigin = GenomeOrigin.RANDOM_SAMPLE
    generation: int = Field(default=0, ge=0)
    #: Parents, plural, because crossover has two. Empty for a root or a
    #: fresh random sample, which descend from nothing.
    parent_genome_ids: tuple[str, ...] = ()
    #: Which strategy proposed it, for the selection explanation.
    proposed_by: str = ""
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @model_validator(mode="after")
    def _sane_lineage(self) -> CandidateGenome:
        if len(set(self.parent_genome_ids)) != len(self.parent_genome_ids):
            raise ValueError("a genome cannot list the same parent twice")
        if self.genome_id in self.parent_genome_ids:
            raise ValueError("a genome cannot be its own parent")
        if self.origin is GenomeOrigin.CROSSOVER and len(self.parent_genome_ids) < 2:
            raise ValueError("a crossover genome needs at least two parents")
        if (
            self.origin in (GenomeOrigin.MUTATION, GenomeOrigin.NEIGHBOUR)
            and not self.parent_genome_ids
        ):
            raise ValueError(f"a {self.origin.value} genome needs the parent it came from")
        if self.origin is GenomeOrigin.ROOT and self.parent_genome_ids:
            raise ValueError("the root genome descends from nothing")
        return self

    def canonical_values(self, space: ContentSearchSpace) -> dict[str, float]:
        """Values snapped to each parameter's resolution, in name order.

        Quantizing here is what makes 21.600 and 21.6001 one candidate. Without
        it a continuous strategy would propose an unbounded stream of
        near-identical points and spend the whole budget on them.
        """
        return {
            name: space.get(name).quantize(self.values[name])
            for name in sorted(self.values)
        }

    def fingerprint(self, space: ContentSearchSpace) -> str:
        """Identity for duplicate detection. Same point, same fingerprint."""
        payload = {
            "space_version": space.search_space_version,
            "values": self.canonical_values(space),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def validate_against(self, space: ContentSearchSpace) -> None:
        """Fail loudly if this point is not in the space it claims to be in.

        Genomes travel between a strategy, a cache and a decoder, and a value
        that drifted out of range would otherwise surface as an ffmpeg error
        after the candidate had already been admitted against the budget.
        """
        missing = set(space.names) - set(self.values)
        if missing:
            raise ValueError(
                f"genome {self.genome_id!r} is missing values for: "
                f"{', '.join(sorted(missing))}"
            )
        unknown = set(self.values) - set(space.names)
        if unknown:
            raise ValueError(
                f"genome {self.genome_id!r} has values for parameters not in the "
                f"space: {', '.join(sorted(unknown))}"
            )
        for name, value in self.values.items():
            if not space.get(name).contains(value):
                raise ValueError(
                    f"genome {self.genome_id!r} sets {name}={value:g}, outside its "
                    f"declared domain"
                )

    def is_neutral(self, space: ContentSearchSpace) -> bool:
        """Whether this point would rebuild the root unchanged."""
        return space.is_neutral(self.values)

    def describe(self, space: ContentSearchSpace) -> str:
        canonical = self.canonical_values(space)
        return ", ".join(f"{name}={value:g}" for name, value in canonical.items())

    def with_values(
        self,
        values: dict[str, float],
        *,
        genome_id: str,
        origin: GenomeOrigin,
        proposed_by: str = "",
    ) -> CandidateGenome:
        """A child of this genome at a new point, keeping the lineage."""
        return CandidateGenome(
            genome_id=genome_id,
            values=values,
            origin=origin,
            generation=self.generation + 1,
            parent_genome_ids=(self.genome_id,),
            proposed_by=proposed_by or self.proposed_by,
        )


def genome_distance(
    left: CandidateGenome, right: CandidateGenome, space: ContentSearchSpace
) -> float:
    """Normalized distance between two points, for diversity and neighbourhood.

    Each ranged parameter contributes its absolute difference divided by the
    width of its domain, so a 3 dB move in a 12 dB range counts the same as a
    0.05 move in a 0.2 range. Categorical parameters contribute 1 when they
    differ and 0 when they match. The mean is returned, so the result is on
    [0, 1] whatever the space contains and spaces of different sizes stay
    comparable.
    """
    left.validate_against(space)
    right.validate_against(space)
    if not space.parameters:  # pragma: no cover - schema forbids it
        return 0.0
    total = 0.0
    for parameter in space.parameters:
        a = left.values[parameter.name]
        b = right.values[parameter.name]
        if parameter.is_ranged:
            assert parameter.low is not None and parameter.high is not None
            width = parameter.high - parameter.low
            total += abs(a - b) / width if width > 0 else 0.0
        else:
            total += 0.0 if a == b else 1.0
    return total / len(space.parameters)


def root_genome(space: ContentSearchSpace, *, genome_id: str = "root") -> CandidateGenome:
    """The unedited starting point: every parameter at its neutral value.

    Only meaningful as a reference. It cannot be materialized, because a plan
    that changes nothing reproduces the source, and the root's score is already
    known from the run the search started from.
    """
    from blackmirror.materialization.schemas import OPERATION_BOUNDS

    return CandidateGenome(
        genome_id=genome_id,
        values={
            parameter.name: OPERATION_BOUNDS[parameter.operation][2]
            for parameter in space.parameters
        },
        origin=GenomeOrigin.ROOT,
        generation=0,
        proposed_by="root",
    )


__all__ = [
    "CandidateGenome",
    "GenomeOrigin",
    "genome_distance",
    "root_genome",
]
