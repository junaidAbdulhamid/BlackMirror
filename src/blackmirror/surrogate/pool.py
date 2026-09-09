"""Steps 26-27: generating many cheap candidates to screen.

WHY A POOL RATHER THAN AN OPTIMIZER OVER THE SURROGATE
    The textbook move is to maximise the acquisition function directly with a
    gradient method or a global optimizer. That is the right choice in high
    dimensions; here the space is two to five dimensions with bounded, quantized
    parameters, so sampling several thousand points covers it densely and costs
    milliseconds. Sampling also avoids an inner optimizer that could converge on
    a sharp artefact of a model fitted to a handful of observations.

WHY NOTHING IS MATERIALIZED
    A pool member is a genome and nothing else. No file is written and no
    inference runs until the acquisition has chosen. That is the entire economy
    of the phase: thousands of candidates screened for the cost of one that is
    not.

WHY THE POOL MIXES SOURCES
    Pure random sampling covers the space but rarely lands near the incumbent,
    where the improvement usually is. Pure local perturbation refines the
    incumbent but never leaves its basin. The mixture gives the acquisition
    function both kinds of candidate to choose between, and lets it, rather than
    the sampler, decide the balance.
"""

from __future__ import annotations

from random import Random

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.search.genome import CandidateGenome, GenomeOrigin
from blackmirror.search.mutation import MutationMagnitude, MutationRegistry, neighbours
from blackmirror.search.space import ContentSearchSpace


class PoolConfig(BaseModel):
    """How many candidates to generate, and from where."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    size: int = Field(default=2000, ge=1, le=200_000)
    #: Share drawn uniformly from the whole space.
    random_share: float = Field(default=0.6, ge=0.0, le=1.0)
    #: Share perturbed around the best observations.
    local_share: float = Field(default=0.4, ge=0.0, le=1.0)
    magnitude: MutationMagnitude = MutationMagnitude.MEDIUM
    #: How many of the best observations to perturb around.
    incumbent_count: int = Field(default=3, ge=1, le=32)
    max_attempts_multiplier: int = Field(default=4, ge=1, le=64)


class CandidatePoolGenerator:
    """Step 27. Produces distinct, valid, unevaluated genomes."""

    def __init__(
        self,
        space: ContentSearchSpace,
        *,
        config: PoolConfig | None = None,
        registry: MutationRegistry | None = None,
    ) -> None:
        self.space = space
        self.config = config or PoolConfig()
        self.registry = registry or MutationRegistry()

    def generate(
        self,
        rng: Random,
        *,
        incumbents: list[CandidateGenome] | None = None,
        exclude: set[str] | None = None,
        generation: int = 0,
    ) -> list[CandidateGenome]:
        """A pool of distinct genomes, excluding fingerprints already seen.

        `exclude` carries the fingerprints of everything already evaluated, so
        the pool never offers a candidate whose answer is known. The generator
        gives up after a bounded number of attempts rather than looping, because
        a small space genuinely runs out and hanging would be worse than a
        short pool.
        """
        seen = set(exclude or ())
        out: list[CandidateGenome] = []
        counter = 0
        attempts = 0
        limit = self.config.size * self.config.max_attempts_multiplier

        local_target = (
            int(self.config.size * self.config.local_share) if incumbents else 0
        )
        # Each local point carries the incumbent it came from, because a genome
        # whose origin is NEIGHBOUR must name its parent: lineage is what lets a
        # later reader say where a candidate came from.
        neighbourhood = self._neighbourhood(incumbents or [])

        while len(out) < self.config.size and attempts < limit:
            attempts += 1
            use_local = bool(neighbourhood) and len(out) < local_target
            parent_id: str | None = None
            if use_local:
                values, parent_id = rng.choice(neighbourhood)
                values = self._perturb(values, rng)
            else:
                values = self.space.sample(rng)
            values = self.space.clamp(values)
            if self.space.is_neutral(values):
                continue
            counter += 1
            genome = CandidateGenome(
                # Namespaced by round: an incumbent from an earlier pool carries
                # a pool id too, and a bare counter would eventually reissue it,
                # making a genome its own parent.
                genome_id=f"p{generation}_{counter:06d}",
                values=values,
                origin=(
                    GenomeOrigin.NEIGHBOUR
                    if use_local and parent_id
                    else GenomeOrigin.RANDOM_SAMPLE
                ),
                generation=generation,
                parent_genome_ids=(parent_id,) if use_local and parent_id else (),
                proposed_by="candidate_pool",
            )
            fingerprint = genome.fingerprint(self.space)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            out.append(genome)
        return out

    def _neighbourhood(
        self, incumbents: list[CandidateGenome]
    ) -> list[tuple[dict[str, float], str]]:
        """Neighbours of the best observations, each tagged with its parent."""
        points: list[tuple[dict[str, float], str]] = []
        for genome in incumbents[: self.config.incumbent_count]:
            points.append((dict(genome.values), genome.genome_id))
            points.extend(
                (values, genome.genome_id)
                for values in neighbours(
                    genome, self.space, magnitude=self.config.magnitude
                )
            )
        return points

    def _perturb(self, values: dict[str, float], rng: Random) -> dict[str, float]:
        """Jitter one parameter, so local candidates are not just the neighbours."""
        parameter = rng.choice(list(self.space.parameters))
        moved = dict(values)
        moved[parameter.name] = self.registry.mutate_value(
            parameter, values[parameter.name], rng, self.config.magnitude
        )
        return moved


__all__ = ["CandidatePoolGenerator", "PoolConfig"]
