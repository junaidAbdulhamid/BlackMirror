"""Steps 36-38: never pay twice for the same candidate.

WHY THE KEY IS WIDE
    A cached fitness is only reusable if everything that could have changed the
    number is the same. That means the genome, but also the media it edits, the
    objective it was scored against, and the versions of the model, the scoring
    engine and the materializer. Leaving any of those out produces a cache that
    returns a number computed under different rules, which is worse than no
    cache: the search would optimise against a mixture of two pipelines and
    nothing would say so.

WHY IT PERSISTS
    Duplicate detection inside a strategy stops a search re-proposing a point
    it already tried. It does nothing for a second search over the same root,
    which is the common case when tuning a space or comparing strategies. At
    roughly 52 minutes per evaluation, a cache that survives the process is
    worth more than most of the algorithms.

WHAT IS NOT CACHED
    Failures. A candidate that failed because ffmpeg was missing or a disk
    filled would be replayed forever as a permanent failure. Failures are
    recorded in the search's own history instead, where they are visible.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.search.fitness import CandidateFitness
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.space import ContentSearchSpace
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_VERSION = "1.0"


class CacheIdentity(BaseModel):
    """Everything that must match for a stored fitness to be reusable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_media_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    space_version: str
    materialization_version: str
    scoring_version: str
    model_fingerprint: str
    cache_version: str = CACHE_VERSION

    def prefix(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    def key_for(self, genome: CandidateGenome, space: ContentSearchSpace) -> str:
        return hashlib.sha256(
            f"{self.prefix()}:{genome.fingerprint(space)}".encode()
        ).hexdigest()


class SearchEvaluationCache:
    """Fitness keyed by candidate identity, stored as one file per entry."""

    def __init__(self, root: Path, identity: CacheIdentity) -> None:
        self.identity = identity
        self.root = Path(root) / "search" / "cache" / identity.prefix()[:16]
        self.hits = 0
        self.misses = 0

    def get(
        self, genome: CandidateGenome, space: ContentSearchSpace
    ) -> CandidateFitness | None:
        path = self.root / f"{self.identity.key_for(genome, space)}.json"
        if not path.is_file():
            self.misses += 1
            return None
        try:
            stored = CandidateFitness.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt or schema-incompatible entry is a miss, not a crash.
            logger.warning("discarding unreadable cache entry %s", path.name)
            self.misses += 1
            return None
        self.hits += 1
        return stored

    def put(
        self, genome: CandidateGenome, space: ContentSearchSpace, fitness: CandidateFitness
    ) -> None:
        """Store a usable result. Failures are deliberately not cached."""
        if not fitness.is_usable:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{self.identity.key_for(genome, space)}.json"
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - renamed below
            mode="w", encoding="utf-8", prefix=".entry.", suffix=".tmp",
            dir=self.root, delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(fitness.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @property
    def hit_rate(self) -> float | None:
        total = self.hits + self.misses
        return None if total == 0 else self.hits / total

    def size(self) -> int:
        return len(list(self.root.glob("*.json"))) if self.root.is_dir() else 0


class CachingEvaluator:
    """Wraps an evaluator so an identical candidate is never re-run.

    A cache hit is reported with its compute cost zeroed and `served_from_cache`
    set, so the ledger records that no inference happened. Carrying the original
    cost forward would make the search look more expensive than it was, and
    reporting nothing at all would lose the fact that a hit occurred.
    """

    def __init__(
        self,
        inner: object,
        cache: SearchEvaluationCache,
        space: ContentSearchSpace,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.space = space

    def evaluate(self, genome: CandidateGenome) -> CandidateFitness:
        cached = self.cache.get(genome, self.space)
        if cached is not None:
            from blackmirror.search.fitness import ComputeCost

            logger.info("cache hit for candidate %s", genome.genome_id)
            return cached.model_copy(
                update={
                    "candidate_id": genome.genome_id,
                    "genome": genome,
                    "compute": ComputeCost(
                        wall_seconds=0.0, tribe_runs=0, served_from_cache=True
                    ),
                }
            )
        fitness = self.inner.evaluate(genome)  # type: ignore[attr-defined]
        self.cache.put(genome, self.space, fitness)
        return fitness


__all__ = [
    "CACHE_VERSION",
    "CacheIdentity",
    "CachingEvaluator",
    "SearchEvaluationCache",
]
