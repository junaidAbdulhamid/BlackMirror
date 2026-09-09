"""Steps 51-53, 72: queueing candidates, ordering them, and human override.

WHY PRIORITY IS A LIST OF REASONS, NOT A SCORE
    Step 53 asks for priority and warns against an opaque one. A single blended
    number would decide the order of hour-long jobs for reasons nobody could
    reconstruct. So each candidate carries the components that ranked it and the
    sentence that explains it, and the score is only the tie-break between
    equals.

WHY CACHED CANDIDATES GO FIRST
    A cache hit costs nothing and may raise the best-so-far immediately, which
    tightens the improvement threshold everything else is measured against.
    Running it before an hour-long job is free information first.

WHY CONCURRENCY IS OFFERED AND DEFAULTED OFF
    Step 51 asks for concurrent evaluation where the infrastructure allows. On
    this machine it does not: inference is memory-bound, and two passes contend
    for the same weights and finish later in total than one after the other.
    The knob exists and is honoured; the default is one, and the reason is
    recorded rather than left as folklore.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.search.fitness import CandidateFitness
from blackmirror.search.genome import CandidateGenome
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

Evaluate = Callable[[CandidateGenome], CandidateFitness]


class PriorityReason(StrEnum):
    PINNED = "pinned"
    CACHED = "cached"
    ELITE_DESCENDANT = "elite_descendant"
    EARLIER_GENERATION = "earlier_generation"
    PROPOSAL_ORDER = "proposal_order"


class SchedulerModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class QueuedCandidate(SchedulerModel):
    """One candidate waiting to run, with why it sits where it does."""

    genome: CandidateGenome
    priority: float
    reasons: tuple[PriorityReason, ...] = ()
    explanation: str = ""
    attempts: int = Field(default=0, ge=0)

    def describe(self) -> str:
        return self.explanation or ", ".join(item.value for item in self.reasons)


class HumanOverride(BaseModel):
    """Step 72. Decisions a person may impose on a running search.

    Held separately from the config so that overriding does not rewrite the
    configuration a past decision was taken under. What a person did is
    recorded as its own fact.
    """

    model_config = ConfigDict(extra="forbid")

    pinned: set[str] = Field(default_factory=set)
    eliminated: set[str] = Field(default_factory=set)
    paused: bool = False

    def pin(self, candidate_id: str) -> None:
        """Force a candidate to be evaluated ahead of the queue."""
        self.eliminated.discard(candidate_id)
        self.pinned.add(candidate_id)

    def eliminate(self, candidate_id: str) -> None:
        """Refuse a candidate before it costs anything."""
        self.pinned.discard(candidate_id)
        self.eliminated.add(candidate_id)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def is_eliminated(self, candidate_id: str) -> bool:
        return candidate_id in self.eliminated


class CandidateScheduler:
    """Orders and runs a batch of candidates under concurrency and override."""

    def __init__(
        self,
        *,
        max_concurrent: int = 1,
        max_attempts: int = 1,
        override: HumanOverride | None = None,
        is_cached: Callable[[CandidateGenome], bool] | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.max_concurrent = max_concurrent
        self.max_attempts = max_attempts
        self.override = override or HumanOverride()
        self.is_cached = is_cached or (lambda genome: False)
        self.skipped: list[tuple[str, str]] = []

    # --- ordering ---------------------------------------------------------

    def prioritize(
        self, genomes: Sequence[CandidateGenome], *, elite_ids: set[str] | None = None
    ) -> list[QueuedCandidate]:
        """Order a batch, recording the reason for each position."""
        elites = elite_ids or set()
        queued: list[QueuedCandidate] = []
        for index, genome in enumerate(genomes):
            if self.override.is_eliminated(genome.genome_id):
                self.skipped.append((genome.genome_id, "eliminated by a person"))
                continue
            reasons: list[PriorityReason] = []
            score = 0.0
            if genome.genome_id in self.override.pinned:
                reasons.append(PriorityReason.PINNED)
                score += 1000.0
            if self.is_cached(genome):
                reasons.append(PriorityReason.CACHED)
                score += 100.0
            if set(genome.parent_genome_ids) & elites:
                reasons.append(PriorityReason.ELITE_DESCENDANT)
                score += 10.0
            reasons.append(PriorityReason.EARLIER_GENERATION)
            score += max(0.0, 5.0 - genome.generation)
            reasons.append(PriorityReason.PROPOSAL_ORDER)
            score -= index * 0.001
            queued.append(
                QueuedCandidate(
                    genome=genome,
                    priority=score,
                    reasons=tuple(reasons),
                    explanation=_explain(reasons, genome),
                )
            )
        return sorted(queued, key=lambda item: (-item.priority, item.genome.genome_id))

    # --- running ----------------------------------------------------------

    def run(
        self,
        queued: Sequence[QueuedCandidate],
        evaluate: Evaluate,
        *,
        should_continue: Callable[[], bool] | None = None,
    ) -> list[CandidateFitness]:
        """Evaluate in priority order, honouring concurrency and retries."""
        keep_going = should_continue or (lambda: True)
        if self.max_concurrent == 1:
            return self._sequential(queued, evaluate, keep_going)
        return self._concurrent(queued, evaluate, keep_going)

    def _sequential(
        self,
        queued: Sequence[QueuedCandidate],
        evaluate: Evaluate,
        keep_going: Callable[[], bool],
    ) -> list[CandidateFitness]:
        results: list[CandidateFitness] = []
        for item in queued:
            if not keep_going():
                self.skipped.append((item.genome.genome_id, "stop requested"))
                continue
            results.append(self._with_retries(item, evaluate))
        return results

    def _concurrent(
        self,
        queued: Sequence[QueuedCandidate],
        evaluate: Evaluate,
        keep_going: Callable[[], bool],
    ) -> list[CandidateFitness]:
        """Run several at once, preserving priority order in the results.

        Order is restored afterwards so that a concurrent run and a sequential
        one produce the same record, which is what keeps a search reproducible
        when the worker count changes.
        """
        admitted = [item for item in queued if keep_going()]
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {
                pool.submit(self._with_retries, item, evaluate): index
                for index, item in enumerate(admitted)
            }
            collected: dict[int, CandidateFitness] = {}
            for future in futures:
                collected[futures[future]] = future.result()
        return [collected[index] for index in sorted(collected)]

    def _with_retries(self, item: QueuedCandidate, evaluate: Evaluate) -> CandidateFitness:
        """Retry a failure up to `max_attempts`, then return it as data.

        Retrying is bounded because a deterministic failure retried forever
        spends the whole budget learning the same thing. The last failure is
        returned rather than raised, so the search continues (Step 54).
        """
        last: CandidateFitness | None = None
        for attempt in range(1, self.max_attempts + 1):
            fitness = evaluate(item.genome)
            if fitness.is_usable or fitness.status.value == "rejected":
                return fitness
            last = fitness
            if attempt < self.max_attempts:
                logger.info(
                    "retrying candidate %s after: %s",
                    item.genome.genome_id,
                    fitness.reason,
                )
        assert last is not None
        return last


def _explain(reasons: list[PriorityReason], genome: CandidateGenome) -> str:
    parts: list[str] = []
    for reason in reasons:
        if reason is PriorityReason.PINNED:
            parts.append("pinned by a person")
        elif reason is PriorityReason.CACHED:
            parts.append("already evaluated, so it costs nothing")
        elif reason is PriorityReason.ELITE_DESCENDANT:
            parts.append("descends from a retained candidate")
        elif reason is PriorityReason.EARLIER_GENERATION:
            parts.append(f"generation {genome.generation}")
    return "; ".join(parts) if parts else "proposal order"


__all__ = [
    "CandidateScheduler",
    "HumanOverride",
    "PriorityReason",
    "QueuedCandidate",
]
