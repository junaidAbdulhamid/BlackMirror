"""Steps 11 and 50: turn a genome into a measured fitness, through Phase 8.

THE CHAIN
    genome -> spec + plan -> materialized file -> Phase 8 request ->
    inference, analytics, content, scoring, comparison -> objective deltas ->
    fitness

    Every expensive step here is Phase 8's. This module builds the request,
    hands it over, and reads the result; it re-implements no part of the
    pipeline. That is deliberate: a second scoring path would eventually
    disagree with the first, and there would be no way to tell which number was
    the real one.

WHY THE REQUEST IS BUILT HERE RATHER THAN BY `request_builder`
    That helper derives the hypothesis-to-objective map from a stored Phase 7
    optimization. A search candidate has none: it was proposed by a strategy,
    not approved from a recommendation. So the maps come from the decoder
    instead, and `ResimulationRequest`'s own validators still enforce every
    consistency rule, including that a hypothesis cannot contradict the
    direction of the objective it names.

    `optimization_request_key` carries `search-<search_id>` rather than a Phase 7
    key. The field is a free string and this is the honest value for it; what
    marks the candidate as machine-proposed is the binding's adapter id,
    `ffmpeg-materialized`, which no human-supplied variant ever carries.

WHY A FAILURE IS RETURNED, NOT RAISED
    Step 54. One candidate failing must not end a search that has already spent
    hours. Every failure path here produces a `CandidateFitness` carrying the
    reason and whatever compute was consumed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from blackmirror.materialization.adapter import binding_from_result
from blackmirror.materialization.ffmpeg import MaterializationError, materialize
from blackmirror.resimulation.orchestrator import ResimulationOrchestrator
from blackmirror.resimulation.schemas import (
    ResimulationRequest,
    ResimulationResult,
    RunStatus,
)
from blackmirror.scoring.schemas import NeuralObjective
from blackmirror.search.decode import DecodeError, GenomeDecoder
from blackmirror.search.fitness import (
    CandidateFitness,
    ComputeCost,
    FitnessStatus,
    directional_fitness,
    failed,
    rejected,
)
from blackmirror.search.genome import CandidateGenome
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Called with (genome, resimulation_id) before the expensive work starts, so a
#: caller can checkpoint the intent to evaluate.
EvaluationHook = Callable[[CandidateGenome, str], None]


class CandidateEvaluator:
    """Evaluates one candidate end to end and reports what it cost."""

    def __init__(
        self,
        artifact_root: Path,
        decoder: GenomeDecoder,
        orchestrator: ResimulationOrchestrator,
        *,
        objectives: tuple[NeuralObjective, ...],
        objective_set_hash: str,
        objective_definition_hash: str,
        candidate_dir: Path | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.decoder = decoder
        self.orchestrator = orchestrator
        self.objectives = objectives
        self.objective_set_hash = objective_set_hash
        self.objective_definition_hash = objective_definition_hash
        self.candidate_dir = Path(
            candidate_dir
            or self.artifact_root / "search" / decoder.search_id / "candidates"
        )
        self.max_attempts = max_attempts

    # --- identity ---------------------------------------------------------

    def resimulation_id(self, genome: CandidateGenome) -> str:
        return f"{self.decoder.search_id}-{genome.genome_id}"

    # --- the chain --------------------------------------------------------

    def evaluate(
        self, genome: CandidateGenome, *, on_start: EvaluationHook | None = None
    ) -> CandidateFitness:
        """Run the full chain. Never raises for an expected failure."""
        started = time.perf_counter()
        candidate_id = genome.genome_id

        # 1. Decode. Constraint violations and neutral genomes cost nothing to
        #    catch here and an hour of inference to catch later.
        try:
            spec = self.decoder.to_spec(genome)
            plan = self.decoder.to_plan(genome)
        except (DecodeError, ValueError) as exc:
            return rejected(genome, f"{type(exc).__name__}: {exc}")

        # 2. Materialize. Reuse is content-addressed, so an identical plan is
        #    free rather than a second encode.
        try:
            produced = materialize(plan, self.candidate_dir)
            self.decoder.check_duration(produced.variant_duration_seconds)
        except (MaterializationError, DecodeError, OSError) as exc:
            return rejected(genome, f"{type(exc).__name__}: {exc}")

        resimulation_id = self.resimulation_id(genome)
        if on_start is not None:
            on_start(genome, resimulation_id)

        # 3. Build the Phase 8 request. Its validators are the contract.
        try:
            request = self._request(genome, spec, produced, resimulation_id)
        except ValueError as exc:
            return rejected(genome, f"request is not valid: {exc}")

        # 4. Evaluate. This is the expensive part.
        try:
            result = self.orchestrator.start(request)
        except Exception as exc:
            logger.warning("candidate %s failed: %s", candidate_id, exc)
            return failed(
                genome,
                f"{type(exc).__name__}: {exc}",
                compute=ComputeCost(
                    wall_seconds=time.perf_counter() - started, tribe_runs=1
                ),
                resimulation_id=resimulation_id,
            )

        compute = ComputeCost(
            wall_seconds=time.perf_counter() - started,
            tribe_runs=0 if produced.reused_existing else 1,
            served_from_cache=produced.reused_existing,
        )
        return self._fitness(genome, result, produced, compute)

    # --- internals --------------------------------------------------------

    def _request(
        self,
        genome: CandidateGenome,
        spec: object,
        produced: object,
        resimulation_id: str,
    ) -> ResimulationRequest:
        hypothesis_id = self.decoder.hypothesis_id(genome)
        return ResimulationRequest(
            resimulation_id=resimulation_id,
            experiment_id=self.decoder.experiment_id,
            # Not a Phase 7 key: this candidate came from a search, and saying
            # so is more useful than borrowing an optimization id it never had.
            optimization_request_key=f"search-{self.decoder.search_id}",
            parent_run_id=self.decoder.parent_variant_id,
            proposed_variant=spec,  # type: ignore[arg-type]
            hypothesis_objective_ids={hypothesis_id: self.decoder.objective.objective_id},
            hypothesis_expected_directions={hypothesis_id: self.decoder.expected_direction},
            binding=binding_from_result(produced),  # type: ignore[arg-type]
            objectives=self.objectives,
            objective_set_hash=self.objective_set_hash,
            objective_definition_hash=self.objective_definition_hash,
            max_attempts=self.max_attempts,
        )

    def _fitness(
        self,
        genome: CandidateGenome,
        result: ResimulationResult,
        produced: object,
        compute: ComputeCost,
    ) -> CandidateFitness:
        variant_path = str(getattr(produced, "variant_path", "")) or None
        variant_sha = getattr(produced, "variant_sha256", None)
        duration_change = float(getattr(produced, "duration_change_seconds", 0.0))
        common = {
            "candidate_id": genome.genome_id,
            "genome": genome,
            "candidate_run_id": result.candidate_run_id,
            "resimulation_id": result.request.resimulation_id,
            "variant_path": variant_path,
            "variant_sha256": variant_sha,
            "duration_change_seconds": duration_change,
            "compute": compute,
        }

        if result.status is not RunStatus.COMPLETED:
            reason = (
                result.failures[-1].message
                if result.failures
                else f"ended as {result.status.value}"
            )
            return CandidateFitness(
                status=FitnessStatus.EVALUATION_FAILED,
                reason=f"{result.status.value}: {reason}",
                **common,  # type: ignore[arg-type]
            )

        raw = {
            delta.objective_id: delta.candidate_raw_value
            for delta in result.objective_deltas
        }
        normalized = {
            delta.objective_id: delta.candidate_score
            for delta in result.objective_deltas
        }
        primary = self.decoder.objective
        scalar = directional_fitness(raw.get(primary.objective_id), primary)

        if scalar is None:
            return CandidateFitness(
                status=FitnessStatus.NOT_MEASURABLE,
                raw_objectives=raw,
                normalized_objectives=normalized,
                reason=(
                    f"objective {primary.objective_id} had no valid value for this "
                    f"candidate, so it cannot be ranked"
                ),
                **common,  # type: ignore[arg-type]
            )

        return CandidateFitness(
            status=FitnessStatus.EVALUATED,
            raw_objectives=raw,
            normalized_objectives=normalized,
            scalar_fitness=scalar,
            primary_objective_id=primary.objective_id,
            **common,  # type: ignore[arg-type]
        )


__all__ = ["CandidateEvaluator", "EvaluationHook"]
