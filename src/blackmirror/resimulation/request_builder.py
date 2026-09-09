"""Assemble a valid Phase 8 request from what Phase 6 and Phase 7 already stored.

WHY THIS MODULE EXISTS
    ``ResimulationRequest`` is deliberately hard to satisfy. It demands the exact
    objective set, its two hashes, a content-addressed media binding, and a
    per-hypothesis objective and direction map that cannot contradict the
    objective it cites. Those constraints are what stop a re-simulation from
    silently measuring something other than what was approved.

    They also make the request impossible to type by hand. Left that way, the
    only usable path would be to relax the schema, which is precisely the
    protection worth keeping. So the mapping is *derived* here from the stored
    Phase 7 hypotheses and the stored Phase 6 score, and never guessed. A
    caller supplies four things a human actually knows: which optimization,
    which approved candidate, which media file, and what to call the run.

WHAT IT REFUSES TO DO
    It does not invent a hypothesis-to-objective mapping when Phase 7 did not
    record one, it does not fall back to the target objective when a hypothesis
    cites a different one, and it does not accept media whose bytes match the
    parent. Each of those is an error naming the exact missing record.
"""

from __future__ import annotations

import json
from pathlib import Path

from blackmirror.optimization.schemas import (
    ExpectedDirection,
    OptimizationHypothesis,
    ProposedVariantSpec,
)
from blackmirror.optimization.storage import OptimizationStore
from blackmirror.resimulation.adapters import AdapterRegistry
from blackmirror.resimulation.schemas import ResimulationRequest
from blackmirror.scoring.schemas import ExperimentScoreResult
from blackmirror.scoring.storage import ScoringStore


class RequestBuildError(ValueError):
    """The request cannot be assembled from what is stored."""


def build_request(
    artifact_root: Path,
    *,
    resimulation_id: str,
    experiment_id: str,
    optimization_request_key: str,
    proposed_variant_id: str,
    variant_media: Path,
    source_media: Path | None = None,
    adapter_id: str = "user-supplied",
    parameters: dict[str, object] | None = None,
    iteration_index: int = 1,
    max_iterations: int = 1,
    max_attempts: int = 3,
    outcome_tolerance: float = 1e-9,
    previous_resimulation_id: str | None = None,
    registry: AdapterRegistry | None = None,
) -> ResimulationRequest:
    """Build a fully validated request for one approved candidate.

    ``source_media`` defaults to the stimulus the parent run was produced from,
    read out of that run's manifest, so the comparison is anchored to the exact
    bytes that generated the parent prediction rather than to a file that merely
    shares its name.
    """
    root = Path(artifact_root)
    store = OptimizationStore(root)
    try:
        optimization = store.read(experiment_id, optimization_request_key)
    except (OSError, ValueError) as exc:
        raise RequestBuildError(
            f"no stored optimization {optimization_request_key!r} for experiment "
            f"{experiment_id!r}; run Phase 7 first"
        ) from exc

    spec = _find_spec(store.specs(experiment_id, optimization_request_key), proposed_variant_id)
    score = _read_score(root, experiment_id, optimization.request.objective_set_hash)
    objectives = score.objectives
    objective_ids = {item.objective_id for item in objectives}

    hypotheses = {item.hypothesis_id: item for item in optimization.hypotheses}
    objective_map, direction_map = _hypothesis_maps(spec, hypotheses, objective_ids)

    parent_run_id = spec.parent_variant_id
    resolved_source = (
        Path(source_media) if source_media is not None else _parent_stimulus(root, parent_run_id)
    )
    binding = (registry or AdapterRegistry()).get(adapter_id).bind(
        resolved_source, Path(variant_media), dict(parameters or {})
    )

    return ResimulationRequest(
        resimulation_id=resimulation_id,
        experiment_id=experiment_id,
        optimization_request_key=optimization_request_key,
        parent_run_id=parent_run_id,
        proposed_variant=spec,
        hypothesis_objective_ids=objective_map,
        hypothesis_expected_directions=direction_map,
        binding=binding,
        objectives=objectives,
        objective_set_hash=score.metadata.objective_set_hash,
        objective_definition_hash=score.metadata.objective_definition_hash,
        previous_resimulation_id=previous_resimulation_id,
        iteration_index=iteration_index,
        max_iterations=max_iterations,
        max_attempts=max_attempts,
        outcome_tolerance=outcome_tolerance,
    )


def _find_spec(specs: list[ProposedVariantSpec], proposed_variant_id: str) -> ProposedVariantSpec:
    for spec in specs:
        if spec.proposed_variant_id == proposed_variant_id:
            return spec
    known = ", ".join(sorted(item.proposed_variant_id for item in specs)) or "none"
    raise RequestBuildError(
        f"no approved candidate specification {proposed_variant_id!r}; a candidate is "
        f"created only from an approved review. Available: {known}"
    )


def _read_score(
    root: Path, experiment_id: str, objective_set_hash: str
) -> ExperimentScoreResult:
    try:
        return ScoringStore(root).read(experiment_id, objective_set_hash)
    except (OSError, ValueError) as exc:
        raise RequestBuildError(
            f"the Phase 6 score this optimization was built from "
            f"({objective_set_hash[:16]!r}) is no longer readable; Phase 8 measures the "
            f"same objective set or it measures nothing"
        ) from exc


def _hypothesis_maps(
    spec: ProposedVariantSpec,
    hypotheses: dict[str, OptimizationHypothesis],
    objective_ids: set[str],
) -> tuple[dict[str, str], dict[str, ExpectedDirection]]:
    """Take each hypothesis's objective and direction from Phase 7's own record.

    A candidate can carry hypotheses about different objectives. Defaulting a
    missing one to the spec's target objective would quietly file its result
    under an objective it was never about, so a hypothesis Phase 7 did not
    record is an error rather than a default.
    """
    objective_map: dict[str, str] = {}
    direction_map: dict[str, ExpectedDirection] = {}
    for hypothesis_id in spec.hypothesis_ids:
        record = hypotheses.get(hypothesis_id)
        if record is None:
            # A synthesised id (build_spec emits one when no recommendation
            # carried a hypothesis) has no recorded objective of its own.
            objective_map[hypothesis_id] = spec.target_objective_id
            direction_map[hypothesis_id] = spec.expected_direction
            continue
        if record.objective_id not in objective_ids:
            raise RequestBuildError(
                f"hypothesis {hypothesis_id!r} targets objective {record.objective_id!r}, "
                f"which is absent from the stored score's objective set"
            )
        objective_map[hypothesis_id] = record.objective_id
        direction_map[hypothesis_id] = record.expected_direction
    if spec.target_objective_id not in objective_ids:
        raise RequestBuildError(
            f"candidate targets objective {spec.target_objective_id!r}, which is absent "
            f"from the stored score's objective set"
        )
    return objective_map, direction_map


def _parent_stimulus(root: Path, parent_run_id: str) -> Path:
    manifest = root / "runs" / parent_run_id / "manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return Path(str(payload["stimulus"]["path"]))
    except (OSError, KeyError, ValueError) as exc:
        raise RequestBuildError(
            f"cannot read the stimulus path of parent run {parent_run_id!r} from "
            f"{manifest}; pass source_media explicitly"
        ) from exc


__all__ = ["RequestBuildError", "build_request"]
