"""Read/create Phase 6 scores over persisted artifacts.

Scoring is cheap next to inference, so the create path recomputes rather than
queueing. The one thing it will not do is re-run TRIBE: every variant must
already be a completed run with analytics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blackmirror.scoring.engine import GoalConditionedScoringEngine, objective_set_hash
from blackmirror.scoring.explanation import explain
from blackmirror.scoring.loader import VariantLoadError, load_variant
from blackmirror.scoring.metrics import default_registry
from blackmirror.scoring.schemas import (
    ExperimentScoreResult,
    NeuralObjective,
    ScoreExplanation,
)
from blackmirror.scoring.storage import ScoringStore
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class ScoringLoader:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store = ScoringStore(self.artifact_root)
        self.engine = GoalConditionedScoringEngine()

    def metrics(self) -> dict[str, dict[str, str]]:
        """Every registered metric with its formula and limitations."""
        registry = default_registry()
        return {
            name: description
            for name, description in registry.describe_all().items()
            if not registry.get(name).requires_reference
        }

    def experiments(self) -> list[str]:
        return self.store.list_experiments()

    def scores(self, experiment_id: str) -> list[ExperimentScoreResult]:
        return self.store.list_for_experiment(experiment_id)

    def score(self, experiment_id: str, objective_set_hash_value: str) -> ExperimentScoreResult:
        return self.store.read(experiment_id, objective_set_hash_value)

    def explanation(
        self, experiment_id: str, objective_set_hash_value: str
    ) -> ScoreExplanation | None:
        payload = self.store.read_explanation(experiment_id, objective_set_hash_value)
        return ScoreExplanation.model_validate_json(payload) if payload else None

    def create(
        self,
        experiment_id: str,
        run_ids: tuple[str, ...],
        objectives: tuple[NeuralObjective, ...],
        *,
        baseline_run_id: str | None = None,
        with_networks: bool = False,
        reuse_cache: bool = True,
    ) -> ExperimentScoreResult:
        """Score an explicit set of completed runs against an objective set."""
        # Validate the objective set BEFORE touching the artifact store. Loading
        # variants reads every predictions.npy, so an unknown metric would
        # otherwise cost that work and then surface as whatever the loader
        # complained about first -- a missing run, say -- which is the wrong
        # error entirely.
        registry = default_registry()
        for objective in objectives:
            metric = registry.get(objective.metric)
            if metric.requires_reference:
                raise ValueError(
                    f"metric {objective.metric} requires a persisted reference-pattern "
                    "contract, which this API does not currently accept"
                )
            if metric.requires_baseline and baseline_run_id is None:
                raise ValueError(f"metric {objective.metric} requires baseline_run_id")

        with_networks = with_networks or any(
            objective.target.type.value == "network" for objective in objectives
        )

        fingerprints = {
            run_id: _input_fingerprint(self.artifact_root, run_id, with_networks=with_networks)
            for run_id in run_ids
        }
        digest = objective_set_hash(
            objectives,
            cache_context={
                "run_ids": run_ids,
                "baseline_run_id": baseline_run_id,
                "with_networks": with_networks,
                "input_artifact_sha256": fingerprints,
            },
        )
        if reuse_cache and self.store.exists(experiment_id, digest):
            # Scoring is deterministic, so an identical request has an identical
            # answer; returning the stored one keeps the citation stable.
            logger.info("Reusing stored score for %s/%s", experiment_id, digest[:16])
            return self.store.read(experiment_id, digest)

        variants = tuple(
            load_variant(self.artifact_root, run_id, with_networks=with_networks)
            for run_id in run_ids
        )
        result = self.engine.score_experiment(
            experiment_id,
            variants,
            objectives,
            baseline_variant_id=baseline_run_id,
            cache_key=digest,
            input_artifact_sha256=fingerprints,
        )
        explanation = explain(
            result, content_descriptions=self._descriptions(run_ids)
        )
        # A score path is immutable scientific evidence. ``reuse_cache=False``
        # may recompute it for verification, but must never replace the cited
        # artifact (including its creation timestamp or explanation).
        try:
            self.store.write(
                result, explanation_json=explanation.model_dump_json(indent=2)
            )
        except FileExistsError:
            return self.store.read(experiment_id, digest)
        return result

    def _descriptions(self, run_ids: tuple[str, ...]) -> dict[str, list[dict[str, object]]]:
        """Phase 4 events per run, for the explanation's content context."""
        output: dict[str, list[dict[str, object]]] = {}
        for run_id in run_ids:
            path = (
                self.artifact_root / "runs" / run_id / "content_analysis" / "metadata.json"
            )
            if not path.is_file():
                continue
            try:
                output[run_id] = json.loads(path.read_text(encoding="utf-8")).get("events", [])
            except (OSError, json.JSONDecodeError):
                logger.warning("Unreadable content analysis for %s; context omitted", run_id)
        return output


__all__ = ["ScoringLoader", "VariantLoadError"]


def _input_fingerprint(artifact_root: Path, run_id: str, *, with_networks: bool) -> str:
    """Hash every persisted input whose change can alter a score or explanation."""
    run_root = Path(artifact_root) / "runs" / run_id
    paths = [
        run_root / "manifest.json",
        run_root / "predictions.npy",
        run_root / "analytics" / "metadata.json",
        run_root / "analytics" / "timeseries.npz",
        run_root / "analytics" / "atlas_mapping.npz",
    ]
    if not run_root.is_dir():
        raise VariantLoadError(f"no run directory for {run_id}")
    content = run_root / "content_analysis" / "metadata.json"
    if content.is_file():
        paths.append(content)
    if with_networks:
        paths.append(Path(artifact_root) / "atlases" / "yeo2011_fsaverage5" / "manifest.json")
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise VariantLoadError(f"run {run_id} lacks required scoring input {path.name}")
        digest.update(str(path.relative_to(artifact_root)).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
