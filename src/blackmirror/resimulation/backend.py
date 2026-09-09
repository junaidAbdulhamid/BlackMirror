"""Concrete adapter over the existing Phase 1/3/4/5/6 public pipelines."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from blackmirror.analytics.pipeline import analyze_run
from blackmirror.comparison.pipeline import compare_runs
from blackmirror.comparison.storage import ComparisonStore
from blackmirror.content.pipeline import analyze_content
from blackmirror.inference.service import InferenceService
from blackmirror.resimulation.schemas import ResimulationRequest, Stage, StageArtifact
from blackmirror.scoring.schemas import ExperimentScoreResult
from blackmirror.scoring.storage import ScoringStore


class ExistingPipelineBackend:
    """Runs existing pipelines; it never edits the supplied media."""

    def __init__(
        self,
        artifact_root: Path,
        inference: InferenceService | None = None,
        *,
        inference_factory: Callable[[], InferenceService] | None = None,
    ) -> None:
        if inference is None and inference_factory is None:
            raise ValueError("an inference service or a factory is required")
        self.artifact_root = Path(artifact_root)
        self._inference = inference
        self._factory = inference_factory

    @property
    def inference(self) -> InferenceService:
        """Constructed on first use, because most calls never reach a stage.

        Binding a run, reading its state and requesting a stop all go through
        this backend without running anything. Building the service eagerly
        would resolve a device and instantiate a model backend for each of
        them, which is work no read has any use for.
        """
        if self._inference is None:
            assert self._factory is not None
            self._inference = self._factory()
        return self._inference

    def infer(self, request: ResimulationRequest) -> tuple[str, StageArtifact]:
        result = self.inference.predict(Path(request.binding.variant_path))
        if result.run_id == request.parent_run_id:
            raise ValueError("inference returned the parent run for distinct variant media")
        if result.stimulus.sha256 != request.binding.variant_sha256:
            raise ValueError("inference stimulus hash differs from the bound variant")
        path = self.artifact_root / "runs" / result.run_id / "manifest.json"
        run_root = path.parent
        return result.run_id, _artifact(
            Stage.INFERENCE,
            result.run_id,
            path,
            {"variant_media": request.binding.variant_sha256},
            result.provenance.blackmirror_version,
            files=tuple(item for item in run_root.iterdir() if item.is_file()),
        )

    def analyze(self, run_id: str, upstream: dict[str, str]) -> StageArtifact:
        result = analyze_run(self.artifact_root, run_id)
        path = self.artifact_root / "runs" / run_id / "analytics" / "metadata.json"
        return _artifact(
            Stage.ANALYTICS,
            run_id,
            path,
            upstream,
            result.metadata.analytics_version,
            files=(
                path,
                self.artifact_root / "runs" / run_id / result.arrays["timeseries"].path,
                self.artifact_root / "runs" / run_id / result.arrays["atlas_mapping"].path,
            ),
        )

    def analyze_content(self, run_id: str, upstream: dict[str, str]) -> StageArtifact:
        result = analyze_content(run_id, artifact_root=self.artifact_root)
        if result.metadata.failed_stages:
            raise ValueError(
                "content analysis has failed stages: " + ", ".join(result.metadata.failed_stages)
            )
        path = self.artifact_root / "runs" / run_id / "content_analysis" / "metadata.json"
        return _artifact(
            Stage.CONTENT,
            run_id,
            path,
            upstream,
            result.metadata.analysis_version,
            files=(path, path.parent / result.arrays.path) if result.arrays else (path,),
        )

    def score(
        self, request: ResimulationRequest, run_id: str, upstream: dict[str, str]
    ) -> tuple[ExperimentScoreResult, StageArtifact]:
        source_score = ScoringStore(self.artifact_root).read(
            request.experiment_id, request.objective_set_hash
        )
        from blackmirror.scoring.engine import objective_set_hash

        if source_score.metadata.objective_definition_hash != request.objective_definition_hash:
            raise ValueError("source score objective definition hash differs from request")
        if objective_set_hash(source_score.objectives) != objective_set_hash(request.objectives):
            raise ValueError("declared objectives differ from the Phase 7 source score")
        if request.parent_run_id not in {item.variant_id for item in source_score.variant_scores}:
            raise ValueError("parent run is absent from the Phase 7 source score")
        # Local import avoids making a read-only API adapter part of base startup.
        from blackmirror.api.scoring_loader import ScoringLoader

        result = ScoringLoader(self.artifact_root).create(
            request.experiment_id,
            (request.parent_run_id, run_id),
            request.objectives,
            baseline_run_id=request.parent_run_id,
            reuse_cache=True,
        )
        path = (
            self.artifact_root
            / "experiments"
            / request.experiment_id
            / "scoring"
            / result.metadata.objective_set_hash
            / "metadata.json"
        )
        return result, _artifact(
            Stage.SCORING,
            result.metadata.objective_set_hash,
            path,
            upstream,
            result.metadata.scoring_version,
            files=tuple(item for item in path.parent.iterdir() if item.is_file()),
        )

    def compare(
        self, request: ResimulationRequest, run_id: str, upstream: dict[str, str]
    ) -> StageArtifact:
        comparison_id = f"phase8-{request.resimulation_id}"
        store = ComparisonStore(self.artifact_root)
        try:
            result = compare_runs(
                self.artifact_root,
                request.parent_run_id,
                (run_id,),
                comparison_id=comparison_id,
            )
        except FileExistsError:
            result = store.read(comparison_id)
        path = self.artifact_root / "comparisons" / result.comparison_id / "metadata.json"
        return _artifact(
            Stage.COMPARISON,
            result.comparison_id,
            path,
            upstream,
            result.metadata.comparison_version,
            files=tuple(item for item in path.parent.rglob("*") if item.is_file()),
        )

    def load_score(self, artifact: StageArtifact) -> ExperimentScoreResult:
        return ExperimentScoreResult.model_validate_json(
            Path(artifact.artifact_path).read_text(encoding="utf-8")
        )


def _artifact(
    stage: Stage,
    artifact_id: str,
    path: Path,
    upstream: dict[str, str],
    version: str,
    *,
    files: tuple[Path, ...] | None = None,
) -> StageArtifact:
    declared = files or (path,)
    hashes = {
        str(item.resolve()): hashlib.sha256(item.read_bytes()).hexdigest() for item in declared
    }
    return StageArtifact(
        stage=stage,
        artifact_id=artifact_id,
        artifact_path=str(path.resolve()),
        sha256=hashlib.sha256("".join(hashes[key] for key in sorted(hashes)).encode()).hexdigest(),
        artifact_files=hashes,
        upstream_sha256=upstream,
        pipeline_version=version,
    )
