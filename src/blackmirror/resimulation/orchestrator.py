"""Dependency-injected, checkpointed Phase 8 state machine."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Protocol

from blackmirror.resimulation.evaluation import evaluate_objectives
from blackmirror.resimulation.schemas import (
    FailedAttempt,
    HypothesisOutcome,
    HypothesisVerdict,
    Outcome,
    ResimulationConflict,
    ResimulationRequest,
    ResimulationResult,
    RunStatus,
    Stage,
    StageArtifact,
    StopReason,
)
from blackmirror.resimulation.storage import ResimulationStore
from blackmirror.scoring.schemas import ExperimentScoreResult


class ResimulationBackend(Protocol):
    """Mockable adapters around existing Phase 1/3/4/5/6 pipelines."""

    def infer(self, request: ResimulationRequest) -> tuple[str, StageArtifact]: ...
    def analyze(self, run_id: str, upstream: dict[str, str]) -> StageArtifact: ...
    def analyze_content(self, run_id: str, upstream: dict[str, str]) -> StageArtifact: ...
    def score(
        self, request: ResimulationRequest, run_id: str, upstream: dict[str, str]
    ) -> tuple[ExperimentScoreResult, StageArtifact]: ...
    def compare(
        self, request: ResimulationRequest, run_id: str, upstream: dict[str, str]
    ) -> StageArtifact: ...
    def load_score(self, artifact: StageArtifact) -> ExperimentScoreResult: ...


class _StopRequested(Exception):
    def __init__(self, reason: StopReason) -> None:
        self.reason = reason


class ResimulationOrchestrator:
    def __init__(
        self,
        artifact_root: Path,
        backend: ResimulationBackend,
        *,
        store: ResimulationStore | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.backend = backend
        self.store = store or ResimulationStore(artifact_root)

    def start(self, request: ResimulationRequest) -> ResimulationResult:
        self._verify_binding(request)
        with self.store.lock(request.resimulation_id):
            bound = self._bind(request)
            if bound.status is RunStatus.COMPLETED:
                return bound
            return self._run(bound)

    def prepare(self, request: ResimulationRequest) -> ResimulationResult:
        """Durably bind a request without executing any stage.

        Split out of `start` for callers that cannot block. A full pass runs
        TRIBE inference and takes hours, so an HTTP request that waited for it
        would time out long before the first stage finished. `prepare` writes
        exactly the checkpoint `start` would write, then returns; executing it
        afterwards is the ordinary idempotent-resume path the state machine
        already guarantees, not a second code path.
        """
        self._verify_binding(request)
        with self.store.lock(request.resimulation_id):
            return self._bind(request)

    def _bind(self, request: ResimulationRequest) -> ResimulationResult:
        """Return the durable state for this request, creating it if absent.

        Rebinding an existing id to different inputs is refused rather than
        overwritten: the id is how every artifact, checkpoint and outcome is
        addressed, and silently repointing it would leave finished work
        attributed to a request that did not produce it.
        """
        cache_key = _cache_key(request)
        if self.store.exists(request.resimulation_id):
            existing = self.store.read(request.resimulation_id)
            if existing.cache_key != cache_key:
                raise ResimulationConflict(
                    "resimulation id is already bound to different inputs"
                )
            self._verify_artifacts(existing)
            return existing
        initial = ResimulationResult(request=request, cache_key=cache_key)
        self.store.checkpoint(initial)
        return initial

    def resume(self, resimulation_id: str) -> ResimulationResult:
        with self.store.lock(resimulation_id):
            result = self.store.recover(resimulation_id)
            if result.cache_key != _cache_key(result.request):
                raise ValueError("persisted request fingerprint changed")
            self._verify_binding(result.request)
            self._verify_artifacts(result)
            if result.status is RunStatus.COMPLETED:
                return result
            return self._run(
                result.model_copy(update={"status": RunStatus.ACTIVE, "stop_reason": None})
            )

    def stop(self, resimulation_id: str, reason: StopReason) -> ResimulationResult:
        if reason not in {StopReason.CANCELLED, StopReason.TIMEOUT, StopReason.RESOURCE_LIMIT}:
            raise ValueError("invalid external stop reason")
        current = self.store.read(resimulation_id)
        if current.status is RunStatus.COMPLETED:
            raise ValueError("a completed resimulation cannot be stopped retroactively")
        self.store.request_stop(resimulation_id, reason)
        requested = current.model_copy(
            update={
                "status": RunStatus.STOP_REQUESTED,
                "requested_stop_reason": reason,
                "updated_at": _now(),
            }
        )
        self.store.checkpoint(requested)
        return requested

    def _run(self, result: ResimulationResult) -> ResimulationResult:
        if result.attempt_count >= result.request.max_attempts:
            stopped = result.model_copy(
                update={"status": RunStatus.STOPPED, "stop_reason": StopReason.MAX_ATTEMPTS}
            )
            self.store.checkpoint(stopped)
            return stopped
        attempt = result.attempt_count + 1
        try:
            self._check_stop(result.request.resimulation_id)
            if result.current_stage < Stage.INFERENCE:
                run_id, artifact = self.backend.infer(result.request)
                result = self._advance(result, Stage.INFERENCE, artifact, candidate_run_id=run_id)
                self._check_stop(result.request.resimulation_id)
            assert result.candidate_run_id is not None
            candidate_run_id = result.candidate_run_id
            upstream = self._upstream(result)
            if result.current_stage < Stage.ANALYTICS:
                result = self._advance(
                    result, Stage.ANALYTICS, self.backend.analyze(candidate_run_id, upstream)
                )
                upstream = self._upstream(result)
                self._check_stop(result.request.resimulation_id)
            if result.current_stage < Stage.CONTENT:
                result = self._advance(
                    result,
                    Stage.CONTENT,
                    self.backend.analyze_content(candidate_run_id, upstream),
                )
                upstream = self._upstream(result)
                self._check_stop(result.request.resimulation_id)
            score: ExperimentScoreResult | None = None
            if result.current_stage < Stage.SCORING:
                score, artifact = self.backend.score(result.request, candidate_run_id, upstream)
                if (
                    score.metadata.objective_definition_hash
                    != result.request.objective_definition_hash
                ):
                    raise ValueError("produced score used a different objective definition")
                result = self._advance(result, Stage.SCORING, artifact)
                result = result.model_copy(
                    update={
                        "source_scoring_cache_key": result.request.objective_set_hash,
                        "produced_scoring_cache_key": score.metadata.objective_set_hash,
                        "produced_objective_definition_hash": (
                            score.metadata.objective_definition_hash
                        ),
                        "produced_scoring_input_sha256": score.metadata.input_artifact_sha256,
                    }
                )
                self.store.checkpoint(result)
                upstream = self._upstream(result)
                self._check_stop(result.request.resimulation_id)
            else:
                score_artifact = next(
                    item for item in result.artifacts if item.stage is Stage.SCORING
                )
                score = self.backend.load_score(score_artifact)
            if result.current_stage < Stage.COMPARISON:
                result = self._advance(
                    result,
                    Stage.COMPARISON,
                    self.backend.compare(result.request, candidate_run_id, upstream),
                )
                self._check_stop(result.request.resimulation_id)
            if score is None:
                raise RuntimeError("resume requires backend score reconstruction before evaluation")
            deltas = evaluate_objectives(
                score,
                parent_run_id=result.request.parent_run_id,
                candidate_run_id=candidate_run_id,
                tolerance=result.request.outcome_tolerance,
            )
            by_objective = {item.objective_id: item.outcome for item in deltas}
            hypotheses = tuple(
                HypothesisOutcome(
                    hypothesis_id=item,
                    expected_direction=result.request.hypothesis_expected_directions.get(
                        item, result.request.proposed_variant.expected_direction
                    ),
                    measured_outcome=(
                        measured := by_objective.get(
                            result.request.hypothesis_objective_ids.get(
                                item, result.request.proposed_variant.target_objective_id
                            ),
                            Outcome.INCONCLUSIVE,
                        )
                    ),
                    verdict=(
                        HypothesisVerdict.PASS
                        if measured is Outcome.IMPROVED
                        else HypothesisVerdict.FAIL
                        if measured is Outcome.WORSENED
                        else HypothesisVerdict.INCONCLUSIVE
                    ),
                    objective_id=result.request.hypothesis_objective_ids.get(
                        item, result.request.proposed_variant.target_objective_id
                    ),
                    note="Outcome is a model-objective comparison, not a causal or human effect.",
                )
                for item in result.request.proposed_variant.hypothesis_ids
            )
            completed = result.model_copy(
                update={
                    "current_stage": Stage.EVALUATED,
                    "status": RunStatus.COMPLETED,
                    "attempt_count": attempt,
                    "objective_deltas": deltas,
                    "hypothesis_outcomes": hypotheses,
                    "stop_reason": StopReason.COMPLETED,
                    "updated_at": _now(),
                }
            )
            self.store.checkpoint(completed)
            return completed
        except _StopRequested as exc:
            stopped = result.model_copy(
                update={
                    "status": RunStatus.STOPPED,
                    "attempt_count": attempt,
                    "stop_reason": exc.reason,
                    "requested_stop_reason": exc.reason,
                    "updated_at": _now(),
                }
            )
            self.store.checkpoint(stopped)
            self.store.clear_stop(result.request.resimulation_id)
            return stopped
        except Exception as exc:
            failed = result.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "attempt_count": attempt,
                    "stop_reason": StopReason.FAILED_STAGE,
                    "failures": (
                        *result.failures,
                        FailedAttempt(
                            attempt=attempt,
                            stage=Stage(min(int(result.current_stage) + 1, int(Stage.EVALUATED))),
                            error_type=type(exc).__name__,
                            message=str(exc),
                        ),
                    ),
                    "updated_at": _now(),
                }
            )
            self.store.checkpoint(failed)
            return failed

    def _check_stop(self, resimulation_id: str) -> None:
        reason = self.store.stop_requested(resimulation_id)
        if reason is not None:
            raise _StopRequested(reason)

    def _advance(
        self,
        result: ResimulationResult,
        stage: Stage,
        artifact: StageArtifact,
        *,
        candidate_run_id: str | None = None,
    ) -> ResimulationResult:
        if stage != Stage(int(result.current_stage) + 1) or artifact.stage is not stage:
            raise ValueError("stage transition or artifact stage is not monotonic")
        self._verify_artifact(artifact, self._upstream(result))
        advanced = result.model_copy(
            update={
                "current_stage": stage,
                "candidate_run_id": candidate_run_id or result.candidate_run_id,
                "artifacts": (*result.artifacts, artifact),
                "updated_at": _now(),
            }
        )
        self.store.checkpoint(advanced)
        return advanced

    def _verify_artifacts(self, result: ResimulationResult) -> None:
        upstream = {"variant_media": result.request.binding.variant_sha256}
        for artifact in result.artifacts:
            self._verify_artifact(artifact, upstream)
            upstream[artifact.stage.name.lower()] = artifact.sha256

    @staticmethod
    def _verify_binding(request: ResimulationRequest) -> None:
        for label, path_text, expected in (
            ("source", request.binding.source_path, request.binding.source_sha256),
            ("variant", request.binding.variant_path, request.binding.variant_sha256),
        ):
            path = Path(path_text)
            if not path.is_file() or _file_hash(path) != expected:
                raise ValueError(f"{label} media fingerprint changed")

    def _verify_artifact(self, artifact: StageArtifact, expected: dict[str, str]) -> None:
        path = Path(artifact.artifact_path).resolve()
        if self.artifact_root not in path.parents:
            raise ValueError("stage artifact is not owned by the configured artifact root")
        if artifact.upstream_sha256 != expected:
            raise ValueError(f"{artifact.stage.name} upstream fingerprint differs")
        actual_files = {}
        for path_text, checksum in artifact.artifact_files.items():
            item = Path(path_text).resolve()
            if self.artifact_root not in item.parents:
                raise ValueError("declared stage file is not source-owned")
            actual = _file_hash(item)
            if actual != checksum:
                raise ValueError(f"{artifact.stage.name} artifact checksum differs")
            actual_files[str(item)] = actual
        if _artifact_digest(actual_files) != artifact.sha256:
            raise ValueError(f"{artifact.stage.name} artifact-set checksum differs")

    @staticmethod
    def _upstream(result: ResimulationResult) -> dict[str, str]:
        values = {"variant_media": result.request.binding.variant_sha256}
        values.update({item.stage.name.lower(): item.sha256 for item in result.artifacts})
        return values


def _cache_key(request: ResimulationRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"resimulation_id"})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_digest(files: dict[str, str]) -> str:
    return hashlib.sha256("".join(files[key] for key in sorted(files)).encode()).hexdigest()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
