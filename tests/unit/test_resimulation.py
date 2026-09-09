from __future__ import annotations

import datetime as dt
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from blackmirror.optimization.schemas import (
    ContentIntervention,
    ExpectedDirection,
    InterventionType,
    OptimizationConstraints,
    ProposedVariantSpec,
)
from blackmirror.resimulation.adapters import AdapterRegistry, UserSuppliedVariantAdapter
from blackmirror.resimulation.evaluation import evaluate_objectives
from blackmirror.resimulation.orchestrator import ResimulationOrchestrator
from blackmirror.resimulation.schemas import (
    Outcome,
    ResimulationRequest,
    RunStatus,
    Stage,
    StageArtifact,
    StopReason,
)
from blackmirror.resimulation.storage import ResimulationStore
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)


def _request(tmp_path: Path, *, attempts: int = 2) -> ResimulationRequest:
    source = tmp_path / "source.mp4"
    variant = tmp_path / "variant.mp4"
    source.write_bytes(b"source")
    variant.write_bytes(b"variant")
    binding = UserSuppliedVariantAdapter().bind(source, variant, {"review": "approved"})
    objective = NeuralObjective(
        objective_id="objective",
        name="Measured response",
        metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE,
    )
    from blackmirror.scoring.engine import objective_set_hash

    intervention = ContentIntervention(
        intervention_id="I1",
        type=InterventionType.VISUAL_EDIT,
        target_interval_start=0,
        target_interval_end=1,
        description="User supplied a revised visual interval.",
        rationale="A measured variant differed in this interval.",
        evidence_ids=("E1",),
        objective_id="objective",
        expected_direction=ExpectedDirection.TEST_FOR_INCREASE,
    )
    spec = ProposedVariantSpec(
        proposed_variant_id="candidate",
        parent_variant_id="parent",
        experiment_id="experiment",
        hypothesis_ids=("H1",),
        interventions=(intervention,),
        constraints=OptimizationConstraints(),
        target_objective_id="objective",
        expected_direction=ExpectedDirection.TEST_FOR_INCREASE,
    )
    return ResimulationRequest(
        resimulation_id="loop-1",
        experiment_id="experiment",
        optimization_request_key="opt-key",
        parent_run_id="parent",
        proposed_variant=spec,
        binding=binding,
        objectives=(objective,),
        objective_set_hash="a" * 64,
        objective_definition_hash=objective_set_hash((objective,)),
        max_iterations=attempts,
        max_attempts=attempts,
    )


class FakeBackend:
    def __init__(self, root: Path, *, fail_once: Stage | None = None) -> None:
        self.root = root
        self.fail_once = fail_once
        self.calls: list[Stage] = []
        self.score_result = _score()

    def _artifact(self, stage: Stage, upstream: dict[str, str]) -> StageArtifact:
        self.calls.append(stage)
        if self.fail_once is stage:
            self.fail_once = None
            raise RuntimeError(f"failed {stage.name}")
        path = self.root / "generated" / f"{stage.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stage.name, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_files = {str(path.resolve()): digest}
        return StageArtifact(
            stage=stage,
            artifact_id=stage.name,
            artifact_path=str(path),
            sha256=hashlib.sha256(digest.encode()).hexdigest(),
            artifact_files=artifact_files,
            upstream_sha256=upstream,
            pipeline_version="test-1",
        )

    def infer(self, request: ResimulationRequest) -> tuple[str, StageArtifact]:
        return "candidate-run", self._artifact(
            Stage.INFERENCE, {"variant_media": request.binding.variant_sha256}
        )

    def analyze(self, run_id: str, upstream: dict[str, str]) -> StageArtifact:
        return self._artifact(Stage.ANALYTICS, upstream)

    def analyze_content(self, run_id: str, upstream: dict[str, str]) -> StageArtifact:
        return self._artifact(Stage.CONTENT, upstream)

    def score(self, request: ResimulationRequest, run_id: str, upstream: dict[str, str]):
        self.score_result.metadata.objective_definition_hash = request.objective_definition_hash
        return self.score_result, self._artifact(Stage.SCORING, upstream)

    def compare(self, request: ResimulationRequest, run_id: str, upstream: dict[str, str]):
        return self._artifact(Stage.COMPARISON, upstream)

    def load_score(self, artifact: StageArtifact):
        return self.score_result


def _score() -> object:
    objective = SimpleNamespace(
        objective_id="objective",
        direction=ObjectiveDirection.MAXIMIZE,
        normalization=SimpleNamespace(target_value=None),
    )
    parent = SimpleNamespace(
        variant_id="parent",
        objective_scores=(
            SimpleNamespace(objective_id="objective", raw_value=1.0, score=0.2, valid=True),
        ),
    )
    candidate = SimpleNamespace(
        variant_id="candidate-run",
        objective_scores=(
            SimpleNamespace(objective_id="objective", raw_value=2.0, score=0.8, valid=True),
        ),
    )
    return SimpleNamespace(
        objectives=(objective,),
        variant_scores=(parent, candidate),
        metadata=SimpleNamespace(
            objective_set_hash="b" * 64,
            objective_definition_hash="c" * 64,
            input_artifact_sha256={"parent": "d" * 64, "candidate-run": "e" * 64},
        ),
    )


def test_user_supplied_adapter_hashes_content_and_rejects_identity(tmp_path: Path) -> None:
    source = tmp_path / "a.mp4"
    variant = tmp_path / "b.mp4"
    source.write_bytes(b"same")
    variant.write_bytes(b"same")
    with pytest.raises(ValueError, match="must differ"):
        UserSuppliedVariantAdapter().bind(source, variant, {})
    with pytest.raises(ValueError, match="unregistered"):
        AdapterRegistry().get("external-generator")


def test_full_state_machine_is_monotonic_content_addressed_and_idempotent(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = FakeBackend(tmp_path)
    orchestrator = ResimulationOrchestrator(tmp_path, backend)
    result = orchestrator.start(request)
    assert result.status is RunStatus.COMPLETED
    assert [item.stage for item in result.artifacts] == list(Stage)[1:-1]
    assert result.objective_deltas[0].outcome is Outcome.IMPROVED
    assert result.hypothesis_outcomes[0].measured_outcome is Outcome.IMPROVED
    assert result.hypothesis_outcomes[0].verdict.value == "pass"
    assert orchestrator.start(request) == result


def test_failure_preserves_prior_stages_and_resume_appends_attempt(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = FakeBackend(tmp_path, fail_once=Stage.SCORING)
    orchestrator = ResimulationOrchestrator(tmp_path, backend)
    failed = orchestrator.start(request)
    assert failed.status is RunStatus.FAILED
    assert failed.current_stage is Stage.CONTENT
    assert [item.stage for item in failed.artifacts] == [
        Stage.INFERENCE,
        Stage.ANALYTICS,
        Stage.CONTENT,
    ]
    completed = orchestrator.resume(request.resimulation_id)
    assert completed.status is RunStatus.COMPLETED
    assert len(completed.failures) == 1
    assert backend.calls.count(Stage.INFERENCE) == 1


def test_resume_rejects_changed_upstream_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = FakeBackend(tmp_path, fail_once=Stage.ANALYTICS)
    orchestrator = ResimulationOrchestrator(tmp_path, backend)
    failed = orchestrator.start(request)
    Path(failed.artifacts[0].artifact_path).write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        orchestrator.resume(request.resimulation_id)


def test_resume_rejects_changed_variant_media(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = FakeBackend(tmp_path, fail_once=Stage.INFERENCE)
    orchestrator = ResimulationOrchestrator(tmp_path, backend)
    orchestrator.start(request)
    Path(request.binding.variant_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="variant media fingerprint changed"):
        orchestrator.resume(request.resimulation_id)


def test_comparison_failure_preserves_scoring_and_resume_does_not_rescore(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = FakeBackend(tmp_path, fail_once=Stage.COMPARISON)
    orchestrator = ResimulationOrchestrator(tmp_path, backend)
    failed = orchestrator.start(request)
    assert failed.current_stage is Stage.SCORING
    assert Stage.SCORING in [item.stage for item in failed.artifacts]
    completed = orchestrator.resume(request.resimulation_id)
    assert completed.status is RunStatus.COMPLETED
    assert backend.calls.count(Stage.SCORING) == 1


def test_duplicate_start_lock_and_crash_temp_recovery(tmp_path: Path) -> None:
    request = _request(tmp_path)
    store = ResimulationStore(tmp_path)
    backend = FakeBackend(tmp_path)
    with store.lock(request.resimulation_id), pytest.raises(RuntimeError, match="already active"):
        ResimulationOrchestrator(tmp_path, backend, store=store).start(request)
    result = ResimulationOrchestrator(tmp_path, backend, store=store).start(request)
    abandoned = store.directory(request.resimulation_id) / ".state.crash.tmp"
    abandoned.write_text("partial", encoding="utf-8")
    assert store.recover(request.resimulation_id) == result


def test_crash_before_atomic_rename_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    store = ResimulationStore(tmp_path)
    result = ResimulationOrchestrator(tmp_path, FakeBackend(tmp_path), store=store).start(request)
    changed = result.model_copy(update={"updated_at": dt.datetime.now(dt.UTC)})
    monkeypatch.setattr(
        "blackmirror.resimulation.storage.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("crash")),
    )
    with pytest.raises(OSError, match="crash"):
        store.checkpoint(changed)
    assert store.read(request.resimulation_id) == result
    assert not list(store.directory(request.resimulation_id).glob(".state.*.tmp"))


def test_external_stop_reasons_are_persisted(tmp_path: Path) -> None:
    request = _request(tmp_path, attempts=1)
    backend = FakeBackend(tmp_path, fail_once=Stage.INFERENCE)
    orchestrator = ResimulationOrchestrator(tmp_path, backend)
    orchestrator.start(request)
    stopped = orchestrator.stop(request.resimulation_id, StopReason.RESOURCE_LIMIT)
    assert stopped.status is RunStatus.STOP_REQUESTED
    assert stopped.requested_stop_reason is StopReason.RESOURCE_LIMIT
    assert orchestrator.resume(request.resimulation_id).status is RunStatus.STOPPED


def test_active_pipeline_observes_cancellation_without_taking_worker_lock(tmp_path: Path) -> None:
    request = _request(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    class BlockingBackend(FakeBackend):
        def analyze(self, run_id: str, upstream: dict[str, str]) -> StageArtifact:
            entered.set()
            assert release.wait(timeout=5)
            return super().analyze(run_id, upstream)

    orchestrator = ResimulationOrchestrator(tmp_path, BlockingBackend(tmp_path))
    output: list[object] = []
    worker = threading.Thread(target=lambda: output.append(orchestrator.start(request)))
    worker.start()
    assert entered.wait(timeout=5)
    requested = orchestrator.stop(request.resimulation_id, StopReason.CANCELLED)
    assert requested.status is RunStatus.STOP_REQUESTED
    assert requested.requested_stop_reason is StopReason.CANCELLED
    assert orchestrator.store.read(request.resimulation_id).status is RunStatus.STOP_REQUESTED
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert output[0].status is RunStatus.STOPPED  # type: ignore[union-attr]
    assert output[0].stop_reason is StopReason.CANCELLED  # type: ignore[union-attr]


def test_dead_process_lock_is_recovered(tmp_path: Path) -> None:
    request = _request(tmp_path)
    store = ResimulationStore(tmp_path)
    store.root.mkdir(parents=True)
    (store.root / f".{request.resimulation_id}.lock").write_text("99999999", encoding="utf-8")
    result = ResimulationOrchestrator(tmp_path, FakeBackend(tmp_path), store=store).start(request)
    assert result.status is RunStatus.COMPLETED


def test_outcomes_compare_raw_values_and_scores_separately_by_direction() -> None:
    objectives = tuple(
        SimpleNamespace(
            objective_id=name,
            direction=direction,
            normalization=SimpleNamespace(target_value=target),
        )
        for name, direction, target in (
            ("max", ObjectiveDirection.MAXIMIZE, None),
            ("min", ObjectiveDirection.MINIMIZE, None),
            ("target", ObjectiveDirection.TARGET, 5.0),
            ("missing", ObjectiveDirection.MAXIMIZE, None),
        )
    )

    def evaluation(name: str, raw: float | None, score: float | None, valid: bool = True):
        return SimpleNamespace(objective_id=name, raw_value=raw, score=score, valid=valid)

    parent = SimpleNamespace(
        variant_id="parent",
        objective_scores=(
            evaluation("max", 1, 0.2),
            evaluation("min", 4, 0.3),
            evaluation("target", 1, 0.4),
            evaluation("missing", None, None, False),
        ),
    )
    candidate = SimpleNamespace(
        variant_id="candidate",
        objective_scores=(
            evaluation("max", 2, 0.8),
            evaluation("min", 2, 0.7),
            evaluation("target", 4, 0.9),
            evaluation("missing", 1, 1, True),
        ),
    )
    score = SimpleNamespace(objectives=objectives, variant_scores=(parent, candidate))
    deltas = evaluate_objectives(
        score, parent_run_id="parent", candidate_run_id="candidate", tolerance=0
    )
    assert [item.outcome for item in deltas] == [
        Outcome.IMPROVED,
        Outcome.IMPROVED,
        Outcome.IMPROVED,
        Outcome.INCONCLUSIVE,
    ]
    assert deltas[0].raw_delta == 1
    assert deltas[0].score_delta == pytest.approx(0.6)


def test_multi_objective_request_requires_per_hypothesis_semantics(tmp_path: Path) -> None:
    request = _request(tmp_path)
    second = request.objectives[0].model_copy(
        update={"objective_id": "second", "direction": ObjectiveDirection.MINIMIZE}
    )
    from blackmirror.scoring.engine import objective_set_hash

    with pytest.raises(ValueError, match="hypothesis-objective mapping"):
        ResimulationRequest.model_validate(
            {
                **request.model_dump(),
                "objectives": (*request.objectives, second),
                "objective_definition_hash": objective_set_hash((*request.objectives, second)),
            }
        )


def test_objective_definition_identity_ignores_provenance_but_rejects_math_change(
    tmp_path: Path,
) -> None:
    from blackmirror.scoring.engine import objective_set_hash
    from blackmirror.scoring.schemas import ObjectiveProvenance

    objective = _request(tmp_path).objectives[0]
    annotated = objective.model_copy(
        update={"provenance": ObjectiveProvenance(created_by="another reviewer", note="same math")}
    )
    changed = objective.model_copy(update={"weight": 0.5})
    assert objective_set_hash((objective,)) == objective_set_hash((annotated,))
    assert objective_set_hash((objective,)) != objective_set_hash((changed,))


def test_hypothesis_direction_cannot_contradict_objective(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="contradicts objective direction"):
        ResimulationRequest.model_validate(
            {
                **request.model_dump(),
                "hypothesis_expected_directions": {
                    "H1": ExpectedDirection.TEST_FOR_DECREASE
                },
            }
        )
