"""Strict, versioned contracts for the Phase 8 re-simulation loop."""

from __future__ import annotations

import datetime as dt
from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from blackmirror.optimization.schemas import ExpectedDirection, ProposedVariantSpec
from blackmirror.scoring.schemas import NeuralObjective

RESIMULATION_VERSION = "1.0"


class ResimulationConflict(ValueError):
    """An existing run contradicts the request being made of it.

    Distinct from a validation error because the request may be perfectly well
    formed: the conflict is with what is already on disk under that id. It
    subclasses ValueError so every existing caller that guards against bad
    input still catches it.
    """


class ResimulationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Stage(IntEnum):
    BOUND = 0
    INFERENCE = 1
    ANALYTICS = 2
    CONTENT = 3
    SCORING = 4
    COMPARISON = 5
    EVALUATED = 6


class RunStatus(StrEnum):
    ACTIVE = "active"
    STOP_REQUESTED = "stop_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class StopReason(StrEnum):
    COMPLETED = "completed"
    FAILED_STAGE = "failed_stage"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    MAX_ITERATIONS = "max_iterations"
    MAX_ATTEMPTS = "max_attempts"
    NO_MEASURABLE_OBJECTIVES = "no_measurable_objectives"


class Outcome(StrEnum):
    IMPROVED = "improved"
    WORSENED = "worsened"
    UNCHANGED = "unchanged"
    INCONCLUSIVE = "inconclusive"


class HypothesisVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class TransformationBinding(ResimulationModel):
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    method: str = Field(min_length=1)
    tool: str | None = None
    model: str | None = None
    parameters: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    parameters_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant_path: str
    variant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dry_run_identical_allowed: bool = False

    @model_validator(mode="after")
    def distinct_content(self) -> TransformationBinding:
        if self.source_sha256 == self.variant_sha256 and not self.dry_run_identical_allowed:
            raise ValueError("source and variant content hashes must differ")
        if self.dry_run_identical_allowed and self.adapter_id != "dry-run-test":
            raise ValueError("identical content is restricted to the dry-run-test adapter")
        return self


class ResimulationRequest(ResimulationModel):
    resimulation_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    optimization_request_key: str = Field(min_length=1, max_length=128)
    parent_run_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    proposed_variant: ProposedVariantSpec
    hypothesis_objective_ids: dict[str, str] = Field(default_factory=dict)
    hypothesis_expected_directions: dict[str, ExpectedDirection] = Field(default_factory=dict)
    binding: TransformationBinding
    objectives: tuple[NeuralObjective, ...] = Field(min_length=1)
    objective_set_hash: str = Field(pattern=r"^[0-9a-f]{16,64}$")
    objective_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_resimulation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    iteration_index: int = Field(default=1, ge=1)
    max_iterations: int = Field(default=1, ge=1, le=100)
    max_attempts: int = Field(default=3, ge=1, le=20)
    outcome_tolerance: float = Field(default=1e-9, ge=0)

    @model_validator(mode="after")
    def lineage_matches(self) -> ResimulationRequest:
        if self.proposed_variant.experiment_id != self.experiment_id:
            raise ValueError("proposed variant experiment differs from request")
        if self.proposed_variant.parent_variant_id != self.parent_run_id:
            raise ValueError("proposed variant parent differs from parent run")
        if self.proposed_variant.target_objective_id not in {
            item.objective_id for item in self.objectives
        }:
            raise ValueError("target objective is absent from the declared objective set")
        objective_ids = {item.objective_id for item in self.objectives}
        from blackmirror.scoring.engine import objective_set_hash

        if objective_set_hash(self.objectives) != self.objective_definition_hash:
            raise ValueError("objective definition hash differs from canonical objectives")
        if self.hypothesis_objective_ids:
            if set(self.hypothesis_objective_ids) != set(self.proposed_variant.hypothesis_ids):
                raise ValueError("hypothesis-objective mapping must cover every hypothesis exactly")
            if not set(self.hypothesis_objective_ids.values()) <= objective_ids:
                raise ValueError("hypothesis mapping cites an undeclared objective")
        elif len(objective_ids) > 1:
            raise ValueError("multi-objective re-simulation requires hypothesis-objective mapping")
        if self.hypothesis_expected_directions and set(self.hypothesis_expected_directions) != set(
            self.proposed_variant.hypothesis_ids
        ):
            raise ValueError("hypothesis direction mapping must cover every hypothesis exactly")
        if len(objective_ids) > 1 and not self.hypothesis_expected_directions:
            raise ValueError("multi-objective re-simulation requires hypothesis direction mapping")
        objective_directions = {item.objective_id: item.direction.value for item in self.objectives}
        expected_for_direction = {
            "maximize": ExpectedDirection.TEST_FOR_INCREASE,
            "minimize": ExpectedDirection.TEST_FOR_DECREASE,
            "target": ExpectedDirection.TEST_FOR_TARGET_PROXIMITY,
        }
        for hypothesis_id in self.proposed_variant.hypothesis_ids:
            objective_id = self.hypothesis_objective_ids.get(
                hypothesis_id, self.proposed_variant.target_objective_id
            )
            expected = self.hypothesis_expected_directions.get(
                hypothesis_id, self.proposed_variant.expected_direction
            )
            if expected is not expected_for_direction[objective_directions[objective_id]]:
                raise ValueError(
                    f"hypothesis {hypothesis_id} expected direction contradicts objective direction"
                )
        if self.iteration_index > self.max_iterations:
            raise ValueError("iteration index exceeds the bounded re-simulation loop")
        if self.iteration_index > 1 and self.previous_resimulation_id is None:
            raise ValueError("later iterations must cite the previous resimulation")
        return self


class StageArtifact(ResimulationModel):
    stage: Stage
    artifact_id: str
    artifact_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_files: dict[str, str] = Field(min_length=1)
    upstream_sha256: dict[str, str]
    pipeline_version: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @model_validator(mode="after")
    def file_hashes_are_valid(self) -> StageArtifact:
        if self.artifact_path not in self.artifact_files:
            raise ValueError("primary artifact path must be part of the declared artifact set")
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in self.artifact_files.values()
        ):
            raise ValueError("every declared artifact file needs a SHA-256")
        return self


class FailedAttempt(ResimulationModel):
    attempt: int = Field(ge=1)
    stage: Stage
    error_type: str
    message: str
    occurred_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class ObjectiveDelta(ResimulationModel):
    objective_id: str
    direction: str
    parent_raw_value: float | None
    candidate_raw_value: float | None
    raw_delta: float | None
    parent_score: float | None
    candidate_score: float | None
    score_delta: float | None
    outcome: Outcome


class HypothesisOutcome(ResimulationModel):
    hypothesis_id: str
    expected_direction: ExpectedDirection
    verdict: HypothesisVerdict
    measured_outcome: Outcome
    objective_id: str
    note: str


class ResimulationResult(ResimulationModel):
    schema_version: str = "1.0"
    resimulation_version: str = RESIMULATION_VERSION
    request: ResimulationRequest
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RunStatus = RunStatus.ACTIVE
    current_stage: Stage = Stage.BOUND
    attempt_count: int = Field(default=0, ge=0)
    candidate_run_id: str | None = None
    source_scoring_cache_key: str | None = None
    produced_scoring_cache_key: str | None = None
    produced_objective_definition_hash: str | None = None
    produced_scoring_input_sha256: dict[str, str] = Field(default_factory=dict)
    artifacts: tuple[StageArtifact, ...] = ()
    failures: tuple[FailedAttempt, ...] = ()
    objective_deltas: tuple[ObjectiveDelta, ...] = ()
    hypothesis_outcomes: tuple[HypothesisOutcome, ...] = ()
    stop_reason: StopReason | None = None
    requested_stop_reason: StopReason | None = None
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    interpretation_notice: str = (
        "Outcomes compare explicitly declared mathematical objectives on model-predicted "
        "responses. They are not causal effects or evidence of real-human improvement."
    )

    @model_validator(mode="after")
    def monotonic_artifacts(self) -> ResimulationResult:
        stages = [item.stage for item in self.artifacts]
        if stages != sorted(set(stages)):
            raise ValueError("stage artifacts must be unique and monotonic")
        if stages and stages[-1] > self.current_stage:
            raise ValueError("artifact is ahead of current state")
        if self.status is RunStatus.COMPLETED and self.stop_reason is not StopReason.COMPLETED:
            raise ValueError("completed state requires completed stop reason")
        return self
