"""Contracts for turning a described edit into an actual media file.

WHY THIS EXISTS
    Phase 7 produces a description of a variant. Phase 8 measures a variant that
    already exists. Nothing joined the two: Phase 8's only adapter binds a file a
    human built, deliberately, because a system that silently generates media is
    a system that can silently generate media nobody reviewed.

    Automated search cannot evaluate what it cannot build, so Phase 9 needs that
    join. This module is the narrowest possible version of it: a fixed set of
    parametric, deterministic edits executed by ffmpeg. It cannot write speech,
    render text, or invent footage. Every output is a mechanical transform of the
    source with the exact parameters recorded.

WHY THE EDIT SET IS SMALL
    Phase 7's `EditInstruction` has seven operations, but only some are
    mechanically realizable. `insert_speech` and `replace_text` need generative
    models, which are explicitly out of scope. Implementing the realizable subset
    honestly is better than implementing all seven badly, because a candidate
    whose edit did not actually happen would be scored anyway and the number
    would look real.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

MATERIALIZATION_VERSION = "1.0"


class MaterializationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class EditOperation(StrEnum):
    """Parametric edits that ffmpeg can apply deterministically.

    Each maps to one filter with one numeric parameter, which is what makes a
    search space over them well defined: every value in range produces a valid
    file, and neighbouring values produce neighbouring content.
    """

    AUDIO_GAIN_DB = "audio_gain_db"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    SPEED = "speed"


#: Inclusive value bounds and the value that changes nothing.
#:
#: The neutral value matters as much as the bounds: a plan made only of neutral
#: steps would produce a file identical to its source, which Phase 8 rejects and
#: which would waste a full inference to learn nothing.
OPERATION_BOUNDS: dict[EditOperation, tuple[float, float, float]] = {
    #: (minimum, maximum, neutral)
    EditOperation.AUDIO_GAIN_DB: (-30.0, 30.0, 0.0),
    EditOperation.BRIGHTNESS: (-1.0, 1.0, 0.0),
    EditOperation.CONTRAST: (0.0, 4.0, 1.0),
    EditOperation.SATURATION: (0.0, 3.0, 1.0),
    EditOperation.SPEED: (0.5, 2.0, 1.0),
}

#: Which stream an operation touches. Used to decide whether a re-encode is
#: needed, and to check a plan against Phase 7's `frozen_modalities`.
OPERATION_MODALITY: dict[EditOperation, str] = {
    EditOperation.AUDIO_GAIN_DB: "audio",
    EditOperation.BRIGHTNESS: "visual",
    EditOperation.CONTRAST: "visual",
    EditOperation.SATURATION: "visual",
    EditOperation.SPEED: "visual",
}

#: Operations that change how long the output is.
DURATION_CHANGING = frozenset({EditOperation.SPEED})


class EditStep(MaterializationModel):
    """One parametric edit."""

    operation: EditOperation
    value: float

    @model_validator(mode="after")
    def _within_bounds(self) -> EditStep:
        low, high, _ = OPERATION_BOUNDS[self.operation]
        if not low <= self.value <= high:
            raise ValueError(
                f"{self.operation.value} must be within [{low}, {high}], got {self.value}"
            )
        return self

    @property
    def is_neutral(self) -> bool:
        return self.value == OPERATION_BOUNDS[self.operation][2]

    @property
    def modality(self) -> str:
        return OPERATION_MODALITY[self.operation]


class MaterializationPlan(MaterializationModel):
    """A complete, hashable description of one variant's construction.

    The plan hash is what names the output file. That is not cosmetic: the
    feature-extraction cache underneath TRIBE is keyed on **file path and time
    range, not content** (`item_uid` in `neuralset.extractors`). Writing
    different content to a reused path would return another variant's cached
    features and score the wrong thing, silently. Naming the file after the plan
    makes the two agree: identical plans share a path and legitimately share
    cached features, and different plans can never collide.
    """

    source_path: Path
    steps: tuple[EditStep, ...] = Field(min_length=1)
    materialization_version: str = MATERIALIZATION_VERSION

    @model_validator(mode="after")
    def _is_a_real_edit(self) -> MaterializationPlan:
        seen = [step.operation for step in self.steps]
        if len(set(seen)) != len(seen):
            raise ValueError("each operation may appear at most once in a plan")
        if all(step.is_neutral for step in self.steps):
            raise ValueError(
                "every step is neutral, so this plan would reproduce the source "
                "byte for byte; that is not a variant and cannot be evaluated"
            )
        return self

    @property
    def effective_steps(self) -> tuple[EditStep, ...]:
        """Steps that actually change something, in stable operation order."""
        return tuple(
            sorted(
                (step for step in self.steps if not step.is_neutral),
                key=lambda step: step.operation.value,
            )
        )

    @property
    def modalities(self) -> frozenset[str]:
        return frozenset(step.modality for step in self.effective_steps)

    @property
    def changes_duration(self) -> bool:
        return any(step.operation in DURATION_CHANGING for step in self.effective_steps)

    def fingerprint(self) -> str:
        """Stable identity of the edit, independent of where the source lives.

        The source's *content* hash is folded in by the caller rather than its
        path, so moving the corpus does not invalidate every candidate.
        """
        payload = {
            "version": self.materialization_version,
            "steps": [
                {"operation": step.operation.value, "value": step.value}
                for step in self.effective_steps
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def describe(self) -> str:
        return ", ".join(
            f"{step.operation.value}={step.value:g}" for step in self.effective_steps
        )


class MaterializationResult(MaterializationModel):
    """What was actually produced, with everything needed to reproduce it."""

    plan: MaterializationPlan
    variant_path: Path
    variant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: sha256 of the plan and the source content together; names the file.
    candidate_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool: str
    tool_version: str
    command: tuple[str, ...]
    source_duration_seconds: float = Field(gt=0)
    variant_duration_seconds: float = Field(gt=0)
    reused_existing: bool = False

    @property
    def duration_change_seconds(self) -> float:
        return self.variant_duration_seconds - self.source_duration_seconds

    @model_validator(mode="after")
    def _content_actually_changed(self) -> MaterializationResult:
        if self.variant_sha256 == self.source_sha256:
            raise ValueError(
                "the produced file is byte-identical to its source; the edit had "
                "no effect and evaluating it would measure nothing"
            )
        return self


__all__ = [
    "DURATION_CHANGING",
    "MATERIALIZATION_VERSION",
    "OPERATION_BOUNDS",
    "OPERATION_MODALITY",
    "EditOperation",
    "EditStep",
    "MaterializationPlan",
    "MaterializationResult",
]
