"""Steps 1-2, 19, 54-55: what a surrogate is trained on and what it predicts.

WHAT A SURROGATE IS FOR
    The real objective is expensive: one evaluation is tens of minutes of model
    inference. A surrogate is a cheap approximation learned from candidates
    already evaluated, used to *rank* many untested candidates so that the
    expensive evaluations go to the promising ones.

    It is a screening model. It never replaces the real evaluator, and a
    predicted score is never an outcome. Every type here keeps the two
    separable by name and by field.

WHY THE DATASET REFUSES SURROGATE LABELS
    A model trained on its own predictions learns its own errors and reports
    them back with growing confidence. Only records carrying a real evaluated
    objective are admitted, and the type system makes that the only way to
    build one.

WHY IDENTITY IS SO WIDE
    A label is only comparable to another label if everything that produced it
    matches: the root media, the objective definition, the search space, and the
    versions of the materializer, the scorer and the model. Phase 9's
    `CacheIdentity` already composes exactly that set, so this reuses the idea
    rather than inventing a parallel scheme. Two records from different
    identities must never land in one training set.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

SURROGATE_VERSION = "1.0"
FEATURE_SCHEMA_VERSION = "1.0"


class SurrogateModelType(StrEnum):
    """Step 12. Only families that are implemented and tested appear here."""

    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    GAUSSIAN_PROCESS = "gaussian_process"


class SurrogateState(StrEnum):
    """Step 53. Where the surrogate is in its lifecycle."""

    UNINITIALIZED = "uninitialized"
    BOOTSTRAPPING = "bootstrapping"
    TRAINING = "training"
    READY = "ready"
    #: Trained, but its validation quality is below the configured threshold.
    DEGRADED = "degraded"
    DISABLED = "disabled"


class SurrogateModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DatasetIdentity(SurrogateModel):
    """Everything that must match for two labels to be comparable.

    A surrogate trained across two identities would be learning a mixture of
    two functions and reporting one number, with nothing in the output saying
    so. Steps 69 and 70 ask for exactly this guard.
    """

    root_media_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_space_version: str
    materialization_version: str
    scoring_version: str
    model_fingerprint: str
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def is_compatible_with(self, other: DatasetIdentity) -> bool:
        return self.fingerprint() == other.fingerprint()

    def differences(self, other: DatasetIdentity) -> tuple[str, ...]:
        """Which fields differ, so a mismatch can be explained rather than just refused."""
        mine = self.model_dump(mode="json")
        theirs = other.model_dump(mode="json")
        return tuple(
            f"{key}: {mine[key]!r} vs {theirs[key]!r}"
            for key in sorted(mine)
            if mine[key] != theirs[key]
        )


class SurrogateTrainingRecord(SurrogateModel):
    """Step 2. One candidate with a real, trusted objective value."""

    candidate_id: str = Field(min_length=1)
    genome: dict[str, float] = Field(min_length=1)
    #: Objective id -> the real measured value, direction untouched.
    objective_values: dict[str, float] = Field(min_length=1)
    #: The primary objective's value with direction folded so higher is better.
    directional_target: float
    primary_objective_id: str = Field(min_length=1)
    feasible: bool = True
    #: Order this observation arrived in, for order-aware validation (Step 14).
    sequence: int = Field(default=0, ge=0)
    evaluation_wall_seconds: float = Field(default=0.0, ge=0)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @model_validator(mode="after")
    def _target_is_finite(self) -> SurrogateTrainingRecord:
        if not np.isfinite(self.directional_target):
            raise ValueError("a training target must be a finite number")
        return self


class DataQualityIssue(SurrogateModel):
    """Step 68. Why a candidate was kept out of the training set."""

    candidate_id: str
    reason: str


class SurrogateDataset(SurrogateModel):
    """Step 1. Real observations only, under one identity.

    Built through `from_fitness`, which is the gate: a `CandidateFitness` that
    did not produce a real measured value never becomes a row, and the reason it
    was excluded is recorded rather than dropped.
    """

    identity: DatasetIdentity
    records: tuple[SurrogateTrainingRecord, ...] = ()
    excluded: tuple[DataQualityIssue, ...] = ()
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Parameter names, sorted, taken from the first record."""
        return () if not self.records else tuple(sorted(self.records[0].genome))

    def targets(self) -> np.ndarray:
        return np.asarray([item.directional_target for item in self.records], dtype=float)

    def target_spread(self) -> float:
        """Range of the labels. Compared against the nuisance floor by callers."""
        values = self.targets()
        return float(values.max() - values.min()) if values.size > 1 else 0.0

    def dataset_hash(self) -> str:
        """Step 83. Order-independent identity of the exact rows and labels.

        Sorting by candidate id before hashing means the same observations in a
        different order produce the same hash, while changing any single label
        changes it.
        """
        rows = sorted(
            (
                {
                    "candidate_id": item.candidate_id,
                    "genome": {k: item.genome[k] for k in sorted(item.genome)},
                    "target": item.directional_target,
                }
                for item in self.records
            ),
            key=lambda row: row["candidate_id"],  # type: ignore[arg-type,return-value]
        )
        payload = json.dumps(
            {"identity": self.identity.fingerprint(), "rows": rows},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def with_record(self, record: SurrogateTrainingRecord) -> SurrogateDataset:
        """Append one observation, replacing any earlier row for that candidate."""
        kept = tuple(
            item for item in self.records if item.candidate_id != record.candidate_id
        )
        return self.model_copy(update={"records": (*kept, record)})

    def consistent_genomes(self) -> bool:
        """Whether every record carries the same parameter names."""
        if not self.records:
            return True
        expected = set(self.records[0].genome)
        return all(set(item.genome) == expected for item in self.records)


class SurrogatePrediction(SurrogateModel):
    """Step 19. A cheap estimate, labelled as one everywhere it appears."""

    candidate_id: str
    genome: dict[str, float]
    #: Predicted directional objective. Never an outcome, never a measurement.
    predicted_score: float
    #: Predictive standard deviation. `None` when the model cannot supply one.
    uncertainty: float | None = None
    #: Distance from the training data. Higher means less trustworthy.
    ood_score: float | None = None
    model_type: SurrogateModelType
    model_version: int = Field(ge=1)
    training_dataset_size: int = Field(ge=0)
    training_dataset_hash: str
    #: False when the prediction should not be acted on, with a reason.
    valid: bool = True
    reason: str = ""

    @model_validator(mode="after")
    def _invalid_needs_a_reason(self) -> SurrogatePrediction:
        if not self.valid and not self.reason:
            raise ValueError("an invalid prediction must say why")
        return self

    @property
    def upper(self) -> float | None:
        return None if self.uncertainty is None else self.predicted_score + self.uncertainty

    @property
    def lower(self) -> float | None:
        return None if self.uncertainty is None else self.predicted_score - self.uncertainty

    def describe(self) -> str:
        """Wording that cannot be mistaken for a measurement."""
        band = "" if self.uncertainty is None else f" ± {self.uncertainty:.4f}"
        return (
            f"surrogate estimate {self.predicted_score:.4f}{band} "
            f"(not a measured objective; {self.model_type.value} trained on "
            f"{self.training_dataset_size} real evaluations)"
        )


class SurrogateModelMetadata(SurrogateModel):
    """Step 54. Enough to reproduce a model and to know when not to trust it."""

    model_type: SurrogateModelType
    model_version: int = Field(ge=1)
    training_dataset_hash: str
    training_dataset_size: int = Field(ge=0)
    dataset_identity_fingerprint: str
    feature_names: tuple[str, ...]
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    hyperparameters: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    training_seed: int = 0
    surrogate_version: str = SURROGATE_VERSION
    trained_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    supports_uncertainty: bool = False
    #: How uncertainty was produced, because a forest's spread and a GP's
    #: posterior are different objects and must not be read the same way.
    uncertainty_kind: str = "none"


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "SURROGATE_VERSION",
    "DataQualityIssue",
    "DatasetIdentity",
    "SurrogateDataset",
    "SurrogateModelMetadata",
    "SurrogateModelType",
    "SurrogatePrediction",
    "SurrogateState",
    "SurrogateTrainingRecord",
]
