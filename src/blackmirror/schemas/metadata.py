"""Model provenance and run reproducibility metadata.

Nothing here is fabricated: every field is either read from the loaded model /
its config, or from the running environment. Fields we cannot determine stay
``None`` rather than being guessed.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelMetadata(BaseModel):
    """Provenance of the cortical prediction model that produced a run."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    name: str = Field(description="Human-readable model name, e.g. 'TRIBE v2'.")
    backend: str = Field(description="BlackMirror backend id, e.g. 'tribe_v2' or 'mock'.")
    model_id: str = Field(description="HuggingFace repo id or local checkpoint directory.")
    source: str | None = Field(default=None, description="Upstream URL the model came from.")
    checkpoint: str | None = Field(default=None, description="Checkpoint filename.")
    checkpoint_sha256: str | None = Field(
        default=None,
        description="Hash of the checkpoint file when cheaply obtainable; None otherwise.",
    )
    version: str | None = Field(default=None, description="Package version or commit.")
    license: str | None = Field(default=None, description="Model license, e.g. 'CC-BY-NC-4.0'.")

    modalities: tuple[str, ...] = Field(
        default=(),
        description="Input modalities the model consumes.",
    )
    feature_extractors: dict[str, str] = Field(
        default_factory=dict,
        description="Frozen upstream encoders by modality, read from the model config.",
    )

    loaded_device: str
    dtype: str
    parameter_count: int | None = Field(
        default=None,
        description="Trainable head parameters where countable; excludes frozen extractors.",
    )

    subject_conditioning: str | None = Field(
        default=None,
        description=(
            "How the model handles inter-subject variability. TRIBE v2's from_pretrained "
            "forces average_subjects=True, so predictions describe an AVERAGE subject and "
            "not any individual viewer. See docs/scientific_limitations.md."
        ),
    )

    is_synthetic: bool = Field(
        default=False,
        description=(
            "True only for the mock backend. Synthetic runs are labelled at every layer "
            "so they can never be mistaken for model predictions."
        ),
    )

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific metadata read from the model config.",
    )

    def fingerprint(self) -> str:
        """Identity of the model for cache-key and comparability purposes.

        The frozen feature extractors are part of the model's identity: swapping
        the text encoder changes the features the head sees, so two such runs are
        not comparable and must not share a cache entry.
        """
        extractors = ",".join(f"{k}={v}" for k, v in sorted(self.feature_extractors.items()))
        return "|".join(
            [
                self.backend,
                self.model_id,
                self.checkpoint or "-",
                self.version or "-",
                self.dtype,
                extractors or "-",
            ]
        )


class RunProvenance(BaseModel):
    """Everything needed to answer 'how exactly was this run produced?'.

    This is what lets us claim two variants were evaluated under identical
    conditions — Phase 5 depends on being able to diff two of these.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    run_id: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    blackmirror_version: str
    python_version: str
    platform: str
    torch_version: str | None = None
    numpy_version: str | None = None
    backend_package_version: str | None = Field(
        default=None, description="Installed version of the backend package (e.g. tribev2)."
    )

    device: str
    dtype: str
    random_seed: int

    stimulus_sha256: str
    model_fingerprint: str
    preprocessing_config: dict[str, Any]
    cache_key: str

    applied_compat_patches: tuple[str, ...] = Field(
        default=(),
        description=(
            "Platform compatibility patches applied to upstream dependencies for this run. "
            "A non-empty list means the run did NOT use a stock upstream environment; it is "
            "recorded so a patched run is never mistaken for a clean one."
        ),
    )
