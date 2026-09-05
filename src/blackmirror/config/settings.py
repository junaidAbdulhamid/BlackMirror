"""Centralized configuration.

All tunable constants live here. Nothing elsewhere in the codebase reads
``os.environ`` directly or hard-codes a path.

Values are read from (in precedence order): explicit constructor arguments,
environment variables prefixed ``BLACKMIRROR_``, a local ``.env`` file, then the
defaults below. See ``.env.example``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: src/blackmirror/config/settings.py -> up 4 levels.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


class DevicePreference(StrEnum):
    """Requested compute device.

    ``AUTO`` resolves to the best device the *selected backend* actually
    supports, which is not necessarily the best device on the machine. TRIBE v2
    has no MPS code path (see docs/tribe_integration.md), so AUTO resolves to
    CPU on Apple Silicon rather than silently producing a broken MPS run.
    """

    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class BackendName(StrEnum):
    """Which cortical prediction backend to use."""

    TRIBE_V2 = "tribe_v2"
    MOCK = "mock"


class Settings(BaseSettings):
    """Runtime configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_prefix="BLACKMIRROR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_id` would otherwise collide with pydantic's protected namespace.
        protected_namespaces=(),
    )

    # --- Backend / model -------------------------------------------------

    backend: BackendName = Field(
        default=BackendName.TRIBE_V2,
        description="Cortical prediction backend. 'mock' produces clearly-labelled synthetic data.",
    )
    model_id: str = Field(
        default="facebook/tribev2",
        description="HuggingFace repo id or local checkpoint directory.",
    )
    checkpoint_name: str = Field(
        default="best.ckpt",
        description="Checkpoint filename inside the model repo/directory.",
    )
    text_encoder_id: str | None = Field(
        default=None,
        description=(
            "Override the frozen TEXT encoder repo. None keeps the checkpoint's own "
            "value (meta-llama/Llama-3.2-3B), which is GATED on HuggingFace and needs "
            "an accepted license plus HF_TOKEN. Set this to a repo you can access — "
            "your own mirror, a local path, or an ungated mirror of the same weights — "
            "when you cannot use the gated repo directly. It must be the SAME model: "
            "the trained head expects Llama-3.2-3B's hidden dimensions, and a different "
            "architecture will either fail to load or produce meaningless predictions. "
            "Any override is recorded in the model fingerprint, so runs using different "
            "text encoders are correctly treated as non-comparable."
        ),
    )
    model_cache_dir: Path = Field(
        default=_PACKAGE_ROOT / "cache",
        description="Feature-extraction cache used by TRIBE (exca TaskInfra). Grows large.",
    )

    # --- Compute ---------------------------------------------------------

    device: DevicePreference = Field(
        default=DevicePreference.AUTO,
        description="Requested compute device.",
    )
    allow_cpu_fallback: bool = Field(
        default=True,
        description=(
            "If False, an explicit CUDA/MPS request that cannot be satisfied raises "
            "DeviceUnavailableError instead of quietly running on CPU. CPU inference for "
            "TRIBE v2 is 100-1000x slower, so a silent fallback can look like a hang."
        ),
    )
    dtype: Literal["float32", "bfloat16", "float16"] = Field(
        default="float32",
        description="Inference dtype. TRIBE v2 was trained and released in float32.",
    )

    # --- Preprocessing ---------------------------------------------------

    remove_empty_segments: bool = Field(
        default=True,
        description=(
            "Mirrors TribeModel.remove_empty_segments. When True (upstream default) the "
            "model DROPS per-TR segments containing no events, so the prediction time axis "
            "becomes non-uniform. BlackMirror always persists per-row segment offsets so "
            "alignment survives either way. Set False for a contiguous timeline."
        ),
    )

    enable_transcription: bool = Field(
        default=True,
        description=(
            "Run TRIBE's WhisperX word-level transcription to build the text stream. "
            "Disabling it yields a REDUCED-MODALITY run (audio/video only): TRIBE is "
            "trimodal, so such predictions are not comparable with full-modality runs. "
            "Provided as an escape hatch where WhisperX cannot run."
        ),
    )

    # --- Storage ---------------------------------------------------------

    artifact_dir: Path = Field(
        default=_PACKAGE_ROOT / "artifacts",
        description="Root directory for run artifacts, exported meshes and benchmarks.",
    )
    reuse_cached_runs: bool = Field(
        default=False,
        description=(
            "If True, a completed run with an identical cache key "
            "(stimulus hash + model fingerprint + preprocessing config) is returned "
            "instead of recomputing."
        ),
    )

    # --- Observability ---------------------------------------------------

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    random_seed: int = Field(
        default=1234,
        description="Seeded for reproducibility. TRIBE v2 inference is deterministic, but "
        "the mock backend and any future stochastic step depend on this.",
    )

    # --- Platform compatibility -----------------------------------------

    enable_macos_compat: bool = Field(
        default=True,
        description=(
            "Apply the documented macOS/Apple-Silicon compatibility patches to TRIBE's "
            "dependencies (notably the WhisperX float16 compute type, unsupported by "
            "CTranslate2 on Apple Silicon CPU). Every applied patch is recorded in the "
            "run provenance. See blackmirror/inference/compat_macos.py."
        ),
    )

    @field_validator("model_cache_dir", "artifact_dir", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    # --- Derived paths ---------------------------------------------------

    @property
    def runs_dir(self) -> Path:
        return self.artifact_dir / "runs"

    @property
    def mesh_dir(self) -> Path:
        return self.artifact_dir / "mesh"

    @property
    def benchmarks_dir(self) -> Path:
        return self.artifact_dir / "benchmarks"

    @property
    def index_path(self) -> Path:
        return self.artifact_dir / "index.jsonl"

    def preprocessing_fingerprint(self) -> dict[str, object]:
        """The preprocessing configuration that participates in the run cache key.

        Only fields that change the *numbers* belong here. Log level and artifact
        directory do not.
        """
        return {
            "remove_empty_segments": self.remove_empty_segments,
            "enable_transcription": self.enable_transcription,
            "dtype": self.dtype,
            "random_seed": self.random_seed,
        }


def get_settings(**overrides: object) -> Settings:
    """Build a :class:`Settings` instance, applying explicit overrides last."""
    return Settings(**overrides)  # type: ignore[arg-type]
