"""The prediction contract.

``PredictionResult`` is the single object every later phase consumes. It is
serialized verbatim to ``manifest.json`` in each run directory, so a Phase 2
visualizer, a Phase 5 A/B engine or a future FastAPI response can all be built
against it without importing TRIBE, torch, or anything else heavy.

Arrays are never embedded in this schema. Numeric data lives in ``.npy``/``.npz``
files; this schema describes and points at them.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.schemas.metadata import ModelMetadata, RunProvenance
from blackmirror.schemas.stimulus import StimulusInput


class PredictionArrayMetadata(BaseModel):
    """Describes the raw prediction matrix stored on disk.

    The array is written once and never modified. Any transformation Phase 2+
    wants (normalization, smoothing, ROI averaging) must produce a *separate*
    array and record its parameters — see docs/scientific_limitations.md.
    """

    model_config = ConfigDict(frozen=True)

    shape: tuple[int, ...] = Field(description="Array shape as stored, e.g. (T, V).")
    dtype: str
    axis_names: tuple[str, ...] = Field(
        description="Meaning of each axis, aligned with `shape`, e.g. ('time', 'vertex')."
    )
    temporal_samples: int = Field(ge=0, description="Length of the time axis (T).")
    cortical_features: int = Field(ge=0, description="Length of the vertex axis (V).")

    units: str | None = Field(
        default=None,
        description=(
            "Physical units, if the model documents any. TRIBE v2 predicts normalized BOLD "
            "response; upstream does not publish a physical unit, so this stays None rather "
            "than asserting one."
        ),
    )
    normalization: str | None = Field(
        default=None,
        description="Normalization applied by the MODEL (not by us), where documented.",
    )
    semantics: str = Field(
        description="Plain-language statement of what a single value means."
    )
    is_raw_model_output: bool = Field(
        default=True,
        description="True when the array is byte-for-byte what the model returned.",
    )
    artifact_path: Path = Field(description="Path to the .npy file, relative to the run dir.")


class TemporalMetadata(BaseModel):
    """How prediction rows map onto stimulus time.

    This is the piece Phase 2 cannot work without. TRIBE v2's ``predict`` drops
    per-TR segments that contain no events (``remove_empty_segments=True``), so
    **row index i is not necessarily time i * TR**. The per-row offsets stored in
    ``temporal.npz`` are the only reliable alignment.
    """

    model_config = ConfigDict(frozen=True)

    n_time_points: int = Field(ge=0)
    tr_seconds: float = Field(
        gt=0.0,
        description="Repetition time: the duration each prediction row covers. TRIBE: 1/frequency.",
    )
    sampling_rate_hz: float = Field(gt=0.0, description="1 / tr_seconds.")

    timeline_is_contiguous: bool = Field(
        description=(
            "True when rows tile the stimulus with no gaps at exactly TR spacing. "
            "False means rows were dropped and playback sync MUST use segment_start_seconds."
        )
    )
    segments_were_filtered: bool = Field(
        description="Whether the backend dropped event-free segments."
    )
    n_segments_total: int | None = Field(
        default=None,
        description=(
            "Segments the backend ENUMERATED before filtering, when it reports it. "
            "This is NOT the number of distinct time points in the stimulus and the "
            "kept/total ratio is NOT the fraction of the timeline retained: TRIBE "
            "counts TR sub-segments across every batch its loader yields, and the "
            "same stretch of stimulus can be enumerated more than once. Measured on a "
            "real 10 s run it reported 10/100 while the 10 kept rows tiled the whole "
            "0-10 s gaplessly. Use timeline_is_contiguous and covered_seconds to judge "
            "coverage; those are derived from the returned segments themselves."
        ),
    )
    n_segments_kept: int | None = Field(default=None)

    first_segment_start_seconds: float | None = None
    last_segment_end_seconds: float | None = None
    covered_seconds: float | None = Field(
        default=None, description="Total stimulus time actually covered by prediction rows."
    )

    hemodynamic_offset_seconds: float = Field(
        default=0.0,
        description=(
            "The hemodynamic lag the MODEL was trained to compensate, in seconds. For "
            "TRIBE v2 this is `data.neuro.offset` (5.0), read from the checkpoint."
        ),
    )
    hemodynamic_offset_verified: bool = Field(
        default=False,
        description=(
            "Whether the offset's MEANING AND SIGN were established from the model's own "
            "code/config rather than prose documentation. Explicit because getting the sign "
            "backwards would misattribute every response to the wrong content event."
        ),
    )
    hemodynamic_offset_source: str | None = Field(
        default=None,
        description="Where the offset and its sign convention were established.",
    )
    output_is_stimulus_aligned: bool = Field(
        default=True,
        description=(
            "True when a prediction row's segment start IS the stimulus content time, "
            "because the model already performed the lag alignment during training. For "
            "TRIBE v2 this is True: training shifts the fMRI recording to "
            "`start - offset`, pairing stimulus at time t with BOLD measured at t+offset. "
            "Phase 2 should therefore sync playback on segment_start_seconds directly."
        ),
    )
    hemodynamic_offset_applied_to_raw: Literal[False] = Field(
        default=False,
        description=(
            "Always False: BlackMirror never shifts the raw prediction matrix. Any time "
            "alignment is expressed as separate derived arrays in temporal.npz."
        ),
    )

    arrays_artifact_path: Path | None = Field(
        default=None, description="Path to temporal.npz, relative to the run dir."
    )
    array_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Key -> description for each array inside temporal.npz.",
    )

    event_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Counts and types of the stimulus events the backend derived.",
    )

    notes: tuple[str, ...] = Field(
        default=(),
        description=(
            "Caveats about this run's temporal alignment that a consumer must see, "
            "e.g. derived stimulus times falling outside the stimulus."
        ),
    )


class CorticalMetadata(BaseModel):
    """How a prediction vector maps onto the cortical surface.

    Establishing this is the main Phase 2 prerequisite: given a row of the
    prediction matrix, which vertex of which hemisphere does each column colour?
    """

    model_config = ConfigDict(frozen=True)

    surface_space: str = Field(description="Template surface, e.g. 'fsaverage5'.")
    vertex_count: int = Field(ge=0, description="Total columns in the prediction matrix.")
    vertices_per_hemisphere: int = Field(ge=0)
    hemisphere_order: tuple[str, ...] = Field(
        description="Hemisphere order along the vertex axis, e.g. ('left', 'right')."
    )
    hemisphere_index_ranges: dict[str, tuple[int, int]] = Field(
        description="Half-open [start, end) column ranges per hemisphere."
    )
    is_surface_based: bool = Field(
        default=True, description="False would indicate a volumetric (voxel) model."
    )
    includes_subcortex: bool = Field(default=False)

    mesh_artifact_dir: Path | None = Field(
        default=None,
        description="Directory holding exported vertices/faces, if the mesh has been exported.",
    )
    medial_wall_handling: str | None = Field(
        default=None,
        description=(
            "How the medial wall is treated. The model emits values for every vertex "
            "including the medial wall, where fMRI signal is not meaningful."
        ),
    )
    atlas: str | None = Field(default=None, description="Parcellation atlas, if any.")
    mapping_notes: str | None = None


class PredictionValidation(BaseModel):
    """Structural and numerical health of a prediction matrix.

    Statistics are computed over the raw array. Anomalies are reported, never
    repaired: a NaN in scientific output is information, not noise to hide.
    """

    model_config = ConfigDict(frozen=True)

    valid: bool = Field(description="True when no structural check failed.")
    shape: tuple[int, ...]
    dtype: str

    nan_count: int = Field(ge=0)
    inf_count: int = Field(ge=0)
    finite_count: int = Field(ge=0)

    # Computed over finite values only; None when nothing is finite.
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    p01: float | None = None
    p50: float | None = None
    p99: float | None = None

    constant_vertices: int | None = Field(
        default=None,
        description="Columns with zero variance over time. A large count suggests a dead model.",
    )
    temporal_variance_mean: float | None = Field(
        default=None,
        description=(
            "Mean across vertices of each vertex's variance over time. Near zero means the "
            "prediction does not change with the content, which would make A/B testing "
            "meaningless."
        ),
    )

    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class PerformanceMetrics(BaseModel):
    """Wall-clock and memory cost of a run. The Phase 1 optimization baseline."""

    model_config = ConfigDict(frozen=True)

    preprocessing_seconds: float = Field(ge=0.0)
    model_load_seconds: float = Field(ge=0.0)
    inference_seconds: float = Field(ge=0.0)
    postprocessing_seconds: float = Field(ge=0.0)
    validation_seconds: float = Field(ge=0.0)
    persistence_seconds: float = Field(ge=0.0)
    total_seconds: float = Field(ge=0.0)

    peak_gpu_allocated_bytes: int | None = None
    peak_gpu_reserved_bytes: int | None = None
    peak_host_rss_bytes: int | None = None

    realtime_factor: float | None = Field(
        default=None,
        description="total_seconds / stimulus duration. 60.0 means 60 s of compute per second "
        "of content.",
    )


class ArtifactPaths(BaseModel):
    """Where a run's files live. Paths are relative to `run_dir` unless absolute."""

    model_config = ConfigDict(frozen=True)

    run_dir: Path
    manifest: Path
    stimulus: Path
    predictions: Path
    prediction_metadata: Path
    temporal: Path | None = None
    events: Path | None = None
    events_summary: Path | None = None
    validation: Path
    performance: Path
    provenance: Path
    diagnostics: tuple[Path, ...] = ()

    def resolve(self, relative: Path) -> Path:
        """Resolve a run-relative path against the run directory."""
        return relative if relative.is_absolute() else self.run_dir / relative


class PredictionResult(BaseModel):
    """The standardized output of a cortical response prediction.

    Consumable with nothing installed but ``pydantic``:

        result = PredictionResult.model_validate_json(manifest_path.read_text())
        matrix = np.load(result.artifacts.resolve(result.prediction.artifact_path))
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    schema_version: str = Field(default="1.0", description="Contract version for Phase 2+.")
    run_id: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    status: Literal["completed", "completed_with_warnings", "failed"] = "completed"

    stimulus: StimulusInput
    model: ModelMetadata
    prediction: PredictionArrayMetadata
    temporal: TemporalMetadata
    cortical: CorticalMetadata
    validation: PredictionValidation
    performance: PerformanceMetrics
    provenance: RunProvenance
    artifacts: ArtifactPaths

    interpretation_notice: str = Field(
        default=(
            "These are MODEL-PREDICTED cortical fMRI/BOLD responses for an average subject, "
            "not measurements from any real viewer. They do not establish emotion, memory, "
            "attention, preference or purchase intent, and are not for medical use. "
            "See docs/scientific_limitations.md."
        ),
        description="Travels with every result so downstream surfaces cannot omit it.",
    )

    def summary_lines(self) -> list[str]:
        """Human-readable inspection summary. Not ROI analysis — validation only."""
        v = self.validation
        p = self.performance

        def num(value: float | None, fmt: str = ".6f") -> str:
            return "n/a" if value is None else format(value, fmt)

        axes = ", ".join(self.prediction.axis_names)
        timeline = (
            "contiguous"
            if self.temporal.timeline_is_contiguous
            else "GAPPY - rows were dropped, use segment_start_seconds"
        )
        memory = (
            f"{p.peak_gpu_allocated_bytes / 1e9:.2f} GB"
            if p.peak_gpu_allocated_bytes is not None
            else "n/a (no CUDA device)"
        )

        return [
            f"Shape:             {list(self.prediction.shape)}  ({axes})",
            f"Temporal samples:  {self.prediction.temporal_samples} "
            f"@ TR={self.temporal.tr_seconds:.3f}s",
            f"Cortical vertices: {self.prediction.cortical_features} "
            f"({self.cortical.surface_space})",
            f"Timeline:          {timeline}",
            f"Mean response:     {num(v.mean)}",
            f"Std:               {num(v.std)}",
            f"Min / Max:         {num(v.min)} / {num(v.max)}",
            f"p01 / p50 / p99:   {num(v.p01)} / {num(v.p50)} / {num(v.p99)}",
            f"NaNs:              {v.nan_count}",
            f"Infs:              {v.inf_count}",
            f"Inference time:    {p.inference_seconds:.2f}s (total {p.total_seconds:.2f}s)",
            f"Peak GPU memory:   {memory}",
        ]
