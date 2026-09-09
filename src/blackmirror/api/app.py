"""Visualization API.

A thin HTTP surface over Phase 1 artifacts. Every route here is read-only with
exactly one exception, named below: responses are assembled from files a
completed run already wrote. Rotating the brain or dragging the timeline must
never touch a model.

THE EXCEPTION: the Phase 8 re-simulation routes do run TRIBE, because measuring
a proposed variant is the entire point of that phase. They never do it inside
the request. A start binds the run durably, hands it to a background worker and
returns the checkpoint that is now on disk; clients poll for the rest. No other
route can reach a model, and this one is confined to `/api/resimulations`.

Transport strategy — arrays are served as raw little-endian binary, not JSON.
A prediction matrix is `T x 20484 float32`; JSON would inflate that roughly 8x
and force the browser to parse millions of numbers. Binary lands directly in a
`Float32Array` and can be uploaded to the GPU without a copy. Metadata stays
JSON, where readability matters and size does not.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from blackmirror.api.analytics_loader import AnalyticsLoader
from blackmirror.api.comparison_loader import ComparisonLoader
from blackmirror.api.content_loader import ContentLoader, get_content_loader
from blackmirror.api.contracts import RunListItem, RunSummary
from blackmirror.api.loader import (
    BinaryArray,
    MeshAlignmentError,
    VisualizationError,
    VisualizationLoader,
    get_loader,
)
from blackmirror.api.optimization_loader import (
    OptimizationLoader,
    OptimizationLoadError,
)
from blackmirror.api.resimulation_loader import (
    RequestBuildError,
    ResimulationLoader,
    ResimulationLoadError,
    get_resimulation_loader,
)
from blackmirror.api.scoring_loader import ScoringLoader, VariantLoadError
from blackmirror.api.search_loader import (
    SearchLoader,
    SearchLoadError,
    get_search_loader,
)
from blackmirror.content.schemas import ContentAnalysisResult
from blackmirror.errors import ArtifactReadError
from blackmirror.optimization.schemas import (
    OptimizationRequest,
    OptimizationResult,
    ProposedVariantSpec,
    RecommendationReview,
)
from blackmirror.resimulation.schemas import (
    ResimulationConflict,
    ResimulationResult,
    StopReason,
)
from blackmirror.scoring.schemas import (
    ExperimentScoreResult,
    NeuralObjective,
    ScoreExplanation,
)
from blackmirror.surrogate.storage import SurrogateStore
from blackmirror.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_ID_PATTERN = r"^[A-Za-z0-9_.\-]{1,128}$"
RunId = Annotated[str, PathParam(pattern=_ID_PATTERN)]
RequestRunId = Annotated[str, Field(pattern=_ID_PATTERN)]

#: Arrays are immutable once a run completes, so they cache hard.
_IMMUTABLE = "public, max-age=31536000, immutable"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging("INFO")
    logger.info(
        "Visualization API ready (read-only except the Phase 8 re-simulation routes, "
        "which run TRIBE on a background worker)"
    )
    yield


app = FastAPI(
    title="NeuroSplit Visualization API",
    version="0.2.0",
    description=(
        "Access to completed TRIBE v2 inference runs for cortical visualization, "
        "goal-conditioned scoring and optimization. Read-only except the Phase 8 "
        "re-simulation routes, which queue inference on a background worker."
    ),
    lifespan=_lifespan,
)

# The Next.js dev server is a separate origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def loader_dep() -> VisualizationLoader:
    return get_loader()


Loader = Annotated[VisualizationLoader, Depends(loader_dep)]


def analytics_loader_dep() -> AnalyticsLoader:
    loader = get_loader()
    return AnalyticsLoader(loader.settings.artifact_dir)


Analytics = Annotated[AnalyticsLoader, Depends(analytics_loader_dep)]


def comparison_loader_dep() -> ComparisonLoader:
    loader = get_loader()
    return ComparisonLoader(loader.settings.artifact_dir)


Comparisons = Annotated[ComparisonLoader, Depends(comparison_loader_dep)]


def scoring_loader_dep() -> ScoringLoader:
    loader = get_loader()
    return ScoringLoader(loader.settings.artifact_dir)


Scoring = Annotated[ScoringLoader, Depends(scoring_loader_dep)]


def optimization_loader_dep() -> OptimizationLoader:
    loader = get_loader()
    return OptimizationLoader(loader.settings.artifact_dir)


Optimization = Annotated[OptimizationLoader, Depends(optimization_loader_dep)]


def resimulation_loader_dep() -> ResimulationLoader:
    loader = get_loader()
    return get_resimulation_loader(loader.settings.artifact_dir, settings=loader.settings)


Resimulation = Annotated[ResimulationLoader, Depends(resimulation_loader_dep)]


class CreateComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    reference_run_id: RequestRunId
    candidate_run_ids: tuple[RequestRunId, ...] = Field(min_length=1)
    alignment_method: Literal["exact_observed_intersection", "linear_candidate_to_reference"] = (
        "exact_observed_intersection"
    )
    max_interpolation_gap_seconds: float = Field(default=10.0, gt=0)

    @field_validator("reference_run_id", "candidate_run_ids")
    @classmethod
    def reject_dot_path_ids(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        values = (value,) if isinstance(value, str) else value
        if any(item in {".", ".."} for item in values):
            raise ValueError("run ids cannot be dot path components")
        return value


def _binary(array: BinaryArray) -> Response:
    """Send a typed array with the metadata a client needs to interpret it."""
    return Response(
        content=array.data,
        media_type="application/octet-stream",
        headers={
            "X-Array-Dtype": array.dtype,
            "X-Array-Shape": ",".join(str(d) for d in array.shape),
            "Cache-Control": _IMMUTABLE,
            # Custom headers are invisible to cross-origin JS unless exposed.
            "Access-Control-Expose-Headers": "X-Array-Dtype, X-Array-Shape",
        },
    )


def _resolve(run_id: str, action: str, call: object) -> object:
    """Translate domain errors into precise HTTP status codes."""
    try:
        return call()  # type: ignore[operator]
    except ArtifactReadError as exc:
        raise HTTPException(404, f"Run '{run_id}' not found or unreadable: {exc}") from exc
    except MeshAlignmentError as exc:
        # 409: the run exists but cannot be rendered correctly. Never fall back
        # to rendering something misaligned.
        raise HTTPException(409, f"Cannot visualize run '{run_id}': {exc}") from exc
    except VisualizationError as exc:
        raise HTTPException(422, f"{action} failed for run '{run_id}': {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, object]:
    """Health, and an honest statement of what this process is able to do.

    `runs_inference` reports a capability, not current activity: the Phase 8
    routes can start a TRIBE pass on a background worker. Every other route
    reads files.
    """
    return {
        "status": "ok",
        "service": "neurosplit-visualization",
        "runs_inference": True,
        "inference_scope": "phase8_resimulation_only",
    }


@app.get("/api/runs", response_model=list[RunListItem])
def list_runs(loader: Loader, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[RunListItem]:
    """Completed runs, newest first."""
    return loader.list_runs(limit=limit)


@app.get("/api/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: RunId, loader: Loader) -> RunSummary:
    """Everything a client needs before fetching any binary payload.

    Alignment between prediction and mesh is validated here, so a run that
    cannot be rendered correctly fails at the first request rather than after
    the browser has drawn a plausible but wrong brain.
    """
    return _resolve(run_id, "Summary", lambda: loader.build_summary(run_id))  # type: ignore[return-value]


@app.get("/api/runs/{run_id}/predictions")
def get_predictions(run_id: RunId, loader: Loader) -> Response:
    """Raw [T, V] float32 predictions, row-major. Unmodified Phase 1 output."""
    array = _resolve(run_id, "Predictions", lambda: loader.load_predictions(run_id))
    return _binary(array)  # type: ignore[arg-type]


@app.get("/api/runs/{run_id}/timeline")
def get_timeline(run_id: RunId, loader: Loader) -> Response:
    """Per-row playback clock (stimulus_time_seconds), float32."""
    array = _resolve(run_id, "Timeline", lambda: loader.load_timeline(run_id))
    return _binary(array)  # type: ignore[arg-type]


@app.get("/api/runs/{run_id}/analytics")
def get_analytics(run_id: RunId, loader: Analytics) -> object:
    """Versioned Phase 3 metadata, ROI summaries, events, and array descriptors."""
    try:
        return loader.metadata(run_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(404, f"Analytics for run '{run_id}' are unavailable: {exc}") from exc


@app.get("/api/runs/{run_id}/analytics/arrays/{key}")
def get_analytics_array(run_id: RunId, key: str, loader: Analytics) -> Response:
    """One derived numeric array as little-endian binary."""
    try:
        return _binary(loader.array(run_id, key))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(404, f"Analytics for run '{run_id}' are unavailable: {exc}") from exc


@app.get("/api/comparisons")
def list_comparisons(loader: Comparisons) -> object:
    """Persisted controlled comparisons, newest first."""
    return loader.list_comparisons()


@app.post("/api/comparisons", status_code=201)
def create_comparison(request: CreateComparisonRequest, loader: Comparisons) -> object:
    """Create and atomically persist a controlled comparison from existing runs."""
    try:
        return loader.create(
            request.reference_run_id,
            request.candidate_run_ids,
            alignment_method=request.alignment_method,
            max_interpolation_gap_seconds=request.max_interpolation_gap_seconds,
        )
    except (ArtifactReadError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise HTTPException(422, f"Comparison could not be created: {exc}") from exc


@app.get("/api/comparisons/{comparison_id}")
def get_comparison(comparison_id: RunId, loader: Comparisons) -> object:
    """Versioned comparison metadata, compatibility checks, summaries, and events."""
    try:
        return loader.metadata(comparison_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(404, f"Comparison '{comparison_id}' is unavailable: {exc}") from exc


@app.get("/api/comparisons/{comparison_id}/pairs/{candidate_run_id}/arrays/{key}")
def get_comparison_array(
    comparison_id: RunId,
    candidate_run_id: RunId,
    key: str,
    loader: Comparisons,
) -> Response:
    """One persisted pairwise difference array as little-endian binary."""
    try:
        return _binary(loader.array(comparison_id, candidate_run_id, key))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(404, f"Comparison '{comparison_id}' is unavailable: {exc}") from exc


@app.get("/api/runs/{run_id}/durations")
def get_durations(run_id: RunId, loader: Loader) -> Response:
    """Per-row segment durations, float32.

    Row `i` covers `[timeline[i], timeline[i] + durations[i])`. A client needs
    this to distinguish "a prediction covers this moment" from "the nearest
    prediction is over there".
    """
    array = _resolve(run_id, "Durations", lambda: loader.load_durations(run_id))
    return _binary(array)  # type: ignore[arg-type]


@app.get("/api/runs/{run_id}/mesh")
def get_mesh_manifest(run_id: RunId, loader: Loader) -> dict[str, object]:
    """Mesh manifest for the surface space this run predicts onto."""

    def build() -> dict[str, object]:
        result = loader.get_result(run_id)
        return loader.mesh_manifest(result.cortical.surface_space)

    return _resolve(run_id, "Mesh manifest", build)  # type: ignore[return-value]


@app.get("/api/runs/{run_id}/mesh/{hemisphere}/{surface}/vertices")
def get_mesh_vertices(
    run_id: RunId,
    hemisphere: Literal["left", "right"],
    surface: Literal["inflated", "pial"],
    loader: Loader,
) -> Response:
    """Vertex coordinates for one hemisphere, [N, 3] float32, in millimetres."""

    def build() -> BinaryArray:
        space = loader.get_result(run_id).cortical.surface_space
        return loader.load_mesh_array(hemisphere, f"{surface}_vertices", space)

    return _binary(_resolve(run_id, "Mesh vertices", build))  # type: ignore[arg-type]


@app.get("/api/runs/{run_id}/mesh/{hemisphere}/faces")
def get_mesh_faces(run_id: RunId, hemisphere: Literal["left", "right"], loader: Loader) -> Response:
    """Triangles for one hemisphere, [F, 3] int32, hemisphere-local indices."""

    def build() -> BinaryArray:
        space = loader.get_result(run_id).cortical.surface_space
        return loader.load_mesh_array(hemisphere, "faces", space)

    return _binary(_resolve(run_id, "Mesh faces", build))  # type: ignore[arg-type]


@app.get("/api/runs/{run_id}/mesh/medial-wall")
def get_medial_wall(run_id: RunId, loader: Loader) -> Response:
    """Medial-wall mask over the full vertex axis, uint8 (1 = medial wall)."""

    def build() -> BinaryArray:
        space = loader.get_result(run_id).cortical.surface_space
        return loader.load_medial_wall(space)

    return _binary(_resolve(run_id, "Medial wall", build))  # type: ignore[arg-type]


def content_loader_dep() -> ContentLoader:
    return get_content_loader()


Content = Annotated[ContentLoader, Depends(content_loader_dep)]


def _content(loader: ContentLoader, run_id: str) -> ContentAnalysisResult:
    """Load a run's content analysis, or 404 with how to create it."""
    if not loader.available(run_id):
        raise HTTPException(
            404,
            f"No content analysis for run '{run_id}'. "
            f"Derive it with `blackmirror analyze-content {run_id}`.",
        )
    try:
        return loader.get(run_id)
    except ArtifactReadError as exc:
        raise HTTPException(404, f"Content analysis for '{run_id}' is unreadable: {exc}") from exc


@app.get("/api/runs/{run_id}/content-analysis")
def get_content_analysis(run_id: RunId, loader: Content) -> object:
    """Full Phase 4 content analysis for a run.

    404s when the analysis has not been derived: this endpoint never triggers
    the pipeline, which takes tens of seconds and loads several models.
    """
    return _content(loader, run_id)


@app.get("/api/runs/{run_id}/content-analysis/events")
def get_content_events(run_id: RunId, loader: Content) -> list[object]:
    """Fused content events only — the timeline payload."""
    analysis = _content(loader, run_id)
    return [event.model_dump(mode="json") for event in analysis.events]


@app.get("/api/runs/{run_id}/content-analysis/transcript")
def get_content_transcript(run_id: RunId, loader: Content) -> list[object]:
    """Timestamped transcript with word timings."""
    analysis = _content(loader, run_id)
    return [segment.model_dump(mode="json") for segment in analysis.transcript]


@app.get("/api/runs/{run_id}/content-analysis/scenes")
def get_content_scenes(run_id: RunId, loader: Content) -> dict[str, object]:
    """Scenes and the shots they group."""
    analysis = _content(loader, run_id)
    return {
        "scenes": [scene.model_dump(mode="json") for scene in analysis.scenes],
        "shots": [shot.model_dump(mode="json") for shot in analysis.shots],
    }


@app.get("/api/runs/{run_id}/content-analysis/context")
def get_content_context(
    run_id: RunId,
    loader: Content,
    time: Annotated[float, Query(ge=0.0, description="Stimulus time in seconds.")],
    window: Annotated[float, Query(ge=0.0, le=30.0)] = 0.0,
) -> dict[str, object]:
    """What the stimulus was doing at, or around, a given time."""
    _content(loader, run_id)  # 404s consistently when the analysis is absent
    return loader.context_at(run_id, time, window)


@app.get("/api/runs/{run_id}/content-analysis/associations")
def get_content_associations(run_id: RunId, loader: Content) -> list[object]:
    """Neural events with the content that co-occurred around them.

    Temporal co-occurrence only — every record carries that statement.
    """
    analysis = _content(loader, run_id)
    return [item.model_dump(mode="json") for item in analysis.associations]


@app.get("/api/runs/{run_id}/content-analysis/features")
def get_content_features(run_id: RunId, loader: Content) -> Response:
    """Content feature matrix as raw float32, row-major [T, F]."""
    _content(loader, run_id)

    def build() -> Response:
        matrix, times, names = loader.features(run_id)
        payload = np.ascontiguousarray(matrix, dtype=np.float32).tobytes()
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "X-Array-Dtype": "float32",
                "X-Array-Shape": ",".join(str(d) for d in matrix.shape),
                "X-Feature-Names": ",".join(names),
                "X-Time-Count": str(int(times.size)),
                "Cache-Control": _IMMUTABLE,
                "Access-Control-Expose-Headers": (
                    "X-Array-Dtype, X-Array-Shape, X-Feature-Names, X-Time-Count"
                ),
            },
        )

    return _resolve(run_id, "Content features", build)  # type: ignore[return-value]


@app.get("/api/runs/{run_id}/content-analysis/features/times")
def get_content_feature_times(run_id: RunId, loader: Content) -> Response:
    """Exact time coordinate for every content feature-matrix row."""
    _content(loader, run_id)
    _, times, _ = loader.features(run_id)
    values = np.ascontiguousarray(times, dtype=np.float32)
    return Response(
        content=values.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Array-Dtype": "float32",
            "X-Array-Shape": str(values.size),
            "Cache-Control": _IMMUTABLE,
            "Access-Control-Expose-Headers": "X-Array-Dtype, X-Array-Shape",
        },
    )


@app.get("/api/runs/{run_id}/content-analysis/keyframes/{name}")
def get_content_keyframe(run_id: RunId, name: str, loader: Content) -> FileResponse:
    """Serve one extracted keyframe image."""
    path = _resolve(run_id, "Keyframe", lambda: loader.keyframe_path(run_id, name))
    return FileResponse(path, headers={"Cache-Control": _IMMUTABLE})  # type: ignore[arg-type]


@app.get("/api/runs/{run_id}/stimulus")
def get_stimulus(run_id: RunId, loader: Loader) -> FileResponse:
    """Stream the original stimulus so the UI can play it beside the brain.

    Phase 1 never copies media into artifacts, so this reads from the recorded
    source path and 404s cleanly when the file has moved.
    """
    path = _resolve(run_id, "Stimulus", lambda: loader.stimulus_path(run_id))
    return FileResponse(path, headers={"Accept-Ranges": "bytes"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Phase 6 - goal-conditioned scoring
# ---------------------------------------------------------------------------


class CreateScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    run_ids: tuple[RequestRunId, ...] = Field(min_length=1)
    objectives: tuple[NeuralObjective, ...] = Field(min_length=1)
    baseline_run_id: RequestRunId | None = None
    with_networks: bool = False
    reuse_cache: bool = True

    @field_validator("run_ids", "baseline_run_id")
    @classmethod
    def _safe_ids(
        cls, value: tuple[str, ...] | str | None
    ) -> tuple[str, ...] | str | None:
        values = value if isinstance(value, tuple) else (() if value is None else (value,))
        if any(item in {".", ".."} for item in values):
            raise ValueError("run ids cannot be dot path components")
        if isinstance(value, tuple) and len(set(value)) != len(value):
            raise ValueError("run ids must be distinct")
        return value


@app.get("/api/objectives/metrics")
def objective_metrics(scoring: Scoring) -> dict[str, dict[str, str]]:
    """Every registered objective metric, with formula and limitations."""
    return scoring.metrics()


@app.get("/api/experiments")
def list_experiments(scoring: Scoring) -> list[str]:
    return scoring.experiments()


@app.post("/api/experiments/{experiment_id}/score", status_code=201)
def create_score(
    experiment_id: RunId, request: CreateScoreRequest, scoring: Scoring
) -> ExperimentScoreResult:
    """Score completed runs against an objective set. Never runs inference."""
    try:
        return scoring.create(
            experiment_id,
            tuple(request.run_ids),
            tuple(request.objectives),
            baseline_run_id=request.baseline_run_id,
            with_networks=request.with_networks,
            reuse_cache=request.reuse_cache,
        )
    except VariantLoadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/experiments/{experiment_id}/scores")
def list_scores(experiment_id: RunId, scoring: Scoring) -> list[ExperimentScoreResult]:
    return scoring.scores(experiment_id)


@app.get("/api/experiments/{experiment_id}/scores/{objective_set_hash}")
def get_score(
    experiment_id: RunId, objective_set_hash: str, scoring: Scoring
) -> ExperimentScoreResult:
    try:
        return scoring.score(experiment_id, objective_set_hash)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail="score not found") from exc


@app.get("/api/experiments/{experiment_id}/scores/{objective_set_hash}/explanation")
def get_explanation(
    experiment_id: RunId, objective_set_hash: str, scoring: Scoring
) -> ScoreExplanation:
    try:
        explanation = scoring.explanation(experiment_id, objective_set_hash)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if explanation is None:
        raise HTTPException(status_code=404, detail="no explanation stored for this score")
    return explanation


# ---------------------------------------------------------------------------
# Phase 7 - optimization agent
# ---------------------------------------------------------------------------


class CreateOptimizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request: OptimizationRequest
    #: variant id -> run id, when a variant was scored under a different label.
    variant_runs: dict[str, RequestRunId] | None = None


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review: RecommendationReview


class CandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    proposed_variant_id: RequestRunId


@app.post("/api/experiments/{experiment_id}/optimize", status_code=201)
def create_optimization(
    experiment_id: RunId, body: CreateOptimizationRequest, optimization: Optimization
) -> OptimizationResult:
    """Produce candidate interventions. Never edits media, never runs TRIBE."""
    if body.request.experiment_id != experiment_id:
        raise HTTPException(
            status_code=422, detail="request experiment_id must match the path"
        )
    try:
        return optimization.create(body.request, variant_runs=body.variant_runs)
    except OptimizationLoadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/experiments/{experiment_id}/optimization")
def list_optimizations(experiment_id: RunId, optimization: Optimization) -> list[str]:
    return optimization.keys(experiment_id)


@app.get("/api/experiments/{experiment_id}/optimization/{key}")
def get_optimization(
    experiment_id: RunId, key: str, optimization: Optimization
) -> OptimizationResult:
    try:
        return optimization.read(experiment_id, key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail="optimization not found") from exc


@app.get("/api/experiments/{experiment_id}/optimization/{key}/reviews")
def list_reviews(
    experiment_id: RunId, key: str, optimization: Optimization
) -> list[RecommendationReview]:
    try:
        return optimization.reviews(experiment_id, key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/experiments/{experiment_id}/optimization/{key}/reviews", status_code=201)
def record_review(
    experiment_id: RunId, key: str, body: ReviewRequest, optimization: Optimization
) -> list[RecommendationReview]:
    """Approve, modify or reject one recommendation. Required before a candidate."""
    try:
        return optimization.review(experiment_id, key, body.review)
    except OptimizationLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail="optimization not found") from exc


@app.post("/api/experiments/{experiment_id}/optimization/{key}/candidate-specs", status_code=201)
def create_candidate(
    experiment_id: RunId, key: str, body: CandidateRequest, optimization: Optimization
) -> ProposedVariantSpec:
    """Turn approved reviews into a Phase 8 specification. Runs no simulation."""
    try:
        return optimization.build_candidate(experiment_id, key, body.proposed_variant_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail="optimization not found") from exc


@app.get("/api/experiments/{experiment_id}/optimization/{key}/candidate-specs")
def list_candidates(
    experiment_id: RunId, key: str, optimization: Optimization
) -> list[ProposedVariantSpec]:
    try:
        return optimization.specs(experiment_id, key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 8 - bounded re-simulation
#
# The only routes in this API that can reach a model. A start returns as soon
# as the run is durable; the pipeline behind it runs on a background worker and
# is observed by polling, because a full pass takes hours.
# ---------------------------------------------------------------------------


class CreateResimulationRequest(BaseModel):
    """What a person actually knows, as opposed to what the schema demands.

    The objective set, its hashes, the per-hypothesis direction map and the
    content hashes are all derived from stored Phase 6 and Phase 7 records. A
    client cannot supply them, which is deliberate: supplying them by hand is
    how a re-simulation ends up measuring an objective nobody approved.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    resimulation_id: RequestRunId
    optimization_request_key: str = Field(pattern=r"^[0-9a-f]{8,64}$")
    proposed_variant_id: RequestRunId
    #: Absolute path to the variant the human built. Media is never uploaded
    #: or copied; Phase 8 reads it where it lies and records its hash.
    variant_media_path: str = Field(min_length=1)
    source_media_path: str | None = None
    adapter_id: str = Field(default="user-supplied", min_length=1, max_length=64)
    parameters: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    max_attempts: int = Field(default=3, ge=1, le=20)
    outcome_tolerance: float = Field(default=1e-9, ge=0)

    @field_validator("resimulation_id", "proposed_variant_id")
    @classmethod
    def _reject_dot_ids(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("ids cannot be dot path components")
        return value


class StopResimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Literal["cancelled", "timeout", "resource_limit"] = "cancelled"


class ResimulationView(BaseModel):
    """A durable state plus the two facts that are not in it.

    `worker_alive` is derived from the run's lock at read time and never
    persisted. A run whose worker was killed still says `active` in its own
    file, because the process that would have recorded the failure is the one
    that died; without this field there is no way to tell that from progress.
    """

    model_config = ConfigDict(extra="forbid")

    state: ResimulationResult
    worker_alive: bool
    worker_error: str | None = None


def _view(loader: ResimulationLoader, state: ResimulationResult) -> ResimulationView:
    resimulation_id = state.request.resimulation_id
    return ResimulationView(
        state=state,
        worker_alive=loader.worker_alive(resimulation_id),
        worker_error=loader.failure(resimulation_id),
    )


@app.post("/api/experiments/{experiment_id}/resimulations", status_code=201)
def create_resimulation(
    experiment_id: RunId, body: CreateResimulationRequest, resimulation: Resimulation
) -> ResimulationView:
    """Bind an approved candidate to real media and queue one measured pass."""
    try:
        request = resimulation.build(
            resimulation_id=body.resimulation_id,
            experiment_id=experiment_id,
            optimization_request_key=body.optimization_request_key,
            proposed_variant_id=body.proposed_variant_id,
            variant_media=FilePath(body.variant_media_path),
            source_media=(
                FilePath(body.source_media_path) if body.source_media_path else None
            ),
            adapter_id=body.adapter_id,
            parameters=dict(body.parameters),
            max_attempts=body.max_attempts,
            outcome_tolerance=body.outcome_tolerance,
        )
        state = resimulation.create(request)
    except ResimulationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RequestBuildError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ResimulationLoadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _view(resimulation, state)


@app.get("/api/resimulations")
def list_resimulations(
    resimulation: Resimulation,
    experiment_id: Annotated[str | None, Query(pattern=_ID_PATTERN)] = None,
) -> list[ResimulationView]:
    return [_view(resimulation, state) for state in resimulation.list(experiment_id)]


@app.get("/api/resimulations/{resimulation_id}")
def get_resimulation(resimulation_id: RunId, resimulation: Resimulation) -> ResimulationView:
    try:
        return _view(resimulation, resimulation.get(resimulation_id))
    except OSError as exc:
        raise HTTPException(status_code=404, detail="re-simulation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/resimulations/{resimulation_id}/resume")
def resume_resimulation(resimulation_id: RunId, resimulation: Resimulation) -> ResimulationView:
    """Verify every upstream hash and continue after the last durable stage."""
    try:
        state = resimulation.resume(resimulation_id)
    except ResimulationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ResimulationLoadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=404, detail="re-simulation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _view(resimulation, state)


@app.post("/api/resimulations/{resimulation_id}/stop")
def stop_resimulation(
    resimulation_id: RunId, body: StopResimulationRequest, resimulation: Resimulation
) -> ResimulationView:
    """Record a stop the worker observes between stages, not a kill."""
    try:
        state = resimulation.stop(resimulation_id, StopReason(body.reason))
    except OSError as exc:
        raise HTTPException(status_code=404, detail="re-simulation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _view(resimulation, state)


# ---------------------------------------------------------------------------
# Phase 9 - automated search
#
# Read-only except the control routes. Starting a search queues Phase 8
# evaluations on a background worker; a six-candidate search is most of a
# working day, so nothing here waits for one.
# ---------------------------------------------------------------------------


def search_loader_dep() -> SearchLoader:
    loader = get_loader()
    return get_search_loader(loader.settings.artifact_dir, settings=loader.settings)


Search = Annotated[SearchLoader, Depends(search_loader_dep)]


class SearchControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: RequestRunId | None = None


def _search(search_id: str, call: object) -> object:
    """Translate storage and control errors into precise status codes."""
    try:
        return call()  # type: ignore[operator]
    except SearchLoadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"search '{search_id}' not found"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=404, detail=f"search '{search_id}' is unreadable: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/search")
def list_searches(search: Search) -> list[dict[str, object]]:
    """Every recorded search, newest activity first."""
    return [search.summary(search_id) for search_id in search.ids()]


@app.get("/api/search/{search_id}")
def get_search(search_id: RunId, search: Search) -> dict[str, object]:
    return _search(search_id, lambda: search.summary(search_id))  # type: ignore[return-value]


@app.get("/api/search/{search_id}/state")
def get_search_state(search_id: RunId, search: Search) -> dict[str, object]:
    """The full checkpoint, including the strategy's resumable state."""
    return _search(search_id, lambda: search.state(search_id))  # type: ignore[return-value]


@app.get("/api/search/{search_id}/best")
def get_search_best(search_id: RunId, search: Search) -> dict[str, object]:
    best = _search(search_id, lambda: search.best(search_id))
    if best is None:
        raise HTTPException(
            status_code=404,
            detail=f"search '{search_id}' has no eligible best candidate yet",
        )
    return best  # type: ignore[return-value]


@app.get("/api/search/{search_id}/trajectory")
def get_search_trajectory(search_id: RunId, search: Search) -> list[dict[str, object]]:
    """Best-so-far against evaluation count, for the optimization plot."""
    return _search(search_id, lambda: search.trajectory(search_id))  # type: ignore[return-value]


@app.get("/api/search/{search_id}/population")
def get_search_population(search_id: RunId, search: Search) -> list[dict[str, object]]:
    return _search(search_id, lambda: search.population(search_id))  # type: ignore[return-value]


@app.get("/api/search/{search_id}/budget")
def get_search_budget(search_id: RunId, search: Search) -> dict[str, object]:
    return _search(search_id, lambda: search.budget(search_id))  # type: ignore[return-value]


@app.get("/api/search/{search_id}/events")
def get_search_events(search_id: RunId, search: Search) -> list[dict[str, object]]:
    return _search(search_id, lambda: search.events(search_id))  # type: ignore[return-value]


@app.get("/api/search/{search_id}/results")
def get_search_results(search_id: RunId, search: Search) -> list[dict[str, object]]:
    """Every evaluated candidate, including failures and disqualifications."""
    return _search(search_id, lambda: search.results(search_id))  # type: ignore[return-value]


@app.get("/api/search/{search_id}/pareto")
def get_search_pareto(search_id: RunId, search: Search) -> dict[str, object]:
    front = _search(search_id, lambda: search.pareto(search_id))
    if front is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"search '{search_id}' has no Pareto frontier; it declared a single "
                f"objective, so there is no trade-off to describe"
            ),
        )
    return front  # type: ignore[return-value]


@app.get("/api/search/{search_id}/report")
def get_search_report(search_id: RunId, search: Search) -> dict[str, object]:
    report = _search(search_id, lambda: search.report(search_id))
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"search '{search_id}' has not finished, so it has no report yet",
        )
    return report  # type: ignore[return-value]


@app.post("/api/search/{search_id}/pause")
def pause_search(search_id: RunId, search: Search) -> dict[str, object]:
    """Hold at the next candidate boundary. The run stays resumable."""
    return _search(search_id, lambda: search.pause(search_id))  # type: ignore[return-value]


@app.post("/api/search/{search_id}/resume")
def resume_search(search_id: RunId, search: Search) -> dict[str, object]:
    return _search(search_id, lambda: search.unpause(search_id))  # type: ignore[return-value]


@app.post("/api/search/{search_id}/stop")
def stop_search(search_id: RunId, search: Search) -> dict[str, object]:
    """End the search at the next candidate boundary, not mid-evaluation."""
    return _search(search_id, lambda: search.stop(search_id))  # type: ignore[return-value]


@app.post("/api/search/{search_id}/pin")
def pin_candidate(
    search_id: RunId, body: SearchControlRequest, search: Search
) -> dict[str, object]:
    """Force a candidate to the front of the queue (Step 72)."""
    if body.candidate_id is None:
        raise HTTPException(status_code=422, detail="candidate_id is required")
    return _search(search_id, lambda: search.pin(search_id, str(body.candidate_id)))  # type: ignore[return-value]


@app.post("/api/search/{search_id}/eliminate")
def eliminate_candidate(
    search_id: RunId, body: SearchControlRequest, search: Search
) -> dict[str, object]:
    """Refuse a candidate before it costs an evaluation."""
    if body.candidate_id is None:
        raise HTTPException(status_code=422, detail="candidate_id is required")
    return _search(  # type: ignore[return-value]
        search_id, lambda: search.eliminate(search_id, str(body.candidate_id))
    )


# ---------------------------------------------------------------------------
# Phase 10 - surrogate modelling
#
# Read-only. A surrogate is trained by the optimization loop, not by an HTTP
# request; these routes report what it learned and how far it should be
# trusted. Every score served here is an estimate, never a measurement.
# ---------------------------------------------------------------------------


def _surrogate_store(search_id: str) -> SurrogateStore:
    loader = get_loader()
    try:
        return SurrogateStore(loader.settings.artifact_dir, search_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_surrogate(search_id: str) -> SurrogateStore:
    store = _surrogate_store(search_id)
    if not store.exists:
        raise HTTPException(
            status_code=404,
            detail=(
                f"search '{search_id}' has no surrogate; one is created by the "
                f"optimization loop once enough real evaluations exist"
            ),
        )
    return store


@app.get("/api/search/{search_id}/surrogate")
def get_surrogate(search_id: RunId) -> dict[str, object]:
    """The current model, its training set, and whether it may be trusted."""
    store = _require_surrogate(search_id)
    dataset = store.read_dataset()
    diagnostics = store.read_diagnostics() or {}
    latest = store.latest_model()
    return {
        "search_id": search_id,
        "training_samples": 0 if dataset is None else dataset.size,
        "target_spread": None if dataset is None else dataset.target_spread(),
        "excluded_records": 0 if dataset is None else len(dataset.excluded),
        "state": diagnostics.get("state"),
        "trusted": diagnostics.get("trusted"),
        "trust_reason": diagnostics.get("trust_reason"),
        "model": None if latest is None else latest.model_dump(mode="json"),
        "model_versions": len(store.read_models()),
    }


@app.get("/api/search/{search_id}/surrogate/metrics")
def get_surrogate_metrics(search_id: RunId) -> dict[str, object]:
    """Validation quality. The prefix scheme is the one that reflects use."""
    diagnostics = _require_surrogate(search_id).read_diagnostics()
    if diagnostics is None:
        raise HTTPException(
            status_code=404,
            detail=f"search '{search_id}' has a surrogate but no diagnostics yet",
        )
    return diagnostics


@app.get("/api/search/{search_id}/surrogate/dataset")
def get_surrogate_dataset(search_id: RunId) -> dict[str, object]:
    """The real observations the surrogate was trained on, and nothing else."""
    dataset = _require_surrogate(search_id).read_dataset()
    if dataset is None:  # pragma: no cover - exists() already checked
        raise HTTPException(status_code=404, detail="dataset unreadable")
    return dataset.model_dump(mode="json")


@app.get("/api/search/{search_id}/surrogate/models")
def get_surrogate_models(search_id: RunId) -> list[dict[str, object]]:
    """Every model version, so a prediction can be traced to what produced it."""
    return [
        item.model_dump(mode="json")
        for item in _require_surrogate(search_id).read_models()
    ]


@app.get("/api/search/{search_id}/acquisition")
def get_acquisition_rounds(search_id: RunId) -> list[dict[str, object]]:
    """Predicted against actual, per round. The diagnostic that matters most."""
    return [
        item.model_dump(mode="json")
        for item in _require_surrogate(search_id).read_rounds()
    ]


@app.get("/api/search/{search_id}/surrogate/report")
def get_surrogate_report(search_id: RunId) -> dict[str, object]:
    report = _require_surrogate(search_id).read_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"search '{search_id}' has no surrogate report yet",
        )
    return report
