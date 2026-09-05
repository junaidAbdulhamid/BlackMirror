"""Visualization API.

A thin read-only HTTP surface over Phase 1 artifacts. It runs **no inference**:
every response is assembled from files a completed run already wrote. Rotating
the brain or dragging the timeline must never touch a model.

Transport strategy — arrays are served as raw little-endian binary, not JSON.
A prediction matrix is `T x 20484 float32`; JSON would inflate that roughly 8x
and force the browser to parse millions of numbers. Binary lands directly in a
`Float32Array` and can be uploaded to the GPU without a copy. Metadata stays
JSON, where readability matters and size does not.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
from blackmirror.content.schemas import ContentAnalysisResult
from blackmirror.errors import ArtifactReadError
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
    logger.info("Visualization API ready (read-only; no inference)")
    yield


app = FastAPI(
    title="NeuroSplit Visualization API",
    version="0.2.0",
    description=(
        "Read-only access to completed TRIBE v2 inference runs for cortical "
        "visualization. Serves predicted cortical responses, never live inference."
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
    return {"status": "ok", "service": "neurosplit-visualization", "runs_inference": False}


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
