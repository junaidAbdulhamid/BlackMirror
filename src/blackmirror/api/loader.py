"""Visualization data loader.

Sits between Phase 1's artifact layer and the HTTP layer. It reuses
:class:`ArtifactStore` and :class:`PredictionResult` — prediction loading is
never reimplemented — and adds only what a renderer needs:

* a JSON-safe summary of one run
* validated binary buffers for predictions, mesh geometry and timing
* an alignment check that refuses to serve mismatched prediction/mesh pairs

That last point is the reason this module exists rather than the route handlers
reading files directly: a scientifically wrong render is worse than an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from blackmirror.api.contracts import (
    ArrayDescriptor,
    CorticalSummary,
    HemisphereRange,
    ModelSummary,
    ResponseScale,
    RunListItem,
    RunSummary,
    StimulusSummary,
    TemporalSummary,
)
from blackmirror.config.settings import Settings
from blackmirror.errors import ArtifactReadError, BlackMirrorError
from blackmirror.schemas.prediction import PredictionResult
from blackmirror.storage.artifact_store import ArtifactStore
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

Surface = Literal["inflated", "pial"]
Hemi = Literal["left", "right"]


class VisualizationError(BlackMirrorError):
    """A run exists but cannot be visualized safely."""


class MeshAlignmentError(VisualizationError):
    """Prediction vertex count does not match the cortical mesh.

    Raised instead of rendering. A misaligned map would paint real predicted
    values onto the wrong anatomy while looking entirely plausible.
    """


@dataclass(frozen=True)
class BinaryArray:
    """A raw buffer plus the metadata a client needs to interpret it."""

    data: bytes
    dtype: str
    shape: tuple[int, ...]

    @property
    def descriptor_kwargs(self) -> dict[str, object]:
        return {"dtype": self.dtype, "shape": self.shape, "byte_length": len(self.data)}


class VisualizationLoader:
    """Reads Phase 1 artifacts and prepares them for a renderer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ArtifactStore(settings.artifact_dir)
        self.mesh_root = settings.mesh_dir

    # --- Runs ------------------------------------------------------------

    def list_runs(self, limit: int = 50) -> list[RunListItem]:
        items: list[RunListItem] = []
        for run_id in self.store.list_runs()[:limit]:
            try:
                result = self.store.read_manifest(run_id)
            except ArtifactReadError:
                logger.warning("Skipping run %s: unreadable manifest", run_id)
                continue
            items.append(
                RunListItem(
                    run_id=result.run_id,
                    created_at=result.created_at.isoformat(),
                    stimulus_filename=result.stimulus.filename,
                    media_type=result.stimulus.media_type.value,
                    backend=result.model.backend,
                    is_synthetic=result.model.is_synthetic,
                    shape=result.prediction.shape,
                    status=result.status,
                )
            )
        return items

    def get_result(self, run_id: str) -> PredictionResult:
        """Load the Phase 1 contract for a run."""
        return self.store.read_manifest(run_id)

    # --- Summary ---------------------------------------------------------

    def build_summary(self, run_id: str) -> RunSummary:
        result = self.get_result(run_id)
        predictions = self.load_predictions_array(run_id)
        self._assert_alignment(result, predictions)

        scale = self._build_scale(result, predictions)
        base = f"/api/runs/{run_id}"
        # Every mesh lookup below uses the space this run actually declares.
        # Defaulting to fsaverage5 here would silently describe a different
        # surface than the one the predictions live on.
        space = result.cortical.surface_space

        arrays: dict[str, ArrayDescriptor] = {
            "predictions": ArrayDescriptor(
                url=f"{base}/predictions",
                dtype="float32",
                shape=tuple(int(d) for d in predictions.shape),
                byte_length=predictions.astype(np.float32).nbytes,
                description=(
                    "Raw predicted cortical response, [time, vertex], row-major. "
                    "Exactly the Phase 1 matrix; no normalization applied."
                ),
            ),
            "timeline": ArrayDescriptor(
                url=f"{base}/timeline",
                dtype="float32",
                shape=(int(result.prediction.temporal_samples),),
                byte_length=int(result.prediction.temporal_samples) * 4,
                description=(
                    "stimulus_time_seconds — the playback clock for each prediction "
                    "row. Use this, never row_index * TR."
                ),
            ),
            "durations": ArrayDescriptor(
                url=f"{base}/durations",
                dtype="float32",
                shape=(int(result.prediction.temporal_samples),),
                byte_length=int(result.prediction.temporal_samples) * 4,
                description=(
                    "segment_duration_seconds — how much stimulus time each row "
                    "covers. Row i covers [timeline[i], timeline[i] + durations[i]). "
                    "Needed to tell genuine temporal coverage from a nearby row."
                ),
            ),
        }
        for hemi in ("left", "right"):
            for surface in ("inflated", "pial"):
                vertices = self.load_mesh_array(hemi, f"{surface}_vertices", space)
                arrays[f"{hemi}_{surface}_vertices"] = ArrayDescriptor(
                    url=f"{base}/mesh/{hemi}/{surface}/vertices",
                    description=(f"{hemi} hemisphere {surface} vertex coordinates (mm), {space}"),
                    **vertices.descriptor_kwargs,  # type: ignore[arg-type]
                )
            faces = self.load_mesh_array(hemi, "faces", space)
            arrays[f"{hemi}_faces"] = ArrayDescriptor(
                url=f"{base}/mesh/{hemi}/faces",
                description=(
                    f"{hemi} hemisphere triangles ({space}); hemisphere-local "
                    f"0-based vertex indices"
                ),
                **faces.descriptor_kwargs,  # type: ignore[arg-type]
            )
        if self._medial_wall_path(space).exists():
            wall = self.load_medial_wall(space)
            arrays["medial_wall"] = ArrayDescriptor(
                url=f"{base}/mesh/medial-wall",
                description=(
                    "Boolean mask over the full vertex axis; 1 = medial wall, where "
                    "fMRI signal is not meaningful."
                ),
                **wall.descriptor_kwargs,  # type: ignore[arg-type]
            )

        return RunSummary(
            run_id=result.run_id,
            created_at=result.created_at.isoformat(),
            status=result.status,
            schema_version=result.schema_version,
            stimulus=StimulusSummary(
                filename=result.stimulus.filename,
                media_type=result.stimulus.media_type.value,
                duration_seconds=result.stimulus.duration_seconds,
                sha256=result.stimulus.sha256,
                playable=result.stimulus.path.exists(),
            ),
            model=ModelSummary(
                name=result.model.name,
                backend=result.model.backend,
                model_id=result.model.model_id,
                license=result.model.license,
                device=result.model.loaded_device,
                dtype=result.model.dtype,
                subject_conditioning=result.model.subject_conditioning,
                is_synthetic=result.model.is_synthetic,
                feature_extractors=dict(result.model.feature_extractors),
            ),
            temporal=TemporalSummary(
                n_time_points=result.temporal.n_time_points,
                tr_seconds=result.temporal.tr_seconds,
                timeline_is_contiguous=result.temporal.timeline_is_contiguous,
                segments_were_filtered=result.temporal.segments_were_filtered,
                n_segments_total=result.temporal.n_segments_total,
                n_segments_kept=result.temporal.n_segments_kept,
                first_segment_start_seconds=result.temporal.first_segment_start_seconds,
                last_segment_end_seconds=result.temporal.last_segment_end_seconds,
                covered_seconds=result.temporal.covered_seconds,
                hemodynamic_offset_seconds=result.temporal.hemodynamic_offset_seconds,
                hemodynamic_offset_verified=result.temporal.hemodynamic_offset_verified,
                output_is_stimulus_aligned=result.temporal.output_is_stimulus_aligned,
                notes=result.temporal.notes,
            ),
            cortical=CorticalSummary(
                surface_space=result.cortical.surface_space,
                vertex_count=result.cortical.vertex_count,
                vertices_per_hemisphere=result.cortical.vertices_per_hemisphere,
                hemispheres=tuple(
                    HemisphereRange(name=name, start=rng[0], end=rng[1])
                    for name, rng in result.cortical.hemisphere_index_ranges.items()
                ),
                medial_wall_available=self._medial_wall_path(space).exists(),
                mapping_notes=result.cortical.mapping_notes,
            ),
            scale=scale,
            arrays=arrays,
            interpretation_notice=result.interpretation_notice,
        )

    @staticmethod
    def _build_scale(result: PredictionResult, predictions: np.ndarray) -> ResponseScale:
        """Derive colour-scale statistics from Phase 1 validation where possible.

        Phase 1 already computed robust percentiles over the finite values; we
        reuse them rather than recomputing a subtly different statistic.
        """
        validation = result.validation
        finite = predictions[np.isfinite(predictions)]

        def pick(stored: float | None, fallback: float) -> float:
            return float(stored) if stored is not None else float(fallback)

        p01 = pick(validation.p01, np.percentile(finite, 1) if finite.size else 0.0)
        p99 = pick(validation.p99, np.percentile(finite, 99) if finite.size else 0.0)
        negative_fraction = float((finite < 0).mean()) if finite.size else 0.0
        abs_max = float(max(abs(p01), abs(p99))) or 1.0

        return ResponseScale(
            min=pick(validation.min, finite.min() if finite.size else 0.0),
            max=pick(validation.max, finite.max() if finite.size else 0.0),
            mean=pick(validation.mean, finite.mean() if finite.size else 0.0),
            std=pick(validation.std, finite.std() if finite.size else 0.0),
            p01=p01,
            p50=pick(validation.p50, np.percentile(finite, 50) if finite.size else 0.0),
            p99=p99,
            abs_max=abs_max,
            negative_fraction=negative_fraction,
            # Predictions straddling zero are only interpretable on a scale that
            # shows the sign; a sequential ramp would hide it.
            diverging_recommended=negative_fraction > 0.15,
        )

    # --- Alignment -------------------------------------------------------

    def _assert_alignment(self, result: PredictionResult, predictions: np.ndarray) -> None:
        """Refuse to serve a run whose predictions cannot map onto the mesh."""
        if predictions.ndim != 2:
            raise MeshAlignmentError(
                f"Run {result.run_id}: predictions are {predictions.ndim}-D, expected "
                f"2-D [time, vertex]."
            )

        declared = result.cortical.vertex_count
        actual = int(predictions.shape[1])
        if actual != declared:
            raise MeshAlignmentError(
                f"Run {result.run_id}: prediction has {actual} vertices but the manifest "
                f"declares {declared}."
            )

        per_hemisphere = result.cortical.vertices_per_hemisphere
        for hemi in ("left", "right"):
            path = self.mesh_root / result.cortical.surface_space / f"{hemi}_inflated_vertices.npy"
            if not path.exists():
                raise MeshAlignmentError(
                    f"Run {result.run_id}: cortical mesh for "
                    f"'{result.cortical.surface_space}' is missing ({path}). "
                    f"Export it with `blackmirror export-mesh`."
                )
            mesh_vertices = int(np.load(path, mmap_mode="r").shape[0])
            if mesh_vertices != per_hemisphere:
                raise MeshAlignmentError(
                    f"Run {result.run_id}: {hemi} mesh has {mesh_vertices} vertices but the "
                    f"prediction expects {per_hemisphere} per hemisphere. Refusing to "
                    f"render misaligned data."
                )

        covered = sum(r[1] - r[0] for r in result.cortical.hemisphere_index_ranges.values())
        if covered != actual:
            raise MeshAlignmentError(
                f"Run {result.run_id}: hemisphere ranges cover {covered} columns but the "
                f"prediction has {actual}. Some columns would be unmapped."
            )

    # --- Binary arrays ---------------------------------------------------

    def load_predictions_array(self, run_id: str) -> np.ndarray:
        return self.store.load_predictions(run_id)

    def load_predictions(self, run_id: str) -> BinaryArray:
        """Raw [T, V] float32, row-major. Unmodified Phase 1 output."""
        array = np.ascontiguousarray(self.load_predictions_array(run_id), dtype=np.float32)
        return BinaryArray(array.tobytes(), "float32", tuple(int(d) for d in array.shape))

    def load_timeline(self, run_id: str) -> BinaryArray:
        """Per-row playback clock, in seconds.

        Read from temporal.npz. Falls back to `index * TR` only when the file is
        absent, and says so loudly — that fallback is wrong whenever segments
        were filtered.
        """
        result = self.get_result(run_id)
        run_dir = self.store.run_dir(run_id)
        path = (
            run_dir / result.temporal.arrays_artifact_path
            if result.temporal.arrays_artifact_path
            else None
        )
        if path is not None and path.exists():
            with np.load(path) as bundle:
                if "stimulus_time_seconds" in bundle:
                    times = np.asarray(bundle["stimulus_time_seconds"], dtype=np.float32)
                    return BinaryArray(
                        np.ascontiguousarray(times).tobytes(), "float32", (int(times.size),)
                    )

        logger.warning(
            "Run %s has no temporal.npz; falling back to index * TR. This is WRONG "
            "whenever event-free segments were dropped.",
            run_id,
        )
        times = (
            np.arange(result.prediction.temporal_samples, dtype=np.float32)
            * result.temporal.tr_seconds
        )
        return BinaryArray(times.tobytes(), "float32", (int(times.size),))

    def load_durations(self, run_id: str) -> BinaryArray:
        """Per-row segment durations, in seconds.

        Together with the timeline these give each row's true coverage interval
        ``[start, start + duration)``. Without durations a client can only ask
        "which row is nearest", which is a different and weaker question.

        Falls back to the TR when the array is absent, which is correct whenever
        segments are uniform and is the value Phase 1 uses to build them.
        """
        result = self.get_result(run_id)
        run_dir = self.store.run_dir(run_id)
        path = (
            run_dir / result.temporal.arrays_artifact_path
            if result.temporal.arrays_artifact_path
            else None
        )
        if path is not None and path.exists():
            with np.load(path) as bundle:
                if "segment_duration_seconds" in bundle:
                    values = np.asarray(bundle["segment_duration_seconds"], dtype=np.float32)
                    return BinaryArray(
                        np.ascontiguousarray(values).tobytes(),
                        "float32",
                        (int(values.size),),
                    )

        logger.warning(
            "Run %s has no segment_duration_seconds; assuming every row covers one TR.",
            run_id,
        )
        values = np.full(
            result.prediction.temporal_samples,
            result.temporal.tr_seconds,
            dtype=np.float32,
        )
        return BinaryArray(values.tobytes(), "float32", (int(values.size),))

    def load_segment_starts(self, run_id: str) -> np.ndarray | None:
        """Untransformed segment start times, for developer validation."""
        result = self.get_result(run_id)
        if not result.temporal.arrays_artifact_path:
            return None
        path = self.store.run_dir(run_id) / result.temporal.arrays_artifact_path
        if not path.exists():
            return None
        with np.load(path) as bundle:
            if "segment_start_seconds" not in bundle:
                return None
            return np.asarray(bundle["segment_start_seconds"], dtype=np.float64)

    # --- Mesh ------------------------------------------------------------

    def _mesh_dir(self, space: str = "fsaverage5") -> Path:
        return self.mesh_root / space

    def _medial_wall_path(self, space: str = "fsaverage5") -> Path:
        return self._mesh_dir(space) / "medial_wall_mask.npy"

    def load_mesh_array(self, hemisphere: str, name: str, space: str = "fsaverage5") -> BinaryArray:
        """Load one mesh array as a typed binary buffer."""
        path = self._mesh_dir(space) / f"{hemisphere}_{name}.npy"
        if not path.exists():
            raise MeshAlignmentError(
                f"Mesh array not found: {path}. Run `blackmirror export-mesh --space {space}`."
            )
        array = np.load(path)
        # int32 indices and float32 coordinates are what WebGL wants directly.
        dtype = "int32" if np.issubdtype(array.dtype, np.integer) else "float32"
        array = np.ascontiguousarray(array, dtype=np.int32 if dtype == "int32" else np.float32)
        return BinaryArray(array.tobytes(), dtype, tuple(int(d) for d in array.shape))

    def load_medial_wall(self, space: str = "fsaverage5") -> BinaryArray:
        path = self._medial_wall_path(space)
        if not path.exists():
            raise MeshAlignmentError(f"Medial wall mask not found: {path}")
        array = np.ascontiguousarray(np.load(path).astype(np.uint8))
        return BinaryArray(array.tobytes(), "uint8", (int(array.size),))

    def mesh_manifest(self, space: str = "fsaverage5") -> dict[str, object]:
        path = self._mesh_dir(space) / "mapping.json"
        if not path.exists():
            raise MeshAlignmentError(
                f"Mesh manifest not found: {path}. Run `blackmirror export-mesh`."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    # --- Stimulus --------------------------------------------------------

    def stimulus_path(self, run_id: str) -> Path:
        result = self.get_result(run_id)
        path = result.stimulus.path
        if not path.exists():
            raise VisualizationError(
                f"Run {run_id}: stimulus file is no longer at {path}. Phase 1 references "
                f"media by path and hash rather than copying it."
            )
        return path


@lru_cache(maxsize=1)
def get_loader() -> VisualizationLoader:
    """Process-wide loader. Cheap to build; cached so settings are read once."""
    from blackmirror.config.settings import get_settings

    return VisualizationLoader(get_settings())
