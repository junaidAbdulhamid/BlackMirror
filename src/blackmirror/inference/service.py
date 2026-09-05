"""The inference service — the orchestration layer and the public API.

This is the module the rest of BlackMirror talks to. It imports **no** model
dependency: it holds a :class:`CorticalPredictorBackend` and knows only the
protocol. A future FastAPI route calls the same object the CLI calls.

    service = InferenceService(settings)
    result = service.predict("examples/sample.mp4")

Pipeline::

    validate -> preprocess -> load/reuse model -> infer
             -> postprocess -> validate predictions -> persist -> PredictionResult

Model lifecycle is independent of any single stimulus: :meth:`predict_many`
processes a batch of variants against one loaded model, which is what Phase 5
needs to compare A/B/N candidates fairly.
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Iterable, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

from blackmirror.config.settings import BackendName, Settings, get_settings
from blackmirror.errors import (
    BackendUnavailableError,
    InferenceError,
    PostprocessingError,
    PredictionValidationError,
    PreprocessingError,
)
from blackmirror.inference.backend import CorticalPredictorBackend, PreparedStimulus, RawPrediction
from blackmirror.inference.postprocessing import (
    build_cortical_metadata,
    build_prediction_array_metadata,
    build_temporal_arrays,
    build_temporal_metadata,
)
from blackmirror.inference.validation import build_stimulus, validate_predictions
from blackmirror.schemas.metadata import ModelMetadata, RunProvenance
from blackmirror.schemas.prediction import (
    PerformanceMetrics,
    PredictionResult,
    PredictionValidation,
)
from blackmirror.schemas.stimulus import StimulusInput
from blackmirror.storage import artifact_store as store_files
from blackmirror.storage.artifact_store import ArtifactStore, generate_run_id
from blackmirror.utils.device import DeviceReport, resolve_device
from blackmirror.utils.hashing import build_cache_key
from blackmirror.utils.logging import configure_logging, get_logger
from blackmirror.utils.timing import (
    StageTimings,
    capture_peak_memory,
    reset_peak_memory,
)

logger = get_logger(__name__)

#: Replaces the standard notice on mock runs. Synthetic output must never carry
#: language that implies a brain prediction was made.
SYNTHETIC_NOTICE = (
    "SYNTHETIC TEST DATA. This run used the mock backend: the values are generated "
    "numbers for pipeline development, NOT predicted cortical responses and NOT "
    "related to any brain. Do not interpret, plot, or report them as results."
)


def _installed_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def build_backend(settings: Settings, device: str) -> CorticalPredictorBackend:
    """Instantiate the configured backend.

    The TRIBE import is local so that `blackmirror` remains importable — and the
    mock backend remains usable — in an environment with no ML stack.
    """
    if settings.backend is BackendName.MOCK:
        from blackmirror.inference.mock_backend import MockBackend

        return MockBackend(device=device, dtype=settings.dtype, seed=settings.random_seed)

    if settings.backend is BackendName.TRIBE_V2:
        from blackmirror.inference.tribe_backend import TribeV2Backend

        return TribeV2Backend(settings=settings, device=device)

    raise BackendUnavailableError(f"Unknown backend: {settings.backend}")


class InferenceService:
    """Orchestrates cortical response prediction and artifact persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: CorticalPredictorBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)

        self.device_report: DeviceReport = resolve_device(
            self.settings.device,
            backend=self.settings.backend.value,
            allow_cpu_fallback=self.settings.allow_cpu_fallback,
        )
        for note in self.device_report.notes:
            logger.info("%s", note)

        self._backend = backend or build_backend(self.settings, self.device_report.selected_device)
        self.store = ArtifactStore(self.settings.artifact_dir)
        self._model_load_seconds = 0.0

    # --- Model lifecycle -------------------------------------------------

    @property
    def backend(self) -> CorticalPredictorBackend:
        return self._backend

    def ensure_model_loaded(self) -> ModelMetadata:
        """Load the model once and reuse it for every subsequent stimulus."""
        if not self._backend.is_loaded():
            timings = StageTimings()
            with timings.stage("load"):
                self._backend.load()
            self._model_load_seconds = timings.get("load")
            logger.info("Model ready in %.2fs", self._model_load_seconds)
        return self._backend.get_model_metadata()

    def unload_model(self) -> None:
        self._backend.unload()

    def get_model_metadata(self) -> ModelMetadata:
        """Load if needed, then return provenance for the active model."""
        return self.ensure_model_loaded()

    # --- Public API ------------------------------------------------------

    def predict(self, content_path: str | Path) -> PredictionResult:
        """Run the full pipeline for one stimulus and return the standardized result."""
        stimulus = build_stimulus(
            content_path, supported_media_types=self._backend.supported_media_types()
        )
        return self.predict_stimulus(stimulus)

    def predict_many(self, content_paths: Iterable[str | Path]) -> list[PredictionResult]:
        """Run several stimuli against a single loaded model.

        The basis for Phase 5 variant comparison: every result shares one model
        instance, device and configuration, so their provenance records match
        and the comparison is fair.
        """
        self.ensure_model_loaded()
        return [self.predict(path) for path in content_paths]

    def predict_stimulus(self, stimulus: StimulusInput) -> PredictionResult:
        """Run the pipeline for an already-validated stimulus."""
        timings = StageTimings()
        run_id = generate_run_id()
        logger.info("Run %s | stimulus %s", run_id, stimulus.summary_line())

        model_metadata = self.ensure_model_loaded()
        cache_key = build_cache_key(
            stimulus_sha256=stimulus.sha256,
            model_fingerprint=model_metadata.fingerprint(),
            preprocessing_fingerprint=self.settings.preprocessing_fingerprint(),
        )

        if self.settings.reuse_cached_runs:
            cached = self.store.find_by_cache_key(cache_key)
            if cached is not None:
                logger.info("Reusing completed run %s (cache key %s)", cached, cache_key[:12])
                return self.store.read_manifest(cached)

        reset_peak_memory()

        prepared = self._preprocess(stimulus, timings)
        raw = self._infer(stimulus, prepared, timings)
        validation = self._validate(raw, timings)

        with timings.stage("postprocess"):
            try:
                temporal_arrays = build_temporal_arrays(raw)
            except Exception as exc:
                raise PostprocessingError(
                    f"Failed to derive temporal metadata for run {run_id}: {exc}"
                ) from exc

        result = self._persist(
            run_id=run_id,
            cache_key=cache_key,
            stimulus=stimulus,
            model_metadata=model_metadata,
            prepared=prepared,
            raw=raw,
            temporal_arrays=temporal_arrays,
            validation=validation,
            timings=timings,
        )

        for warning in validation.warnings:
            logger.warning("Validation: %s", warning)
        logger.info(
            "Validation %s | artifacts: %s",
            "passed" if validation.valid and not validation.warnings else "completed with warnings",
            result.artifacts.run_dir,
        )
        return result

    # --- Stages ----------------------------------------------------------

    def _preprocess(self, stimulus: StimulusInput, timings: StageTimings) -> PreparedStimulus:
        with timings.stage("preprocess"):
            try:
                prepared = self._backend.preprocess(stimulus)
            except Exception as exc:
                raise PreprocessingError(
                    f"Preprocessing failed for stimulus {stimulus.short_sha} "
                    f"({stimulus.filename}): {type(exc).__name__}: {exc}"
                ) from exc
        n_events = prepared.event_summary.get("n_events", "?")
        logger.info("Generated %s events in %.2fs", n_events, timings.get("preprocess"))
        return prepared

    def _infer(
        self, stimulus: StimulusInput, prepared: PreparedStimulus, timings: StageTimings
    ) -> RawPrediction:
        logger.info("Starting inference on %s", self.device_report.selected_device)
        with timings.stage("inference"):
            try:
                raw = self._backend.infer(prepared)
            except Exception as exc:
                raise InferenceError(
                    f"Inference failed for stimulus {stimulus.short_sha} "
                    f"({stimulus.filename}) on {self.device_report.selected_device}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        logger.info(
            "Prediction shape: %s dtype=%s (%.2fs)",
            tuple(raw.array.shape),
            raw.array.dtype,
            timings.get("inference"),
        )

        if len(raw.segments) != raw.array.shape[0]:
            raise PostprocessingError(
                f"Backend returned {raw.array.shape[0]} prediction rows but "
                f"{len(raw.segments)} segment records. Temporal alignment cannot be "
                f"established, so the run is rejected rather than silently misaligned."
            )
        return raw

    def _validate(self, raw: RawPrediction, timings: StageTimings) -> PredictionValidation:
        expected = raw.vertices_per_hemisphere * len(raw.hemisphere_order)
        with timings.stage("validation"):
            validation = validate_predictions(raw.array, expected_vertex_count=expected)
        if not validation.valid:
            raise PredictionValidationError(
                "Predictions are structurally unusable: " + "; ".join(validation.errors)
            )
        return validation

    # --- Persistence -----------------------------------------------------

    def _persist(
        self,
        *,
        run_id: str,
        cache_key: str,
        stimulus: StimulusInput,
        model_metadata: ModelMetadata,
        prepared: PreparedStimulus,
        raw: RawPrediction,
        temporal_arrays: dict[str, np.ndarray],
        validation: PredictionValidation,
        timings: StageTimings,
    ) -> PredictionResult:
        persistence_started = time.perf_counter()
        with timings.stage("persistence"):
            self.store.create_run_dir(run_id)

            self.store.write_predictions(run_id, raw.array)
            self.store.write_arrays(run_id, store_files.TEMPORAL_FILE, temporal_arrays)

            events_path = None
            if prepared.events_table is not None:
                events_path = self.store.write_events_table(run_id, prepared.events_table)
            self.store.write_json(
                run_id, store_files.EVENTS_SUMMARY_FILE, prepared.event_summary
            )

            mesh_dir = self.settings.mesh_dir / raw.surface_space
            prediction_meta = build_prediction_array_metadata(
                raw, artifact_path=Path(store_files.PREDICTIONS_FILE)
            )
            temporal_meta = build_temporal_metadata(
                raw,
                arrays=temporal_arrays,
                event_summary=prepared.event_summary,
                arrays_artifact_path=Path(store_files.TEMPORAL_FILE),
            )
            cortical_meta = build_cortical_metadata(
                raw, mesh_artifact_dir=mesh_dir if mesh_dir.exists() else None
            )

            provenance = RunProvenance(
                run_id=run_id,
                blackmirror_version=_installed_version("blackmirror") or "0.1.0",
                python_version=sys.version.split()[0],
                platform=platform.platform(),
                torch_version=self.device_report.torch_version,
                numpy_version=np.__version__,
                backend_package_version=self._backend_package_version(),
                device=self.device_report.selected_device,
                dtype=self.settings.dtype,
                random_seed=self.settings.random_seed,
                stimulus_sha256=stimulus.sha256,
                model_fingerprint=model_metadata.fingerprint(),
                preprocessing_config=self.settings.preprocessing_fingerprint(),
                cache_key=cache_key,
                applied_compat_patches=tuple(raw.extra.get("applied_compat_patches", ())),
            )

            memory = capture_peak_memory()
            performance = PerformanceMetrics(
                preprocessing_seconds=timings.get("preprocess"),
                model_load_seconds=self._model_load_seconds,
                inference_seconds=timings.get("inference"),
                postprocessing_seconds=timings.get("postprocess"),
                validation_seconds=timings.get("validation"),
                # Measured explicitly: these metrics are built *inside* the
                # persistence stage, so the stage timer has not recorded yet.
                # This covers the array writes, which dominate.
                persistence_seconds=time.perf_counter() - persistence_started,
                total_seconds=timings.total_seconds,
                peak_gpu_allocated_bytes=memory.peak_gpu_allocated_bytes,
                peak_gpu_reserved_bytes=memory.peak_gpu_reserved_bytes,
                peak_host_rss_bytes=memory.peak_host_rss_bytes,
                realtime_factor=(
                    timings.total_seconds / stimulus.duration_seconds
                    if stimulus.duration_seconds
                    else None
                ),
            )

            paths = self.store.paths_for(
                run_id, has_events=events_path is not None, has_temporal=True
            )
            notice = (
                SYNTHETIC_NOTICE if model_metadata.is_synthetic else PredictionResult.model_fields[
                    "interpretation_notice"
                ].default
            )
            result = PredictionResult(
                interpretation_notice=notice,
                run_id=run_id,
                status="completed_with_warnings" if validation.warnings else "completed",
                stimulus=stimulus,
                model=model_metadata,
                prediction=prediction_meta,
                temporal=temporal_meta,
                cortical=cortical_meta,
                validation=validation,
                performance=performance,
                provenance=provenance,
                artifacts=paths,
            )

            self.store.write_json(run_id, store_files.STIMULUS_FILE, stimulus)
            self.store.write_json(run_id, store_files.PREDICTION_METADATA_FILE, prediction_meta)
            self.store.write_json(run_id, store_files.VALIDATION_FILE, validation)
            self.store.write_json(run_id, store_files.PERFORMANCE_FILE, performance)
            self.store.write_json(run_id, store_files.PROVENANCE_FILE, provenance)
            self.store.write_json(run_id, store_files.MANIFEST_FILE, result)

            self.store.append_index(
                {
                    "run_id": run_id,
                    "cache_key": cache_key,
                    "stimulus_sha256": stimulus.sha256,
                    "stimulus_filename": stimulus.filename,
                    "backend": model_metadata.backend,
                    "model_fingerprint": model_metadata.fingerprint(),
                    "is_synthetic": model_metadata.is_synthetic,
                    "status": result.status,
                    "created_at": result.created_at.isoformat(),
                }
            )

        # Note: performance metrics are snapshotted before the JSON files are
        # written, so persistence_seconds excludes the manifest write itself.
        # The gap is milliseconds and the alternative (rewriting the manifest to
        # describe its own write) is circular.
        return result

    def _backend_package_version(self) -> str | None:
        if self.settings.backend is BackendName.TRIBE_V2:
            return _installed_version("tribev2")
        return None


def summarize_runs(store: ArtifactStore, run_ids: Sequence[str]) -> list[PredictionResult]:
    """Load several manifests. Convenience for CLI/reporting."""
    return [store.read_manifest(run_id) for run_id in run_ids]
