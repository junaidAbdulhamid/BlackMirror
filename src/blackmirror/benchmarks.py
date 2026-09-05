"""Performance baselines.

Phase 1 collects a baseline so later optimization has something to beat, and so
we know whether a Phase 5 variant sweep is affordable before designing one.

This is assembled from a completed run's manifest plus the filesystem, so it
costs nothing extra and can be regenerated for any stored run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blackmirror.errors import ArtifactWriteError
from blackmirror.schemas.prediction import PredictionResult
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


def _directory_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def build_benchmark(result: PredictionResult) -> dict[str, Any]:
    """Assemble the benchmark record for one completed run."""
    run_dir = result.artifacts.run_dir
    predictions_path = run_dir / result.prediction.artifact_path

    performance = result.performance
    duration = result.stimulus.duration_seconds

    return {
        "run_id": result.run_id,
        "created_at": result.created_at.isoformat(),
        "stimulus": {
            "filename": result.stimulus.filename,
            "media_type": result.stimulus.media_type.value,
            "duration_seconds": duration,
            "file_size_bytes": result.stimulus.file_size_bytes,
            "sha256": result.stimulus.sha256,
        },
        "model": {
            "name": result.model.name,
            "backend": result.model.backend,
            "is_synthetic": result.model.is_synthetic,
            "device": result.model.loaded_device,
            "dtype": result.model.dtype,
            "parameter_count": result.model.parameter_count,
        },
        "environment": {
            "platform": result.provenance.platform,
            "python_version": result.provenance.python_version,
            "torch_version": result.provenance.torch_version,
            "backend_package_version": result.provenance.backend_package_version,
            "applied_compat_patches": list(result.provenance.applied_compat_patches),
        },
        "prediction": {
            "shape": list(result.prediction.shape),
            "dtype": result.prediction.dtype,
            "temporal_samples": result.prediction.temporal_samples,
            "cortical_features": result.prediction.cortical_features,
        },
        "timings_seconds": {
            "preprocessing": performance.preprocessing_seconds,
            "model_load": performance.model_load_seconds,
            "inference": performance.inference_seconds,
            "postprocessing": performance.postprocessing_seconds,
            "validation": performance.validation_seconds,
            "persistence": performance.persistence_seconds,
            "total": performance.total_seconds,
        },
        "throughput": {
            "realtime_factor": performance.realtime_factor,
            "seconds_of_compute_per_second_of_content": (
                performance.total_seconds / duration if duration else None
            ),
        },
        "memory": {
            "peak_gpu_allocated_bytes": performance.peak_gpu_allocated_bytes,
            "peak_gpu_reserved_bytes": performance.peak_gpu_reserved_bytes,
            "peak_host_rss_bytes": performance.peak_host_rss_bytes,
        },
        "artifact_sizes_bytes": {
            "predictions_npy": (
                predictions_path.stat().st_size if predictions_path.exists() else None
            ),
            "run_directory_total": _directory_size_bytes(run_dir) if run_dir.exists() else None,
        },
    }


def write_benchmark(result: PredictionResult, benchmarks_dir: Path) -> Path:
    """Persist a run's benchmark record. Returns the written path."""
    record = build_benchmark(result)
    try:
        benchmarks_dir.mkdir(parents=True, exist_ok=True)
        path = benchmarks_dir / f"{result.run_id}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError as exc:
        raise ArtifactWriteError(f"Could not write the benchmark record: {exc}") from exc
    logger.info("Wrote benchmark: %s", path)
    return path
