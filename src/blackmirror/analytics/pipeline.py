"""Run-level adapter from persisted Phase 1 artifacts to Phase 3 analytics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blackmirror.analytics.atlas import DestrieuxAtlas, validate_atlas_mapping
from blackmirror.analytics.engine import NeuralAnalyticsEngine
from blackmirror.analytics.schemas import NeuralAnalyticsResult
from blackmirror.analytics.storage import AnalyticsStore
from blackmirror.storage.artifact_store import ArtifactStore


def analyze_run(
    artifact_root: Path,
    run_id: str,
    *,
    atlas_data_dir: Path | None = None,
) -> NeuralAnalyticsResult:
    """Derive and persist analytics for a completed run."""
    raw_store = ArtifactStore(artifact_root)
    result = raw_store.read_manifest(run_id)
    responses = raw_store.load_predictions(run_id)
    if result.temporal.arrays_artifact_path is None:
        raise ValueError("run has no temporal arrays")
    temporal_path = raw_store.run_dir(run_id) / result.temporal.arrays_artifact_path
    with np.load(temporal_path, allow_pickle=False) as temporal:
        times = np.asarray(temporal["stimulus_time_seconds"], dtype=np.float64)

    mesh_dir = artifact_root / "mesh" / result.cortical.surface_space
    atlas = DestrieuxAtlas(data_dir=atlas_data_dir).load(mesh_dir)
    validation = validate_atlas_mapping(
        atlas, prediction_vertices=responses.shape[1], mesh_dir=mesh_dir
    )
    analytics, arrays = NeuralAnalyticsEngine().analyze(
        run_id=run_id,
        responses=responses,
        times=times,
        atlas=atlas,
        validation=validation,
        hemisphere_ranges=result.cortical.hemisphere_index_ranges,
    )
    AnalyticsStore(artifact_root).write(analytics, arrays, atlas)
    return analytics
