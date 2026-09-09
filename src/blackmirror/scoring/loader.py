"""Building a ScoringVariant from persisted Phase 1/3/4 artifacts.

WHY THIS EXISTS
    The scoring engine is deliberately ignorant of storage: it takes arrays. But
    assembling those arrays correctly is fiddly and easy to get subtly wrong --
    which time axis, which mask, which region ordering -- so it is done once here
    rather than at every call site.

THE NETWORK CASE
    Phase 3 persists no functional-network series. The verified Yeo 2011 mapping
    lives in Phase 5 (`comparison.network`), where it is checksum-pinned and
    required to match this project's mesh coordinates and medial wall exactly.
    Network objectives therefore aggregate here, at scoring time, using that same
    verified mapping -- never a re-derived or approximate one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackmirror.scoring.evaluator import ScoringVariant
from blackmirror.scoring.windows import ContentEventInterval


class VariantLoadError(ValueError):
    """A run cannot be assembled into a scorable variant."""


def load_variant(
    artifact_root: Path,
    run_id: str,
    *,
    with_networks: bool = False,
    variant_id: str | None = None,
) -> ScoringVariant:
    """Assemble one completed run into a scorable variant."""
    from blackmirror.scoring.storage import validate_id

    validate_id(run_id, "run")
    root = Path(artifact_root) / "runs" / run_id
    if not root.is_dir():
        raise VariantLoadError(f"no run directory for {run_id}")

    manifest = _json(root / "manifest.json", run_id, "manifest")
    analytics = _json(root / "analytics" / "metadata.json", run_id, "analytics")

    try:
        series = np.load(root / "analytics" / "timeseries.npz", allow_pickle=False)
        mapping = np.load(root / "analytics" / "atlas_mapping.npz", allow_pickle=False)
    except OSError as exc:
        raise VariantLoadError(
            f"run {run_id} has no analytics arrays; run `blackmirror analyze {run_id}` first"
        ) from exc

    times = np.asarray(series["times"], dtype=np.float64)
    roi = np.asarray(series["roi_timeseries"], dtype=np.float64)
    wall = np.asarray(mapping["medial_wall_mask"], dtype=bool)
    vertex_to_region = (
        np.asarray(mapping["vertex_to_region"], dtype=np.int64)
        if "vertex_to_region" in mapping
        else None
    )
    responses = np.load(root / "predictions.npy", allow_pickle=False)

    if responses.ndim != 2 or times.ndim != 1 or roi.ndim != 2:
        raise VariantLoadError(f"run {run_id} scoring arrays have invalid dimensions")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise VariantLoadError(f"run {run_id} timestamps must be finite and strictly increasing")
    if responses.shape[0] != times.size:
        raise VariantLoadError(
            f"run {run_id} has {responses.shape[0]} prediction rows but {times.size} "
            f"analytics timestamps; the analytics are stale relative to the prediction"
        )
    if roi.shape[0] != times.size:
        raise VariantLoadError(f"run {run_id} ROI series does not match its time axis")

    regions = analytics["atlas"]["regions"]
    if [region["region_id"] for region in regions] != list(range(len(regions))):
        raise VariantLoadError(
            f"run {run_id} atlas region ids are not a dense 0..n-1 sequence, so a "
            f"region_id cannot be used as a column index"
        )
    if roi.shape[1] != len(regions):
        raise VariantLoadError(f"run {run_id} ROI columns do not match atlas regions")
    declared_vertices = int(manifest["cortical"]["vertex_count"])
    if responses.shape[1] != declared_vertices or wall.shape != (declared_vertices,):
        raise VariantLoadError(f"run {run_id} cortical arrays do not match declared vertices")

    networks: np.ndarray | None = None
    network_names: tuple[str, ...] = ()
    network_hash: str | None = None
    if with_networks:
        networks, network_names, network_hash = _aggregate_networks(
            Path(artifact_root), responses
        )

    return ScoringVariant(
        variant_id=variant_id or run_id,
        responses=responses,
        times=times,
        roi_timeseries=roi,
        region_names=tuple(region["name"] for region in regions),
        medial_wall_mask=wall,
        vertex_to_region=vertex_to_region,
        duration_seconds=float(manifest["stimulus"]["duration_seconds"]),
        hemisphere_ranges={
            name: (int(bounds[0]), int(bounds[1]))
            for name, bounds in manifest["cortical"]["hemisphere_index_ranges"].items()
        },
        atlas_name=analytics["atlas"]["name"],
        atlas_version=analytics["atlas"]["version"],
        network_timeseries=networks,
        network_names=network_names,
        network_mapping_sha256=network_hash,
        analytics_version=str(analytics["metadata"]["analytics_version"]),
        model_fingerprint=str(manifest["provenance"]["model_fingerprint"]),
        content_events=_content_events(root),
    )


def _aggregate_networks(
    artifact_root: Path, responses: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """Aggregate to Yeo networks using Phase 5's verified mapping only."""
    from blackmirror.comparison.network import load_yeo7_mapping

    try:
        mapping = load_yeo7_mapping(
            artifact_root / "atlases" / "yeo2011_fsaverage5",
            artifact_root / "mesh" / "fsaverage5",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise VariantLoadError(
            f"the verified Yeo mapping could not be loaded, so network objectives are "
            f"unavailable: {exc}"
        ) from exc
    return mapping.aggregate(responses), mapping.network_names, mapping.mapping_sha256


def _content_events(root: Path) -> list[ContentEventInterval] | None:
    """Phase 4 events, or None when the run has no content analysis."""
    path = root / "content_analysis" / "metadata.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return [
        ContentEventInterval(
            event_type=str(event["event_type"]),
            start_time=float(event["start_time"]),
            end_time=float(event["end_time"]),
        )
        for event in payload.get("events", [])
    ]


def _json(path: Path, run_id: str, label: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantLoadError(f"run {run_id} has no readable {label}") from exc
