"""Content analysis persistence.

Mirrors Phase 1/3 conventions exactly: JSON holds metadata a human can read,
NPZ holds dense arrays, and the source media is referenced rather than copied.

    artifacts/runs/<run_id>/content_analysis/
        metadata.json      the full ContentAnalysisResult
        features.npz       feature matrix + every dense signal
        keyframes/         extracted representative frames
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import IO

import numpy as np

from blackmirror.content.schemas import ContentAnalysisResult
from blackmirror.errors import ArtifactReadError, ArtifactWriteError
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

CONTENT_DIR = "content_analysis"
METADATA_FILE = "metadata.json"
FEATURES_FILE = "features.npz"
KEYFRAME_DIR = "keyframes"


class ContentStore:
    """Reads and writes a run's content analysis."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)

    def directory(self, run_id: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id) is None or run_id in {".", ".."}:
            raise ValueError("invalid run id")
        return self.artifact_root / "runs" / run_id / CONTENT_DIR

    def keyframe_directory(self, run_id: str) -> Path:
        return self.directory(run_id) / KEYFRAME_DIR

    def metadata_path(self, run_id: str) -> Path:
        return self.directory(run_id) / METADATA_FILE

    def features_path(self, run_id: str, filename: str = FEATURES_FILE) -> Path:
        if Path(filename).name != filename:
            raise ArtifactReadError(
                "Content array path must be a filename within the run directory"
            )
        return self.directory(run_id) / filename

    def exists(self, run_id: str) -> bool:
        return self.metadata_path(run_id).exists()

    def write(
        self,
        result: ContentAnalysisResult,
        arrays: dict[str, np.ndarray] | None = None,
    ) -> Path:
        directory = self.directory(result.run_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if arrays:
                filename = result.arrays.path if result.arrays is not None else FEATURES_FILE
                _atomic(
                    self.features_path(result.run_id, filename),
                    lambda handle: np.savez_compressed(handle, **arrays),  # type: ignore[arg-type]
                )
            _atomic(
                self.metadata_path(result.run_id),
                lambda handle: handle.write(result.model_dump_json(indent=2).encode("utf-8")),
            )
            # Only after the manifest naming the current array is durable: a
            # feature file is content-addressed, so re-analysing at a new
            # version writes a new name and leaves the old one behind. Pruning
            # before the manifest lands could delete the array a reader is
            # still being pointed at.
            self._prune_stale_features(result)
        except OSError as exc:
            raise ArtifactWriteError(
                f"Could not write content analysis for {result.run_id}: {exc}"
            ) from exc
        logger.info("Wrote content analysis to %s", directory)
        return directory

    def _prune_stale_features(self, result: ContentAnalysisResult) -> None:
        """Delete feature arrays this run's manifest no longer references.

        Without this a run directory accumulates one array per analysis. That
        is not merely wasted space: the files differ in *column count* across
        analysis versions, so a consumer that picks one by globbing rather than
        by reading the manifest silently pairs a current feature-name list with
        a stale matrix.
        """
        keep = result.arrays.path if result.arrays is not None else FEATURES_FILE
        keep_name = Path(keep).name
        directory = self.directory(result.run_id)
        for path in directory.glob("features-*.npz"):
            if path.name == keep_name:
                continue
            try:
                path.unlink()
                logger.info("Removed stale feature array %s", path.name)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning("Could not remove stale feature array %s: %s", path.name, exc)

    def read(self, run_id: str) -> ContentAnalysisResult:
        path = self.metadata_path(run_id)
        if not path.exists():
            raise ArtifactReadError(
                f"No content analysis for run {run_id}. Run `blackmirror analyze-content {run_id}`."
            )
        try:
            return ContentAnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactReadError(
                f"Content analysis for {run_id} is unreadable: {exc}"
            ) from exc

    def read_arrays(self, run_id: str) -> dict[str, np.ndarray]:
        result = self.read(run_id)
        filename = result.arrays.path if result.arrays is not None else FEATURES_FILE
        path = self.features_path(run_id, filename)
        if not path.exists():
            return {}
        try:
            with np.load(path) as bundle:
                return {key: np.asarray(bundle[key]) for key in bundle.files}
        except (OSError, ValueError) as exc:
            raise ArtifactReadError(f"Content features for {run_id} are unreadable: {exc}") from exc

    def cached_result(self, run_id: str, cache_key: str) -> ContentAnalysisResult | None:
        """Return a stored analysis only when its cache key matches.

        Content analysis is by far the most expensive part of Phase 4, so a
        page load must never trigger it. The key covers stimulus bytes, models,
        version and configuration — anything that would change the answer.
        """
        if not self.exists(run_id):
            return None
        try:
            existing = self.read(run_id)
        except ArtifactReadError:
            return None
        if existing.metadata.cache_key != cache_key:
            logger.info("Cached content analysis for %s is stale; recomputing.", run_id)
            return None
        if existing.metadata.failed_stages:
            logger.info("Cached content analysis for %s has failed stages; recomputing.", run_id)
            return None
        return existing


def _atomic(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Write via a temp file + rename so a crash never leaves a partial artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed below, then renamed
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
