"""Persistent, reproducible run artifacts.

Layout of one run::

    artifacts/runs/<run_id>/
        manifest.json             # the full PredictionResult contract
        stimulus.json
        events.parquet            # backend event table, when produced
        events_summary.json
        predictions.npy           # [T, V] raw model output — IMMUTABLE
        temporal.npz              # per-row timing arrays
        prediction_metadata.json
        validation.json
        performance.json
        provenance.json
        diagnostics/

Source media is never copied. It is referenced by absolute path, filename and
SHA-256, which is enough to prove provenance without duplicating gigabytes.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import secrets
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np
from pydantic import BaseModel

from blackmirror.errors import ArtifactReadError, ArtifactWriteError
from blackmirror.schemas.prediction import ArtifactPaths, PredictionResult
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical filenames. Referenced by docs/data_contracts.md; changing one is a
# contract change.
MANIFEST_FILE = "manifest.json"
STIMULUS_FILE = "stimulus.json"
PREDICTIONS_FILE = "predictions.npy"
TEMPORAL_FILE = "temporal.npz"
PREDICTION_METADATA_FILE = "prediction_metadata.json"
VALIDATION_FILE = "validation.json"
PERFORMANCE_FILE = "performance.json"
PROVENANCE_FILE = "provenance.json"
EVENTS_FILE = "events.parquet"
EVENTS_SUMMARY_FILE = "events_summary.json"
DIAGNOSTICS_DIR = "diagnostics"


def generate_run_id() -> str:
    """A sortable, collision-resistant run identifier."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


@contextmanager
def _atomic_write(path: Path) -> Iterator[IO[bytes]]:
    """Yield a handle whose contents replace ``path`` only on clean completion.

    A crash mid-write leaves the original file intact and no partial artifact
    behind, which matters because a truncated predictions.npy would look like
    valid scientific data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed below, then renamed
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle as tmp:
            yield tmp
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace ``path`` with ``payload``."""
    with _atomic_write(path) as handle:
        handle.write(payload)


class ArtifactStore:
    """Reads and writes run directories under a root artifact directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.index_path = self.root / "index.jsonl"

    # --- Layout ----------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id) is None or run_id in {".", ".."}:
            raise ValueError("invalid run id")
        return self.runs_dir / run_id

    def create_run_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactWriteError(f"Run directory already exists: {path}") from exc
        except OSError as exc:
            raise ArtifactWriteError(f"Cannot create run directory {path}: {exc}") from exc
        return path

    def paths_for(self, run_id: str, *, has_events: bool, has_temporal: bool) -> ArtifactPaths:
        """Build the path contract for a run. Sub-paths are run-relative."""
        return ArtifactPaths(
            run_dir=self.run_dir(run_id),
            manifest=Path(MANIFEST_FILE),
            stimulus=Path(STIMULUS_FILE),
            predictions=Path(PREDICTIONS_FILE),
            prediction_metadata=Path(PREDICTION_METADATA_FILE),
            temporal=Path(TEMPORAL_FILE) if has_temporal else None,
            events=Path(EVENTS_FILE) if has_events else None,
            events_summary=Path(EVENTS_SUMMARY_FILE) if has_events else None,
            validation=Path(VALIDATION_FILE),
            performance=Path(PERFORMANCE_FILE),
            provenance=Path(PROVENANCE_FILE),
        )

    # --- Writers ---------------------------------------------------------

    def write_json(self, run_id: str, filename: str, payload: Any) -> Path:
        """Write a JSON artifact. Pydantic models are serialized via their own dump."""
        path = self.run_dir(run_id) / filename
        try:
            if isinstance(payload, BaseModel):
                text = payload.model_dump_json(indent=2)
            else:
                text = json.dumps(payload, indent=2, sort_keys=False, default=str)
            _atomic_write_bytes(path, text.encode("utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            raise ArtifactWriteError(f"Failed to write {filename} for run {run_id}: {exc}") from exc
        return path

    def write_predictions(self, run_id: str, array: np.ndarray) -> Path:
        """Persist the raw prediction matrix, byte-for-byte as the model returned it."""
        path = self.run_dir(run_id) / PREDICTIONS_FILE
        try:
            with _atomic_write(path) as handle:
                np.save(handle, array, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ArtifactWriteError(
                f"Failed to write predictions for run {run_id}: {exc}"
            ) from exc
        logger.info(
            "Wrote %s (%s, %.2f MB)",
            path.name,
            f"{tuple(array.shape)} {array.dtype}",
            path.stat().st_size / 1e6,
        )
        return path

    def write_arrays(self, run_id: str, filename: str, arrays: dict[str, np.ndarray]) -> Path:
        """Persist a set of related arrays as a compressed ``.npz``."""
        path = self.run_dir(run_id) / filename
        try:
            with _atomic_write(path) as handle:
                np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
        except (OSError, ValueError) as exc:
            raise ArtifactWriteError(f"Failed to write {filename} for run {run_id}: {exc}") from exc
        return path

    def write_events_table(self, run_id: str, table: Any) -> Path | None:
        """Persist the backend's event table as Parquet.

        Returns None (with a warning) if the table cannot be serialized — event
        metadata is valuable but never worth failing a completed inference over.
        """
        path = self.run_dir(run_id) / EVENTS_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
        except Exception as exc:
            logger.warning(
                "Could not persist events table as Parquet (%s: %s); "
                "events_summary.json still records the event structure.",
                type(exc).__name__,
                exc,
            )
            return None
        return path

    def append_index(self, entry: dict[str, Any]) -> None:
        """Append one line to the run index used for cache-key lookup."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            # Index loss degrades caching, it does not invalidate the run.
            logger.warning("Could not append to run index %s: %s", self.index_path, exc)

    # --- Readers ---------------------------------------------------------

    def read_manifest(self, run_id: str) -> PredictionResult:
        """Load a run's manifest back into the typed contract."""
        path = self.run_dir(run_id) / MANIFEST_FILE
        if not path.exists():
            raise ArtifactReadError(f"No manifest for run {run_id} at {path}")
        try:
            return PredictionResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactReadError(f"Manifest for run {run_id} is unreadable: {exc}") from exc

    def load_predictions(self, run_id: str) -> np.ndarray:
        """Load a run's raw prediction matrix."""
        path = self.run_dir(run_id) / PREDICTIONS_FILE
        if not path.exists():
            raise ArtifactReadError(f"No predictions for run {run_id} at {path}")
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ArtifactReadError(f"Predictions for run {run_id} are unreadable: {exc}") from exc

    def list_runs(self) -> list[str]:
        """Run ids with a manifest, newest first (run ids sort chronologically)."""
        if not self.runs_dir.exists():
            return []
        runs = [
            entry.name
            for entry in self.runs_dir.iterdir()
            if entry.is_dir() and (entry / MANIFEST_FILE).exists()
        ]
        return sorted(runs, reverse=True)

    def find_by_cache_key(self, cache_key: str) -> str | None:
        """Return the most recent completed run id for a cache key, if any.

        This is the whole of Phase 1 caching: an append-only JSONL index and a
        linear scan. It exists so the seam is real, not so it scales — a proper
        cache slots in behind the same method signature later.
        """
        if not self.index_path.exists():
            return None
        match: str | None = None
        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("cache_key") == cache_key and entry.get("status") != "failed":
                        match = entry.get("run_id")
        except OSError as exc:
            logger.warning("Could not read run index: %s", exc)
            return None

        if match and (self.run_dir(match) / MANIFEST_FILE).exists():
            return match
        return None
