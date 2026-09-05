"""Atomic, versioned persistence for derived analytics beside immutable raw data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from blackmirror.analytics.atlas import AtlasMapping
from blackmirror.analytics.schemas import NeuralAnalyticsResult

ANALYTICS_DIR = "analytics"
METADATA_FILE = "metadata.json"
TIMESERIES_FILE = "timeseries.npz"
ATLAS_MAPPING_FILE = "atlas_mapping.npz"


class AnalyticsStore:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)

    def write(
        self,
        result: NeuralAnalyticsResult,
        arrays: dict[str, NDArray[np.generic]],
        atlas: AtlasMapping,
    ) -> Path:
        run_dir = self.artifact_root / "runs" / result.run_id
        if not (run_dir / "predictions.npy").exists():
            raise FileNotFoundError(f"completed source run not found: {run_dir}")
        output = run_dir / ANALYTICS_DIR
        output.mkdir(parents=True, exist_ok=True)
        self._write_npz(output / TIMESERIES_FILE, arrays)
        self._write_npz(
            output / ATLAS_MAPPING_FILE,
            {
                "vertex_to_region": atlas.vertex_to_region.astype(np.int32, copy=False),
                "medial_wall_mask": atlas.medial_wall_mask.astype(bool, copy=False),
            },
        )
        self._write_bytes(output / METADATA_FILE, result.model_dump_json(indent=2).encode())
        return output

    def read(self, run_id: str) -> NeuralAnalyticsResult:
        path = self.artifact_root / "runs" / run_id / ANALYTICS_DIR / METADATA_FILE
        return NeuralAnalyticsResult.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_npz(path: Path, arrays: dict[str, NDArray[np.generic]]) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_bytes(path: Path, payload: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
