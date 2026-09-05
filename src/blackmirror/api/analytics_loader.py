"""Read-only access to already-derived Phase 3 artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blackmirror.analytics.schemas import NeuralAnalyticsResult
from blackmirror.analytics.storage import AnalyticsStore
from blackmirror.api.loader import BinaryArray


class AnalyticsLoader:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store = AnalyticsStore(self.artifact_root)

    def metadata(self, run_id: str) -> NeuralAnalyticsResult:
        return self.store.read(run_id)

    def array(self, run_id: str, key: str) -> BinaryArray:
        metadata = self.metadata(run_id)
        allowed = metadata.arrays["timeseries"].keys
        if key not in allowed:
            raise KeyError(f"unknown analytics array '{key}'")
        run_dir = self.artifact_root / "runs" / run_id
        path = run_dir / metadata.arrays["timeseries"].path
        if run_dir.resolve() not in path.resolve().parents:
            raise ValueError("analytics array path escapes its source run")
        with np.load(path, allow_pickle=False) as archive:
            array = np.asarray(archive[key])
        little = array.astype(array.dtype.newbyteorder("<"), copy=False)
        return BinaryArray(
            data=little.tobytes(order="C"), dtype=str(little.dtype), shape=little.shape
        )
