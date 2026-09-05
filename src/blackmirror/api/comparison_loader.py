"""Read-only access to persisted Phase 5 comparison artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blackmirror.api.loader import BinaryArray
from blackmirror.comparison.pipeline import compare_runs
from blackmirror.comparison.schemas import NeuralComparisonResult
from blackmirror.comparison.storage import ComparisonStore, resolve_artifact_path


class ComparisonLoader:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store = ComparisonStore(self.artifact_root)

    def metadata(self, comparison_id: str) -> NeuralComparisonResult:
        return self.store.read(comparison_id)

    def create(
        self,
        reference_run_id: str,
        candidate_run_ids: tuple[str, ...],
        *,
        alignment_method: str = "exact_observed_intersection",
        max_interpolation_gap_seconds: float = 10.0,
    ) -> NeuralComparisonResult:
        return compare_runs(
            self.artifact_root,
            reference_run_id,
            candidate_run_ids,
            alignment_method=alignment_method,
            max_interpolation_gap_seconds=max_interpolation_gap_seconds,
        )

    def list_comparisons(self) -> list[NeuralComparisonResult]:
        if not self.store.root.exists():
            return []
        results: list[NeuralComparisonResult] = []
        for directory in sorted(self.store.root.iterdir(), reverse=True):
            if not directory.is_dir() or not (directory / "metadata.json").exists():
                continue
            results.append(self.store.read(directory.name))
        return results

    def array(self, comparison_id: str, candidate_run_id: str, key: str) -> BinaryArray:
        result = self.metadata(comparison_id)
        pair = next(
            (item for item in result.pairs if item.candidate_run_id == candidate_run_id), None
        )
        if pair is None:
            raise KeyError(f"candidate '{candidate_run_id}' is not in comparison")
        if key not in pair.array_keys:
            raise KeyError(f"unknown comparison array '{key}'")
        path = resolve_artifact_path(self.store.root / comparison_id, pair.arrays_path)
        with np.load(path, allow_pickle=False) as archive:
            array = np.asarray(archive[key])
        little = array.astype(array.dtype.newbyteorder("<"), copy=False)
        return BinaryArray(
            data=little.tobytes(order="C"), dtype=str(little.dtype), shape=little.shape
        )
