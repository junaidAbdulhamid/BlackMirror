"""Atomic immutable storage for Phase 5 comparison artifacts."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from blackmirror.comparison.schemas import NeuralComparisonResult


class ComparisonStore:
    def __init__(self, artifact_root: Path) -> None:
        self.root = Path(artifact_root) / "comparisons"

    def write(
        self,
        result: NeuralComparisonResult,
        pair_arrays: dict[str, dict[str, NDArray[np.generic]]],
    ) -> Path:
        validate_id(result.comparison_id, "comparison")
        expected_candidates = {pair.candidate_run_id for pair in result.pairs}
        if set(pair_arrays) != expected_candidates:
            raise ValueError("pair arrays must exactly match comparison candidates")
        for candidate_id in pair_arrays:
            validate_id(candidate_id, "candidate run")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / result.comparison_id
        if destination.exists():
            raise FileExistsError(f"comparison already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{result.comparison_id}.", dir=self.root))
        try:
            (temporary / "pairs").mkdir()
            (temporary / "metadata.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            for candidate_id, arrays in pair_arrays.items():
                with (temporary / "pairs" / f"{candidate_id}.npz").open("wb") as handle:
                    np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def read(self, comparison_id: str) -> NeuralComparisonResult:
        validate_id(comparison_id, "comparison")
        path = self.root / comparison_id / "metadata.json"
        return NeuralComparisonResult.model_validate_json(path.read_text(encoding="utf-8"))


def validate_id(value: str, label: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError(f"invalid {label} id")


def resolve_artifact_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve a metadata path while containing it beneath its artifact directory."""
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its comparison directory")
    return candidate
