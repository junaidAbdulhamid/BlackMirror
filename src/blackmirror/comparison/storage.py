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
            # The manifest was written without an fsync while the arrays it
            # describes were flushed. A crash between the two leaves a
            # comparison directory whose metadata may be absent or truncated
            # while its arrays are intact — the failure mode immutable
            # artifacts exist to prevent.
            metadata_path = temporary / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                handle.write(result.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            for candidate_id, arrays in pair_arrays.items():
                with (temporary / "pairs" / f"{candidate_id}.npz").open("wb") as handle:
                    np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
                    handle.flush()
                    os.fsync(handle.fileno())
            _fsync_directory(temporary / "pairs")
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            # Without this the rename itself can be lost on power failure, and
            # the comparison would vanish despite write() having returned.
            _fsync_directory(self.root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def read(self, comparison_id: str) -> NeuralComparisonResult:
        validate_id(comparison_id, "comparison")
        path = self.root / comparison_id / "metadata.json"
        return NeuralComparisonResult.model_validate_json(path.read_text(encoding="utf-8"))


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry, where the platform supports it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform without directory fds
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - some filesystems refuse this
        pass
    finally:
        os.close(descriptor)


def validate_id(value: str, label: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError(f"invalid {label} id")


def resolve_artifact_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve a metadata path while containing it beneath its artifact directory."""
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its comparison directory")
    return candidate
