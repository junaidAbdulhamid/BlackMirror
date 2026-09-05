"""Immutable, atomic persistence for Phase 6 score results.

WHY SCORES ARE PERSISTED AT ALL
    Scoring is cheap next to inference -- 331 ms for 50 variants against 2h34m
    for one TRIBE run -- so persistence is not a performance measure. It exists
    so that a score can be *cited*: the objective set, the metric formulas, the
    resolved windows and the engine version are stored together, and a number
    quoted from a report can be traced back to the definition that produced it.

SHAPE
    artifacts/experiments/<experiment_id>/scoring/<objective_set_hash>/
        metadata.json          the full ExperimentScoreResult
        objective_set.json     the objectives alone, for reuse
        explanation.json       structured explanations, when generated

    Keying by objective-set hash rather than by timestamp means re-scoring the
    same variants against the same objectives lands in the same place, which is
    what makes the cache lookup in `find` correct rather than merely convenient.

IMMUTABILITY
    A directory is written once via a temporary directory and an atomic rename.
    Re-scoring with the same key is a no-op unless `overwrite=True`, because a
    stored score is evidence and silently replacing it would break the citation.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

from blackmirror.scoring.schemas import ExperimentScoreResult, NeuralObjective

SCORING_DIRNAME = "scoring"


class ScoringStore:
    def __init__(self, artifact_root: Path) -> None:
        self.root = Path(artifact_root) / "experiments"

    def directory(self, experiment_id: str, objective_set_hash: str) -> Path:
        validate_id(experiment_id, "experiment")
        validate_hash(objective_set_hash)
        return self.root / experiment_id / SCORING_DIRNAME / objective_set_hash

    def exists(self, experiment_id: str, objective_set_hash: str) -> bool:
        return (self.directory(experiment_id, objective_set_hash) / "metadata.json").is_file()

    def write(
        self,
        result: ExperimentScoreResult,
        *,
        explanation_json: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        destination = self.directory(result.experiment_id, result.metadata.objective_set_hash)
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"a score for this experiment and objective set already exists at "
                    f"{destination}; scoring is deterministic, so re-running it produces the "
                    f"same numbers. Pass overwrite=True only to replace stored evidence."
                )
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        temporary = Path(tempfile.mkdtemp(prefix=".scoring.", dir=destination.parent))
        try:
            _write(temporary / "metadata.json", result.model_dump_json(indent=2))
            _write(
                temporary / "objective_set.json",
                _objective_set_json(result.objectives),
            )
            if explanation_json is not None:
                _write(temporary / "explanation.json", explanation_json)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def read(self, experiment_id: str, objective_set_hash: str) -> ExperimentScoreResult:
        path = self.directory(experiment_id, objective_set_hash) / "metadata.json"
        return ExperimentScoreResult.model_validate_json(path.read_text(encoding="utf-8"))

    def read_explanation(self, experiment_id: str, objective_set_hash: str) -> str | None:
        path = self.directory(experiment_id, objective_set_hash) / "explanation.json"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def list_for_experiment(self, experiment_id: str) -> list[ExperimentScoreResult]:
        """Every readable score for one experiment, newest first.

        An unreadable directory is skipped rather than raised, matching how the
        run index and the comparison listing handle partial writes: one corrupt
        entry must not hide every healthy one.
        """
        validate_id(experiment_id, "experiment")
        base = self.root / experiment_id / SCORING_DIRNAME
        if not base.is_dir():
            return []
        results: list[ExperimentScoreResult] = []
        for directory in sorted(base.iterdir(), reverse=True):
            manifest = directory / "metadata.json"
            if not manifest.is_file():
                continue
            try:
                results.append(
                    ExperimentScoreResult.model_validate_json(
                        manifest.read_text(encoding="utf-8")
                    )
                )
            except (ValueError, OSError):
                continue
        return sorted(results, key=lambda item: item.metadata.created_at, reverse=True)

    def list_experiments(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and (entry / SCORING_DIRNAME).is_dir()
        )


def _objective_set_json(objectives: tuple[NeuralObjective, ...]) -> str:
    import json

    return json.dumps(
        [objective.model_dump(mode="json") for objective in objectives], indent=2
    )


def _write(path: Path, payload: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform without directory fds
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(descriptor)


def validate_id(value: str, label: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError(f"invalid {label} id")


def validate_hash(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{16,64}", value) is None:
        raise ValueError("invalid objective set hash")
