"""Step 76: the artifact layout for a search, and the final report.

    artifacts/search/<search_id>/
        config.json          the experiment definition as it was declared
        search_space.json    what was allowed to vary
        budget.json          the ceilings and the ledger
        state.json           the resumable checkpoint
        trajectory.json      best-so-far against evaluation count
        population/          one file per generation
        evaluations/         one file per evaluated candidate
        pareto_front.json    written only for a multi-objective search
        events.json          the ordered history
        final_report.json    what was found and what it does not show

WHY SEPARATE FILES RATHER THAN ONE BLOB
    `state.json` is rewritten after every evaluation, so it must stay small
    enough that writing it is not itself a cost. Everything a reader wants to
    open on its own, and everything that only changes once, lives beside it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from blackmirror.search.fitness import CandidateFitness
from blackmirror.search.schemas import SearchExperiment
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class SearchStore:
    """Reads and writes one search's artifacts."""

    def __init__(self, artifact_root: Path) -> None:
        self.root = Path(artifact_root) / "search"

    def directory(self, search_id: str) -> Path:
        _validate_id(search_id)
        return self.root / search_id

    def exists(self, search_id: str) -> bool:
        return (self.directory(search_id) / "state.json").is_file()

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and (entry / "state.json").is_file()
        )

    # --- writing ----------------------------------------------------------

    def write_definition(self, experiment: SearchExperiment) -> Path:
        """The parts that do not change while the search runs."""
        directory = self.directory(experiment.search_id)
        directory.mkdir(parents=True, exist_ok=True)
        _write(directory / "config.json", experiment.config.model_dump_json(indent=2))
        _write(directory / "search_space.json", experiment.space.model_dump_json(indent=2))
        _write(directory / "budget.json", experiment.budget.model_dump_json(indent=2))
        return directory

    def write_state(self, search_id: str, snapshot: dict[str, object]) -> Path:
        directory = self.directory(search_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "state.json"
        _write(path, json.dumps(snapshot, indent=2, default=str))
        return path

    def write_evaluation(self, search_id: str, fitness: CandidateFitness) -> Path:
        directory = self.directory(search_id) / "evaluations"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{fitness.candidate_id}.json"
        _write(path, fitness.model_dump_json(indent=2))
        return path

    def write_population(self, search_id: str, generation: int, payload: object) -> Path:
        directory = self.directory(search_id) / "population"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"generation-{generation:03d}.json"
        _write(path, json.dumps(payload, indent=2, default=str))
        return path

    def write_named(self, search_id: str, name: str, payload: object) -> Path:
        if not re.fullmatch(r"[a-z0-9_]+\.json", name):
            raise ValueError(f"unsafe artifact name: {name!r}")
        directory = self.directory(search_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        _write(path, json.dumps(payload, indent=2, default=str))
        return path

    # --- reading ----------------------------------------------------------

    def read_state(self, search_id: str) -> dict[str, object]:
        path = self.directory(search_id) / "state.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):  # pragma: no cover - defensive
            raise ValueError("search state is not an object")
        return loaded

    def read_named(self, search_id: str, name: str) -> dict[str, object] | None:
        path = self.directory(search_id) / name
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None

    def read_evaluations(self, search_id: str) -> list[CandidateFitness]:
        directory = self.directory(search_id) / "evaluations"
        if not directory.is_dir():
            return []
        out: list[CandidateFitness] = []
        for path in sorted(directory.glob("*.json")):
            try:
                out.append(
                    CandidateFitness.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                logger.warning("skipping unreadable evaluation %s", path.name)
        return out

    def read_populations(self, search_id: str) -> list[dict[str, object]]:
        directory = self.directory(search_id) / "population"
        if not directory.is_dir():
            return []
        out: list[dict[str, object]] = []
        for path in sorted(directory.glob("generation-*.json")):
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(loaded, dict):
                out.append(loaded)
        return out


def _write(path: Path, payload: str) -> None:
    """Atomic, so a crash mid-write cannot leave a half-file behind."""
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - renamed below
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_id(value: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError("invalid search id")


__all__ = ["SearchStore"]
