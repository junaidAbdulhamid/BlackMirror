"""Phase 9 API service. A search runs for hours, so nothing here waits for it.

WHY THIS MIRRORS THE PHASE 8 LOADER
    A search is a sequence of Phase 8 evaluations, so it inherits their cost: a
    six-candidate search on a ten-second clip is most of a working day. The same
    shape applies. Starting a search writes its definition, hands the loop to a
    background worker, and returns the state that is now on disk. Clients poll.

WHY ONE WORKER, GLOBALLY
    Not one per search: one in total. Two searches running at once would each be
    running TRIBE, contending for the same weights on a machine that is already
    memory-bound. Queuing the second is slower for that search and faster
    overall, and it keeps the wall-clock figures in each ledger meaningful
    rather than reflecting contention with an unrelated run.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from blackmirror.config.settings import Settings
from blackmirror.search.orchestrator import NeuralSearchOrchestrator
from blackmirror.search.scheduler import HumanOverride
from blackmirror.search.schemas import SearchStatus
from blackmirror.search.storage import SearchStore
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class SearchLoadError(ValueError):
    """The search cannot be started, read or controlled as asked."""


class SearchLoader:
    """Starts searches on a background worker and reads their artifacts."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        settings: Settings | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.store = SearchStore(self.artifact_root)
        self.settings = settings
        self._executor = executor or _shared_executor()
        self._lock = threading.Lock()
        self._jobs: dict[str, Future[None]] = {}
        self._running: dict[str, NeuralSearchOrchestrator] = {}
        self._overrides: dict[str, HumanOverride] = {}

    # --- lifecycle --------------------------------------------------------

    def start(
        self, orchestrator: NeuralSearchOrchestrator, *, execute: bool = True
    ) -> dict[str, object]:
        """Persist the definition, queue the loop, return the initial state."""
        search_id = orchestrator.experiment.search_id
        with self._lock:
            existing = self._jobs.get(search_id)
            if existing is not None and not existing.done():
                raise SearchLoadError(f"search {search_id!r} is already running")
            self.store.write_definition(orchestrator.experiment)
            snapshot = orchestrator.snapshot()
            self.store.write_state(search_id, snapshot)
            self._running[search_id] = orchestrator
            self._overrides[search_id] = orchestrator.override
            if execute:
                self._jobs[search_id] = self._executor.submit(_guarded, orchestrator)
        return snapshot

    def resume(self, search_id: str, orchestrator: NeuralSearchOrchestrator) -> dict[str, object]:
        """Continue a search from its last durable checkpoint."""
        snapshot = self.store.read_state(search_id)
        orchestrator.resume_from(snapshot)
        return self.start(orchestrator)

    def pause(self, search_id: str) -> dict[str, object]:
        """Hold at the next candidate boundary, keeping the run resumable."""
        override = self._overrides.get(search_id)
        if override is None:
            raise SearchLoadError(f"search {search_id!r} is not running here")
        override.pause()
        return self.state(search_id)

    def unpause(self, search_id: str) -> dict[str, object]:
        override = self._overrides.get(search_id)
        if override is None:
            raise SearchLoadError(f"search {search_id!r} is not running here")
        override.resume()
        return self.state(search_id)

    def stop(self, search_id: str) -> dict[str, object]:
        """End the search at the next candidate boundary."""
        orchestrator = self._running.get(search_id)
        if orchestrator is None:
            raise SearchLoadError(f"search {search_id!r} is not running here")
        orchestrator.request_stop()
        return self.state(search_id)

    # --- human override ---------------------------------------------------

    def pin(self, search_id: str, candidate_id: str) -> dict[str, object]:
        self._override(search_id).pin(candidate_id)
        return {"pinned": sorted(self._override(search_id).pinned)}

    def eliminate(self, search_id: str, candidate_id: str) -> dict[str, object]:
        self._override(search_id).eliminate(candidate_id)
        return {"eliminated": sorted(self._override(search_id).eliminated)}

    def _override(self, search_id: str) -> HumanOverride:
        override = self._overrides.get(search_id)
        if override is None:
            raise SearchLoadError(f"search {search_id!r} is not running here")
        return override

    # --- reads ------------------------------------------------------------

    def ids(self) -> list[str]:
        return self.store.list_ids()

    def state(self, search_id: str) -> dict[str, object]:
        return self.store.read_state(search_id)

    def is_running(self, search_id: str) -> bool:
        job = self._jobs.get(search_id)
        return job is not None and not job.done()

    def summary(self, search_id: str) -> dict[str, object]:
        """Enough for a list row without loading every evaluation."""
        state = self.store.read_state(search_id)
        experiment = state.get("experiment")
        metrics = state.get("metrics")
        return {
            "search_id": search_id,
            "experiment_id": (
                experiment.get("experiment_id") if isinstance(experiment, dict) else None
            ),
            "status": experiment.get("status") if isinstance(experiment, dict) else None,
            "strategy": (
                experiment.get("config", {}).get("strategy")
                if isinstance(experiment, dict)
                else None
            ),
            "stopping_reason": state.get("stopping_reason"),
            "best_candidate_id": state.get("best_candidate_id"),
            "metrics": metrics if isinstance(metrics, dict) else {},
            "running": self.is_running(search_id),
            "paused": search_id in self._overrides and self._overrides[search_id].paused,
        }

    def best(self, search_id: str) -> dict[str, object] | None:
        state = self.store.read_state(search_id)
        best = state.get("best")
        return best if isinstance(best, dict) else None

    def trajectory(self, search_id: str) -> list[dict[str, object]]:
        state = self.store.read_state(search_id)
        points = state.get("trajectory")
        return points if isinstance(points, list) else []

    def events(self, search_id: str) -> list[dict[str, object]]:
        state = self.store.read_state(search_id)
        events = state.get("events")
        return events if isinstance(events, list) else []

    def budget(self, search_id: str) -> dict[str, object]:
        state = self.store.read_state(search_id)
        experiment = state.get("experiment")
        if not isinstance(experiment, dict):
            return {}
        return {
            "budget": experiment.get("budget", {}),
            "ledger": experiment.get("ledger", {}),
        }

    def population(self, search_id: str) -> list[dict[str, object]]:
        stored = self.store.read_populations(search_id)
        if stored:
            return stored
        state = self.store.read_state(search_id)
        live = state.get("populations")
        return live if isinstance(live, list) else []

    def pareto(self, search_id: str) -> dict[str, object] | None:
        stored = self.store.read_named(search_id, "pareto_front.json")
        if stored is not None:
            return stored
        report = self.store.read_named(search_id, "final_report.json")
        if isinstance(report, dict):
            front = report.get("pareto_front")
            if isinstance(front, dict):
                return front
        return None

    def report(self, search_id: str) -> dict[str, object] | None:
        return self.store.read_named(search_id, "final_report.json")

    def results(self, search_id: str) -> list[dict[str, object]]:
        state = self.store.read_state(search_id)
        results = state.get("results")
        return results if isinstance(results, list) else []


def _guarded(orchestrator: NeuralSearchOrchestrator) -> None:
    try:
        orchestrator.run()
    except Exception:
        logger.exception(
            "search %s failed outside the loop", orchestrator.experiment.search_id
        )
        raise


_EXECUTOR: ThreadPoolExecutor | None = None
_LOADERS: dict[str, SearchLoader] = {}
_MODULE_LOCK = threading.Lock()


def _shared_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _MODULE_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search")
        return _EXECUTOR


def get_search_loader(
    artifact_root: Path, *, settings: Settings | None = None
) -> SearchLoader:
    """One loader per artifact root, so in-flight jobs are not forgotten."""
    key = str(Path(artifact_root).resolve())
    with _MODULE_LOCK:
        loader = _LOADERS.get(key)
        if loader is None:
            loader = SearchLoader(Path(artifact_root), settings=settings)
            _LOADERS[key] = loader
        return loader


def reset_search_loaders() -> None:
    """Drop cached loaders. For tests, which must not share worker state."""
    with _MODULE_LOCK:
        _LOADERS.clear()


__all__ = [
    "SearchLoadError",
    "SearchLoader",
    "SearchStatus",
    "get_search_loader",
    "reset_search_loaders",
]
