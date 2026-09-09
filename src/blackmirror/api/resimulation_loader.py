"""Phase 8 API service, kept separate from route wiring.

WHY WORK IS QUEUED RATHER THAN AWAITED
    Every other loader in this API answers from files a completed run already
    wrote, which is why the surface can be described as read-only. Phase 8 is
    the one exception in the whole product: it runs TRIBE. A real pass took
    roughly two hours per ten-second variant on this machine, so a handler that
    returned only when the pipeline finished would return nothing at all.

    The state machine was already built to survive that. It checkpoints after
    every stage and is idempotent on restart, so the HTTP path binds the run
    durably, hands execution to a worker, and answers immediately with the state
    that is now on disk. Clients poll. Nothing is held in memory that would be
    lost if the process died mid-run.

WHY ONE WORKER
    Inference is memory-bound before it is compute-bound. Two concurrent passes
    on one machine do not run twice as fast; they contend for the same weights
    and swap. Requests are serialized through a single worker, and the store's
    exclusive lock independently refuses a duplicate start from another process.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from blackmirror.config.settings import Settings
from blackmirror.inference.service import InferenceService
from blackmirror.resimulation.backend import ExistingPipelineBackend
from blackmirror.resimulation.orchestrator import ResimulationOrchestrator
from blackmirror.resimulation.request_builder import RequestBuildError, build_request
from blackmirror.resimulation.schemas import (
    ResimulationConflict,
    ResimulationRequest,
    ResimulationResult,
    StopReason,
)
from blackmirror.resimulation.storage import ResimulationStore
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: One unit of orchestrator work handed to the background worker.
Work = Callable[[ResimulationOrchestrator], ResimulationResult]

#: How many finished jobs to remember for failure reporting. The durable state
#: file is the record of every run; this is only the recent tail.
_RETAINED_JOBS = 64


class ResimulationLoadError(ValueError):
    """The re-simulation cannot be assembled or started from what is stored."""


class ResimulationLoader:
    """Binds requests durably and executes them on a background worker."""

    def __init__(
        self,
        artifact_root: Path,
        inference: InferenceService | None = None,
        *,
        settings: Settings | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.store = ResimulationStore(self.artifact_root)
        self.settings = settings
        self._inference = inference
        self._lock = threading.Lock()
        self._executor = executor or _shared_executor()
        self._jobs: dict[str, Future[ResimulationResult]] = {}

    # --- construction ----------------------------------------------------

    def build(
        self,
        *,
        resimulation_id: str,
        experiment_id: str,
        optimization_request_key: str,
        proposed_variant_id: str,
        variant_media: Path,
        source_media: Path | None = None,
        adapter_id: str = "user-supplied",
        parameters: dict[str, object] | None = None,
        iteration_index: int = 1,
        max_iterations: int = 1,
        max_attempts: int = 3,
        outcome_tolerance: float = 1e-9,
        previous_resimulation_id: str | None = None,
    ) -> ResimulationRequest:
        """Derive a valid request from stored Phase 6 and Phase 7 records."""
        return build_request(
            self.artifact_root,
            resimulation_id=resimulation_id,
            experiment_id=experiment_id,
            optimization_request_key=optimization_request_key,
            proposed_variant_id=proposed_variant_id,
            variant_media=variant_media,
            source_media=source_media,
            adapter_id=adapter_id,
            parameters=parameters,
            iteration_index=iteration_index,
            max_iterations=max_iterations,
            max_attempts=max_attempts,
            outcome_tolerance=outcome_tolerance,
            previous_resimulation_id=previous_resimulation_id,
        )

    # --- execution -------------------------------------------------------

    def create(self, request: ResimulationRequest, *, execute: bool = True) -> ResimulationResult:
        """Bind the request durably, then queue the pipeline behind it.

        The returned state is the checkpoint on disk at the moment of binding,
        not a prediction of what the run will become.
        """
        state = self._orchestrator().prepare(request)
        if execute and state.status.value != "completed":
            self._submit(request.resimulation_id, lambda o: o.start(request))
        return state

    def resume(self, resimulation_id: str, *, execute: bool = True) -> ResimulationResult:
        """Queue a verified resume and return the current durable state."""
        state = self.store.read(resimulation_id)
        if state.status.value == "completed":
            return state
        if execute:
            self._submit(resimulation_id, lambda o: o.resume(resimulation_id))
        return state

    def stop(self, resimulation_id: str, reason: StopReason) -> ResimulationResult:
        """Record a stop request; the worker observes it between stages."""
        return self._orchestrator().stop(resimulation_id, reason)

    def run_blocking(self, request: ResimulationRequest) -> ResimulationResult:
        """Execute inline. Used by the CLI, never by an HTTP handler."""
        return self._orchestrator().start(request)

    # --- reads -----------------------------------------------------------

    def get(self, resimulation_id: str) -> ResimulationResult:
        return self.store.read(resimulation_id)

    def worker_alive(self, resimulation_id: str) -> bool:
        """Whether a live process currently holds this run's lock."""
        if self.store.active_owner(resimulation_id) is not None:
            return True
        job = self._jobs.get(resimulation_id)
        return job is not None and not job.done()

    def list(self, experiment_id: str | None = None) -> list[ResimulationResult]:
        if not self.store.root.is_dir():
            return []
        results = []
        for directory in sorted(self.store.root.iterdir()):
            if not directory.is_dir() or not (directory / "state.json").is_file():
                continue
            try:
                state = self.store.read(directory.name)
            except (OSError, ValueError):
                # A half-written or schema-incompatible state must not take the
                # whole listing down with it.
                logger.warning("skipping unreadable resimulation state: %s", directory.name)
                continue
            if experiment_id is None or state.request.experiment_id == experiment_id:
                results.append(state)
        return sorted(results, key=lambda item: item.updated_at, reverse=True)

    def failure(self, resimulation_id: str) -> str | None:
        """The error a finished background job raised, if it raised one.

        A job that dies before its first checkpoint (a fingerprint mismatch, a
        missing file) leaves nothing in the state file to explain itself. This
        surfaces that message instead of letting the run sit at `active` with no
        account of why nothing is happening.
        """
        job = self._jobs.get(resimulation_id)
        if job is None or not job.done():
            return None
        exception = job.exception()
        return None if exception is None else f"{type(exception).__name__}: {exception}"

    # --- internals -------------------------------------------------------

    def _submit(self, resimulation_id: str, work: Work) -> None:
        with self._lock:
            existing = self._jobs.get(resimulation_id)
            if existing is not None and not existing.done():
                raise ResimulationLoadError(
                    f"re-simulation {resimulation_id!r} is already running on this server"
                )
            orchestrator = self._orchestrator()
            self._jobs[resimulation_id] = self._executor.submit(
                _guarded, resimulation_id, orchestrator, work
            )
            self._prune_finished_jobs()

    def _prune_finished_jobs(self) -> None:
        """Keep only recent finished jobs, so a long-lived server does not grow.

        Each entry holds a completed `ResimulationResult`, and the durable state
        file is the real record of every run. What is kept here is only the
        recent tail needed to report a failure that happened before the state
        machine could write one down. Callers hold `_lock`.
        """
        finished = [key for key, job in self._jobs.items() if job.done()]
        for key in finished[: max(0, len(finished) - _RETAINED_JOBS)]:
            del self._jobs[key]

    def _orchestrator(self) -> ResimulationOrchestrator:
        return ResimulationOrchestrator(
            self.artifact_root,
            ExistingPipelineBackend(self.artifact_root, inference_factory=self._service),
            store=self.store,
        )

    def _service(self) -> InferenceService:
        """Build the inference service on first use, not at import.

        Constructing it resolves a device and instantiates a backend. Doing that
        while the API starts would make every read-only route pay for a feature
        most requests never touch.
        """
        if self._inference is None:
            self._inference = InferenceService(self.settings)
        return self._inference


def _guarded(
    resimulation_id: str, orchestrator: ResimulationOrchestrator, work: Work
) -> ResimulationResult:
    try:
        return work(orchestrator)
    except Exception:
        logger.exception("re-simulation %s failed outside the state machine", resimulation_id)
        raise


_EXECUTOR: ThreadPoolExecutor | None = None
_LOADERS: dict[str, ResimulationLoader] = {}
_MODULE_LOCK = threading.Lock()


def _shared_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _MODULE_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resimulation")
        return _EXECUTOR


def get_resimulation_loader(
    artifact_root: Path, *, settings: Settings | None = None
) -> ResimulationLoader:
    """One loader per artifact root, shared across requests.

    In-flight jobs are tracked on the instance. A loader rebuilt per HTTP
    request would forget them and report a running pass as idle, so the
    instance outlives the request that first needed it.
    """
    key = str(Path(artifact_root).resolve())
    with _MODULE_LOCK:
        loader = _LOADERS.get(key)
        if loader is None:
            loader = ResimulationLoader(Path(artifact_root), settings=settings)
            _LOADERS[key] = loader
        return loader


def reset_resimulation_loaders() -> None:
    """Drop cached loaders. For tests, which must not share worker state."""
    with _MODULE_LOCK:
        _LOADERS.clear()


__all__ = [
    "RequestBuildError",
    "ResimulationConflict",
    "ResimulationLoadError",
    "ResimulationLoader",
    "get_resimulation_loader",
    "reset_resimulation_loaders",
]
