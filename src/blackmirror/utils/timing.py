"""Stage timing and peak-memory capture.

Phase 1 collects these to establish a baseline we can optimize against later
(and to know whether a variant comparison in Phase 5 is affordable).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTimings:
    """Accumulates per-stage wall-clock durations for one run."""

    stages: dict[str, float] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start)

    def get(self, name: str) -> float:
        return self.stages.get(name, 0.0)

    @property
    def total_seconds(self) -> float:
        """Wall-clock time since this object was created, not the sum of stages.

        The difference (setup, I/O between stages) is real time the user waited,
        so we report it honestly rather than under-counting.
        """
        return time.perf_counter() - self._started


@dataclass(frozen=True)
class MemorySnapshot:
    """Peak memory observed during a run, where the platform can report it."""

    peak_gpu_allocated_bytes: int | None = None
    peak_gpu_reserved_bytes: int | None = None
    peak_host_rss_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_gpu_allocated_bytes": self.peak_gpu_allocated_bytes,
            "peak_gpu_reserved_bytes": self.peak_gpu_reserved_bytes,
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
        }


def reset_peak_memory() -> None:
    """Reset CUDA peak-memory counters, if CUDA is present."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def capture_peak_memory() -> MemorySnapshot:
    """Read peak memory counters.

    On CPU-only machines (including Apple Silicon) CUDA counters are absent and
    we fall back to host RSS via ``resource``, which is available on POSIX.
    """
    gpu_allocated: int | None = None
    gpu_reserved: int | None = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_allocated = int(torch.cuda.max_memory_allocated())
            gpu_reserved = int(torch.cuda.max_memory_reserved())
    except ImportError:
        pass

    host_rss: int | None = None
    try:
        import resource
        import sys

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS/BSD report bytes.
        host_rss = int(raw) if sys.platform == "darwin" else int(raw) * 1024
    except (ImportError, AttributeError):
        pass

    return MemorySnapshot(
        peak_gpu_allocated_bytes=gpu_allocated,
        peak_gpu_reserved_bytes=gpu_reserved,
        peak_host_rss_bytes=host_rss,
    )
