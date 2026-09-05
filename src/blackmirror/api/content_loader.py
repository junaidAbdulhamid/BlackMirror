"""Content analysis API loader.

Read-only access to persisted Phase 4 artifacts, mirroring how
`analytics_loader` exposes Phase 3 and `loader` exposes Phase 1. It runs no
analysis: an un-analysed run 404s with instructions rather than triggering a
17-second CLIP pipeline on a page load.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from blackmirror.config.settings import Settings
from blackmirror.content.schemas import ContentAnalysisResult, ContentEvent
from blackmirror.content.storage import ContentStore
from blackmirror.content.timeline import ContentTimeline
from blackmirror.errors import ArtifactReadError
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class ContentLoader:
    """Serves stored content analysis for visualization."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ContentStore(settings.artifact_dir)

    def available(self, run_id: str) -> bool:
        return self.store.exists(run_id)

    def get(self, run_id: str) -> ContentAnalysisResult:
        return self.store.read(run_id)

    def timeline(self, run_id: str) -> ContentTimeline:
        return ContentTimeline(self.get(run_id).events)

    def context_at(self, run_id: str, time: float, window: float = 0.0) -> dict[str, Any]:
        """What the stimulus was doing at (or around) an instant.

        `window` of 0 asks for the containing event; a positive window asks for
        everything nearby, which is what a neural event needs.
        """
        analysis = self.get(run_id)
        timeline = ContentTimeline(analysis.events)

        if window > 0:
            events: list[ContentEvent] = timeline.window(time, window)
        else:
            found = timeline.get_event_at(time)
            events = [found] if found else []

        return {
            "time": round(time, 3),
            "window_seconds": window,
            "covered": bool(events),
            "events": [event.model_dump(mode="json") for event in events],
            "visual": timeline.get_visual_state_at(time),
            "speech": timeline.get_speech_at(time),
            "audio": timeline.get_audio_state_at(time),
            "semantic": timeline.get_semantic_context_at(time),
            "interpretation": analysis.metadata.interpretation_notice,
        }

    def features(self, run_id: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """The feature matrix, its time base, and its column names."""
        analysis = self.get(run_id)
        arrays = self.store.read_arrays(run_id)
        matrix = np.asarray(arrays.get("features", np.zeros((0, 0))), dtype=np.float32)
        times = np.asarray(arrays.get("feature_times", np.zeros(0)), dtype=np.float32)
        names = analysis.arrays.feature_names if analysis.arrays else ()
        return matrix, times, names

    def signal(self, run_id: str, key: str) -> np.ndarray:
        """One dense signal by name."""
        arrays = self.store.read_arrays(run_id)
        if key not in arrays:
            raise ArtifactReadError(
                f"Signal '{key}' is not present for run {run_id}. Available: {sorted(arrays)}"
            )
        return np.ascontiguousarray(arrays[key], dtype=np.float32)

    def keyframe_path(self, run_id: str, name: str) -> Path:
        """Resolve a keyframe, refusing anything outside the run's directory."""
        directory = self.store.keyframe_directory(run_id).resolve()
        candidate = (directory / name).resolve()
        # Path traversal guard: a crafted name must not escape the run.
        if not candidate.is_relative_to(directory) or not candidate.is_file():
            raise ArtifactReadError(f"Keyframe '{name}' not found for run {run_id}.")
        return candidate


@lru_cache(maxsize=1)
def get_content_loader() -> ContentLoader:
    from blackmirror.config.settings import get_settings

    return ContentLoader(get_settings())
