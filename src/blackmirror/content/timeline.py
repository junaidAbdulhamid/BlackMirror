"""ContentTimeline — the query interface over fused events.

WHAT IT DOES
    Answers "what was happening at time t?" and "what happened between a and b?"
    without every caller re-implementing interval search.

WHY IT EXISTS
    The API, the UI, the association engine and the future A/B engine all ask
    the same temporal questions. One implementation means one set of boundary
    conventions — half-open [start, end) throughout, matching Phase 2's coverage
    semantics — rather than four subtly different ones.

IN:  ContentEvent[]
OUT: lookups by instant and by range
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterator

from blackmirror.content.schemas import ContentEvent


class ContentTimeline:
    """Ordered, queryable view of a stimulus's fused content events."""

    def __init__(self, events: list[ContentEvent] | tuple[ContentEvent, ...]) -> None:
        self._events: list[ContentEvent] = sorted(events, key=lambda e: e.start_time)
        for previous, following in zip(self._events, self._events[1:], strict=False):
            if not math.isclose(previous.end_time, following.start_time, abs_tol=1e-6):
                relation = "overlap" if previous.end_time > following.start_time else "gap"
                raise ValueError(f"Content events must form a contiguous partition ({relation})")
        self._starts: list[float] = [event.start_time for event in self._events]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[ContentEvent]:
        return iter(self._events)

    @property
    def events(self) -> list[ContentEvent]:
        return list(self._events)

    @property
    def duration(self) -> float:
        return self._events[-1].end_time if self._events else 0.0

    def get_event_at(self, time: float) -> ContentEvent | None:
        """The event whose half-open interval contains `time`.

        Binary search over start times, then a containment check: events form a
        partition, so at most one can match.
        """
        if not self._events:
            return None
        index = bisect.bisect_right(self._starts, time) - 1
        if index < 0:
            return None
        candidate = self._events[index]
        if candidate.start_time <= time < candidate.end_time:
            return candidate
        # The final event owns its closing instant, so scrubbing to the very end
        # still resolves rather than returning nothing.
        last = self._events[-1]
        if time == last.end_time:
            return last
        return None

    def get_events_between(self, start: float, end: float) -> list[ContentEvent]:
        """Every event overlapping [start, end)."""
        if end <= start:
            return []
        return [
            event
            for event in self._events
            if event.start_time < end - 1e-9 and event.end_time > start + 1e-9
        ]

    def get_visual_state_at(self, time: float) -> dict[str, object]:
        event = self.get_event_at(time)
        if event is None:
            return {}
        return {
            "description": event.visual_description,
            "objects": list(event.objects),
            "features": dict(event.visual_features),
            "shot_indices": list(event.shot_indices),
            "scene_index": event.scene_index,
        }

    def get_speech_at(self, time: float) -> str | None:
        event = self.get_event_at(time)
        return event.speech_text if event else None

    def get_audio_state_at(self, time: float) -> dict[str, object]:
        event = self.get_event_at(time)
        if event is None:
            return {}
        return {
            "description": event.audio_description,
            "features": dict(event.audio_features),
        }

    def get_semantic_context_at(self, time: float) -> dict[str, object]:
        event = self.get_event_at(time)
        if event is None:
            return {}
        return {
            "semantic_description": event.semantic_description,
            "language_features": dict(event.language_features),
            "on_screen_text": list(event.on_screen_text),
        }

    def window(self, centre: float, half_width: float) -> list[ContentEvent]:
        """Events within ±half_width of an instant.

        The primitive behind neural-event context: a BOLD response reflects a
        stimulus interval, not a single frame.
        """
        if half_width < 0:
            raise ValueError("half_width must be non-negative")
        return self.get_events_between(max(0.0, centre - half_width), centre + half_width)
