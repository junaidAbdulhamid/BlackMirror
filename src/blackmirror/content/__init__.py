"""Phase 4 — multimodal content intelligence.

Turns a stimulus into timestamped, typed, traceable content events and aligns
them to the predicted cortical responses from Phases 1-3.
"""

from blackmirror.content.schemas import (
    ContentAnalysisResult,
    ContentEvent,
    NeuralContentAssociation,
)
from blackmirror.content.timeline import ContentTimeline

__all__ = [
    "ContentAnalysisResult",
    "ContentEvent",
    "ContentTimeline",
    "NeuralContentAssociation",
]
