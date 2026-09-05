"""Inference pipeline.

Only ``tribe_backend`` and ``tribe_loader`` know that TRIBE v2 exists, and both
import it lazily. Everything else speaks the ``CorticalPredictorBackend``
protocol defined in ``backend.py``.
"""

from blackmirror.inference.backend import (
    CorticalPredictorBackend,
    PreparedStimulus,
    RawPrediction,
    SegmentRecord,
)
from blackmirror.inference.service import InferenceService

__all__ = [
    "CorticalPredictorBackend",
    "InferenceService",
    "PreparedStimulus",
    "RawPrediction",
    "SegmentRecord",
]
