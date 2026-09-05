"""The data contracts every later phase consumes.

Importing this package pulls in nothing heavier than pydantic and numpy.
"""

from blackmirror.schemas.metadata import ModelMetadata, RunProvenance
from blackmirror.schemas.prediction import (
    ArtifactPaths,
    CorticalMetadata,
    PerformanceMetrics,
    PredictionArrayMetadata,
    PredictionResult,
    PredictionValidation,
    TemporalMetadata,
)
from blackmirror.schemas.stimulus import MediaType, StimulusInput

__all__ = [
    "ArtifactPaths",
    "CorticalMetadata",
    "MediaType",
    "ModelMetadata",
    "PerformanceMetrics",
    "PredictionArrayMetadata",
    "PredictionResult",
    "PredictionValidation",
    "RunProvenance",
    "StimulusInput",
    "TemporalMetadata",
]
