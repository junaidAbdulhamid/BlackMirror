"""Exception hierarchy for BlackMirror.

Every failure mode in the inference pipeline raises a specific exception that
names *what* failed and preserves the original cause via ``raise ... from exc``.
A bare ``RuntimeError`` tells a developer nothing; ``InferenceError: TRIBE
inference failed for stimulus 6fd3a1... (cause: CUDA out of memory)`` tells them
where to look.
"""

from __future__ import annotations


class BlackMirrorError(Exception):
    """Base class for every BlackMirror-raised error."""


# --- Stimulus -------------------------------------------------------------


class StimulusError(BlackMirrorError):
    """Base for problems with the input content."""


class UnsupportedStimulusError(StimulusError):
    """The stimulus has a media type or extension the backend cannot process."""


class StimulusValidationError(StimulusError):
    """The stimulus exists but failed a validation check (empty, unreadable, ...)."""


# --- Model lifecycle ------------------------------------------------------


class ModelLoadError(BlackMirrorError):
    """The cortical prediction model could not be located, downloaded or loaded."""


class DeviceUnavailableError(BlackMirrorError):
    """The requested compute device is unavailable or unusable for this model."""


class BackendUnavailableError(BlackMirrorError):
    """A backend's dependencies are not installed in this environment."""


# --- Pipeline stages ------------------------------------------------------


class PreprocessingError(BlackMirrorError):
    """Converting the stimulus into model-ready input failed."""


class InferenceError(BlackMirrorError):
    """The model forward pass failed."""


class PostprocessingError(BlackMirrorError):
    """Converting raw model output into the standardized contract failed."""


class PredictionValidationError(BlackMirrorError):
    """Predictions were produced but are structurally unusable.

    Raised only for *structural* failures (wrong rank, zero-length time axis,
    non-numeric dtype). Numerical anomalies such as NaN or Inf are recorded in
    :class:`~blackmirror.schemas.prediction.PredictionValidation` and reported,
    never silently repaired — see docs/scientific_limitations.md.
    """


# --- Persistence ----------------------------------------------------------


class ArtifactWriteError(BlackMirrorError):
    """An artifact could not be written to the artifact store."""


class ArtifactReadError(BlackMirrorError):
    """An artifact could not be read back from the artifact store."""
