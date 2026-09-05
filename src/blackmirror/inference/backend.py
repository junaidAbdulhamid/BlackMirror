"""The abstraction boundary.

This module defines the only contract the inference service knows about. Swap
TRIBE v2 for any other cortical prediction model by writing one class that
satisfies :class:`CorticalPredictorBackend`; nothing else in BlackMirror changes.

Nothing here imports torch, tribev2, or any model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from blackmirror.schemas.metadata import ModelMetadata
from blackmirror.schemas.stimulus import StimulusInput


@dataclass(frozen=True)
class SegmentRecord:
    """One prediction row's position on the stimulus timeline.

    TRIBE v2 splits the stimulus into segments of length TR and — by default —
    drops those containing no events, so these records are the authoritative
    mapping from row index to time. See ``TribeModel.predict``.
    """

    index: int
    start_seconds: float
    duration_seconds: float
    n_events: int = 0
    event_types: tuple[str, ...] = ()


@dataclass
class PreparedStimulus:
    """Backend-specific, model-ready representation of a stimulus.

    ``payload`` is deliberately opaque to the service — for TRIBE it holds the
    events DataFrame. ``event_summary`` is the modality-neutral description that
    survives into the artifacts, because later phases need to correlate content
    events with neural changes without re-running preprocessing.
    """

    stimulus: StimulusInput
    payload: Any
    event_summary: dict[str, Any] = field(default_factory=dict)
    #: Optional pandas DataFrame, persisted verbatim as ``events.parquet``.
    events_table: Any | None = None


@dataclass(frozen=True)
class RawPrediction:
    """Exactly what the model returned, plus the metadata needed to interpret it.

    ``array`` must be the untouched model output. Backends do not normalize,
    clip, reorder or resample.
    """

    array: np.ndarray
    segments: list[SegmentRecord]
    tr_seconds: float

    surface_space: str
    vertices_per_hemisphere: int
    hemisphere_order: tuple[str, ...] = ("left", "right")
    includes_subcortex: bool = False

    hemodynamic_offset_seconds: float = 0.0
    hemodynamic_offset_verified: bool = False
    #: True when the model already aligned its output to stimulus time, so a
    #: prediction row's segment start IS the stimulus content time.
    output_is_stimulus_aligned: bool = True
    #: Where the offset value and its meaning were established.
    hemodynamic_offset_source: str | None = None

    segments_were_filtered: bool = False
    n_segments_total: int | None = None

    units: str | None = None
    normalization: str | None = None
    semantics: str = "Predicted cortical response amplitude."
    medial_wall_handling: str | None = None
    atlas: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CorticalPredictorBackend(Protocol):
    """Any model that predicts cortical responses to multimodal stimuli.

    Lifecycle is deliberately split from inference so a single loaded model can
    serve many stimuli (Phase 5 compares variants A/B/C without reloading).
    """

    #: Stable identifier, also used for device-support lookup.
    name: str

    def load(self) -> None:
        """Load the model. Idempotent; safe to call before every predict."""
        ...

    def is_loaded(self) -> bool:
        """Whether the model is resident and ready."""
        ...

    def unload(self) -> None:
        """Release the model and its memory."""
        ...

    def get_model_metadata(self) -> ModelMetadata:
        """Provenance of the loaded model. Requires :meth:`load` to have run."""
        ...

    def supported_media_types(self) -> frozenset[str]:
        """Media types this backend can actually process end to end."""
        ...

    def preprocess(self, stimulus: StimulusInput) -> PreparedStimulus:
        """Turn a validated stimulus into model-ready input."""
        ...

    def infer(self, prepared: PreparedStimulus) -> RawPrediction:
        """Run the forward pass and return the untouched output."""
        ...
