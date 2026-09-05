"""Synthetic backend — CLEARLY LABELLED TEST DATA, NOT MODEL PREDICTIONS.

Purpose, and the reason it is production code rather than a test fixture:

1. It proves the abstraction. If the contract, artifact store, validation, CLI
   and (later) the FastAPI layer all work against a completely different
   "model", then swapping TRIBE v2 for another cortical predictor really is a
   one-class change.
2. It makes unit tests fast. No 709 MB checkpoint, no gated Llama access, no
   hours of CPU feature extraction.
3. It unblocks Phase 2 development. A visualizer can be built against real
   fsaverage5-shaped output before the model is installed.

Every output is stamped ``is_synthetic=True`` and every value is deterministic
given the seed. This data must never be presented as a prediction about a brain.
"""

from __future__ import annotations

import numpy as np

from blackmirror.inference.backend import PreparedStimulus, RawPrediction, SegmentRecord
from blackmirror.schemas.metadata import ModelMetadata
from blackmirror.schemas.stimulus import StimulusInput
from blackmirror.utils.hashing import sha256_text
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Matches TRIBE v2 so downstream code exercises realistic shapes.
FSAVERAGE5_VERTICES_PER_HEMISPHERE = 10242
DEFAULT_TR_SECONDS = 1.0


class MockBackend:
    """A deterministic stand-in satisfying :class:`CorticalPredictorBackend`."""

    name = "mock"

    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        seed: int = 1234,
        tr_seconds: float = DEFAULT_TR_SECONDS,
        vertices_per_hemisphere: int = FSAVERAGE5_VERTICES_PER_HEMISPHERE,
        default_duration_seconds: float = 30.0,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._seed = seed
        self._tr_seconds = tr_seconds
        self._vertices_per_hemisphere = vertices_per_hemisphere
        self._default_duration = default_duration_seconds
        self._loaded = False

    # --- Lifecycle -------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        logger.warning(
            "Using the MOCK backend. Output is SYNTHETIC test data, not a prediction "
            "of any brain response."
        )
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="Mock Cortical Predictor (SYNTHETIC)",
            backend=self.name,
            model_id="blackmirror/mock",
            source=None,
            checkpoint=None,
            version="1.0",
            license="N/A (synthetic)",
            modalities=("video", "audio", "text"),
            feature_extractors={},
            loaded_device=self._device,
            dtype=self._dtype,
            parameter_count=0,
            subject_conditioning="none (synthetic)",
            is_synthetic=True,
            extra={
                "warning": "Synthetic data for pipeline development. Not a brain prediction.",
                "seed": self._seed,
            },
        )

    def supported_media_types(self) -> frozenset[str]:
        return frozenset({"video", "audio", "text"})

    # --- Pipeline --------------------------------------------------------

    def preprocess(self, stimulus: StimulusInput) -> PreparedStimulus:
        duration = stimulus.duration_seconds or self._default_duration
        n_segments = max(1, int(np.floor(duration / self._tr_seconds)))

        # Deterministic in the stimulus content, so re-running the same file
        # reproduces the same synthetic output exactly.
        seed_material = sha256_text(f"{stimulus.sha256}:{self._seed}")
        payload = {
            "n_segments": n_segments,
            "duration_seconds": duration,
            "seed": int(seed_material[:8], 16),
        }

        event_summary = {
            "backend": self.name,
            "synthetic": True,
            "n_events": n_segments,
            "event_types": {"MockSegment": n_segments},
            "modalities_present": [stimulus.media_type.value],
            "source_media_type": stimulus.media_type.value,
        }
        logger.info("Generated %d synthetic events", n_segments)
        return PreparedStimulus(
            stimulus=stimulus, payload=payload, event_summary=event_summary, events_table=None
        )

    def infer(self, prepared: PreparedStimulus) -> RawPrediction:
        payload = prepared.payload
        n_segments: int = payload["n_segments"]
        n_vertices = self._vertices_per_hemisphere * 2

        rng = np.random.default_rng(payload["seed"])

        # Smooth, spatially structured, time-varying signal: enough structure
        # that a visualizer and the temporal-variance check have something real
        # to work against, without pretending to be neuroscience.
        time_axis = np.arange(n_segments, dtype=np.float32)[:, None]
        vertex_phase = rng.uniform(0, 2 * np.pi, size=(1, n_vertices)).astype(np.float32)
        vertex_gain = rng.uniform(0.4, 1.2, size=(1, n_vertices)).astype(np.float32)

        signal = vertex_gain * np.sin(2 * np.pi * time_axis / 12.0 + vertex_phase)
        noise = rng.normal(0.0, 0.15, size=(n_segments, n_vertices)).astype(np.float32)
        array = (signal + noise).astype(np.float32)

        segments = [
            SegmentRecord(
                index=i,
                start_seconds=float(i * self._tr_seconds),
                duration_seconds=float(self._tr_seconds),
                n_events=1,
                event_types=("MockSegment",),
            )
            for i in range(n_segments)
        ]

        return RawPrediction(
            array=array,
            segments=segments,
            tr_seconds=self._tr_seconds,
            surface_space="fsaverage5",
            vertices_per_hemisphere=self._vertices_per_hemisphere,
            hemisphere_order=("left", "right"),
            includes_subcortex=False,
            hemodynamic_offset_seconds=0.0,
            hemodynamic_offset_verified=True,
            segments_were_filtered=False,
            n_segments_total=n_segments,
            units=None,
            normalization="synthetic, approximately zero-mean unit-scale",
            semantics=(
                "SYNTHETIC value. Not a predicted cortical response. Generated for "
                "pipeline development and testing only."
            ),
            medial_wall_handling=(
                "Not modelled: synthetic values are emitted for all vertices including "
                "the medial wall."
            ),
            atlas=None,
            extra={"synthetic": True},
        )
