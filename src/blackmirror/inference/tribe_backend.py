"""TRIBE v2 backend — the only place in BlackMirror that knows how TRIBE works.

Everything TRIBE-specific is confined here and in ``tribe_loader.py``:

* how a stimulus becomes an events DataFrame
* how ``predict`` returns ``(preds, segments)`` and what those segments mean
* that predictions land on fsaverage5 with left and right hemispheres
  concatenated
* that predictions carry a 5-second hemodynamic offset

Replacing TRIBE with another cortical prediction model means writing one new
class satisfying ``CorticalPredictorBackend`` — no other module changes.
"""

from __future__ import annotations

import logging
import numbers
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from blackmirror.config.settings import Settings
from blackmirror.errors import InferenceError, PostprocessingError
from blackmirror.inference.backend import PreparedStimulus, RawPrediction, SegmentRecord
from blackmirror.inference.preprocessing import build_events_dataframe, summarize_events
from blackmirror.inference.tribe_loader import (
    HEMODYNAMIC_OFFSET_SOURCE,
    TribeModelLoader,
)
from blackmirror.schemas.metadata import ModelMetadata
from blackmirror.schemas.stimulus import StimulusInput
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: fsaverage5 vertices per hemisphere. Asserted by
#: ``tribev2/utils_fmri.py``: ``FSAVERAGE_5 = _FmriTemplateSpaceSpec("fsaverage5", (10242,))``.
#: Used only to cross-check what the model actually returns — never to reshape it.
FSAVERAGE_SIZES: dict[str, int] = {
    "fsaverage3": 642,
    "fsaverage4": 2562,
    "fsaverage5": 10242,
    "fsaverage6": 40962,
    "fsaverage7": 163842,
}

PREDICTION_SEMANTICS = (
    "Predicted fMRI BOLD response amplitude at one cortical surface vertex for one "
    "repetition time, for an average subject. Not a measurement, not a firing rate, "
    "and not a psychological quantity."
)


@dataclass
class _SegmentCounts:
    """Pre- and post-filter segment counts reported by TRIBE."""

    kept: int | None = None
    total: int | None = None


class _SegmentCountHandler(logging.Handler):
    """Reads TRIBE's segment tallies from its own LogRecord.

    ``TribeModel.predict`` knows how many per-TR segments existed before
    ``remove_empty_segments`` filtering, but returns only the survivors. The
    count reaches us solely through::

        logger.info("Predicted %d / %d segments (%.1f%% kept)", n_kept, n_samples, pct)

    We read ``record.args`` — the structured integers the caller passed — rather
    than formatting and re-parsing the rendered message. That still depends on the
    call's argument order, so every field is validated and any mismatch degrades
    to ``None`` rather than to a wrong number.
    """

    def __init__(self, counts: _SegmentCounts, thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self.counts = counts
        # Records are accepted only from the thread that opened this capture.
        # Two concurrent predicts share one module-level logger, so without
        # this each handler would see BOTH tallies and the last writer would
        # win — silently attributing one run's segment counts to the other.
        self.thread_id = thread_id

    @staticmethod
    def _as_count(value: object) -> int | None:
        """Coerce a logged tally to int, or None if it is not a whole count.

        Upstream builds `n_kept` with `keep.sum()` on a NumPy bool array, so it
        arrives as `np.int64` — and `isinstance(np.int64(3), int)` is False.
        Checking `numbers.Integral` accepts both Python and NumPy integers while
        still rejecting floats, strings and None. `bool` is excluded because it
        is Integral but never a segment count.
        """
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            return None
        return int(value)

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        message = str(record.msg)
        if "Predicted" not in message or "segments" not in message:
            return
        args = record.args
        if not isinstance(args, tuple) or len(args) < 2:
            return
        kept = self._as_count(args[0])
        total = self._as_count(args[1])
        if kept is None or total is None or not total >= kept >= 0:
            return
        self.counts.kept = kept
        self.counts.total = total


#: Guards the shared logger's level/propagate flags. Concurrent captures each
#: get their own handler, but they all mutate one logger, so the mutation is
#: reference-counted rather than saved and restored independently — otherwise
#: the first capture to finish would restore the level while another is still
#: running and silently lose that run's tally.
_LOGGER_MUTATION_LOCK = threading.Lock()


@dataclass
class _LoggerMutation:
    """Reference-counted record of how we changed the shared TRIBE logger."""

    depth: int = 0
    level: int = logging.NOTSET
    propagate: bool = True


_LOGGER_MUTATION = _LoggerMutation()


@contextmanager
def _capture_segment_counts() -> Iterator[_SegmentCounts]:
    """Attach a handler for the duration of one predict call.

    Concurrency
    -----------
    ``tribev2.demo_utils`` has a single module-level logger, so two predicts
    running at once would both see each other's records. Each capture therefore
    filters by the thread that opened it (see :class:`_SegmentCountHandler`),
    and the shared level/propagate mutation is reference-counted under a lock.

    Level
    -----
    The tally is emitted at INFO. If the effective level is higher — a user
    running at WARNING, say — the record would never reach a handler and the
    count would silently vanish, so the level is lowered for the duration.
    Lowering it would also let previously-suppressed records escape to the
    user's own handlers, so propagation is disabled for exactly that case.
    Visible log output is therefore unchanged either way.
    """
    counts = _SegmentCounts()
    tribe_logger = logging.getLogger("tribev2.demo_utils")
    handler = _SegmentCountHandler(counts, threading.get_ident())

    with _LOGGER_MUTATION_LOCK:
        tribe_logger.addHandler(handler)
        if _LOGGER_MUTATION.depth > 0:
            _LOGGER_MUTATION.depth += 1
        elif tribe_logger.getEffectiveLevel() > logging.INFO:
            _LOGGER_MUTATION.level = tribe_logger.level
            _LOGGER_MUTATION.propagate = tribe_logger.propagate
            tribe_logger.setLevel(logging.INFO)
            tribe_logger.propagate = False
            _LOGGER_MUTATION.depth = 1

    try:
        yield counts
    finally:
        with _LOGGER_MUTATION_LOCK:
            tribe_logger.removeHandler(handler)
            if _LOGGER_MUTATION.depth == 1:
                tribe_logger.setLevel(_LOGGER_MUTATION.level)
                tribe_logger.propagate = _LOGGER_MUTATION.propagate
                _LOGGER_MUTATION.depth = 0
            elif _LOGGER_MUTATION.depth > 1:
                _LOGGER_MUTATION.depth -= 1


class TribeV2Backend:
    """Adapts TRIBE v2 to the :class:`CorticalPredictorBackend` protocol."""

    name = "tribe_v2"

    def __init__(self, settings: Settings, device: str) -> None:
        self.settings = settings
        self.loader = TribeModelLoader(settings, device)

    # --- Lifecycle -------------------------------------------------------

    def load(self) -> None:
        self.loader.load()

    def is_loaded(self) -> bool:
        return self.loader.is_loaded()

    def unload(self) -> None:
        self.loader.unload()

    def get_model_metadata(self) -> ModelMetadata:
        return self.loader.get_model_metadata()

    def supported_media_types(self) -> frozenset[str]:
        # Mirrors tribev2.demo_utils.VALID_SUFFIXES.
        return frozenset({"video", "audio", "text"})

    # --- Pipeline --------------------------------------------------------

    def preprocess(self, stimulus: StimulusInput) -> PreparedStimulus:
        model = self.loader.load()
        transcribe = self.settings.enable_transcription
        events = build_events_dataframe(model, stimulus, transcribe=transcribe)
        summary = summarize_events(events, stimulus=stimulus, transcribed=transcribe)
        return PreparedStimulus(
            stimulus=stimulus,
            payload=events,
            event_summary=summary,
            events_table=events,
        )

    def infer(self, prepared: PreparedStimulus) -> RawPrediction:
        model = self.loader.load()

        try:
            with _capture_segment_counts() as counts:
                predictions, segments = model.predict(
                    events=prepared.payload, verbose=False
                )
        except Exception as exc:
            raise InferenceError(
                f"TRIBE v2 forward pass failed on {self.loader.device}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        array = np.asarray(predictions)
        tr_seconds = self.loader.get_tr_seconds()
        surface_space = self.loader.get_surface_space()

        if counts.total is not None and counts.kept != len(segments):
            # The tally must describe the array we actually received, or it is not
            # about this call. Report nothing rather than something misleading.
            logger.warning(
                "TRIBE reported %s kept segments but returned %d; discarding the "
                "segment tally for this run.",
                counts.kept,
                len(segments),
            )
            counts.total = None

        records = _to_segment_records(segments, tr_seconds)
        per_hemisphere = _resolve_vertices_per_hemisphere(array, surface_space)

        return RawPrediction(
            array=array,
            segments=records,
            tr_seconds=tr_seconds,
            surface_space=surface_space,
            vertices_per_hemisphere=per_hemisphere,
            hemisphere_order=("left", "right"),
            includes_subcortex=False,
            hemodynamic_offset_seconds=self.loader.get_hemodynamic_offset_seconds(),
            # Established from the model's own config and training-time code, not
            # from prose docs. See HEMODYNAMIC_OFFSET_SOURCE and
            # docs/tribe_integration.md.
            hemodynamic_offset_verified=True,
            hemodynamic_offset_source=HEMODYNAMIC_OFFSET_SOURCE,
            output_is_stimulus_aligned=True,
            segments_were_filtered=self.settings.remove_empty_segments,
            n_segments_total=counts.total,
            units=None,
            normalization=(
                "Model-internal. TRIBE v2 predicts normalized BOLD response; upstream "
                "publishes no physical unit."
            ),
            semantics=PREDICTION_SEMANTICS,
            medial_wall_handling=(
                "The model emits a value for every fsaverage5 vertex, including the medial "
                "wall, where fMRI signal is not meaningful. Mask before interpreting; "
                "`blackmirror export-mesh` writes medial_wall_mask.npy."
            ),
            atlas=None,
            extra={
                "applied_compat_patches": list(self.loader.applied_compat_patches),
                "remove_empty_segments": self.settings.remove_empty_segments,
                "transcription_enabled": self.settings.enable_transcription,
            },
        )


def _to_segment_records(segments: list, tr_seconds: float) -> list[SegmentRecord]:
    """Convert neuralset ``Segment`` objects into our transport-neutral records.

    ``Segment`` exposes ``start`` (absolute position on the stimulus timeline),
    ``duration`` and ``ns_events``. ``TribeModel.predict`` builds each row's
    segment via ``segment.copy(offset=t, duration=TR)``, and ``copy`` sets
    ``start = self.start + offset``, so ``start`` is already absolute.

    Attributes are read defensively: TRIBE is a research package and a layout
    change upstream must produce a clear error here, not a silently wrong
    timeline.
    """
    records: list[SegmentRecord] = []
    for index, segment in enumerate(segments):
        start = getattr(segment, "start", None)
        if start is None:
            raise PostprocessingError(
                f"Segment {index} has no 'start' attribute (got {type(segment).__name__}). "
                f"Temporal alignment cannot be established, so the run is rejected rather "
                f"than persisted with a guessed timeline."
            )
        duration = getattr(segment, "duration", None)

        events = getattr(segment, "ns_events", None) or []
        event_types = tuple(
            sorted({str(getattr(event, "type", type(event).__name__)) for event in events})
        )

        records.append(
            SegmentRecord(
                index=index,
                start_seconds=float(start),
                duration_seconds=float(duration if duration is not None else tr_seconds),
                n_events=len(events),
                event_types=event_types,
            )
        )
    return records


def _resolve_vertices_per_hemisphere(array: np.ndarray, surface_space: str) -> int:
    """Determine vertices per hemisphere from the ACTUAL output shape.

    The declared space is cross-checked against the returned width; a mismatch
    is reported rather than assumed away, because a wrong split would mirror the
    two hemispheres in every downstream visualization.
    """
    if array.ndim < 2:
        raise PostprocessingError(
            f"Expected a 2-D [time, vertex] prediction, got shape {array.shape}."
        )
    width = int(array.shape[1])

    expected = FSAVERAGE_SIZES.get(surface_space)
    if expected is not None and width == expected * 2:
        return expected

    for space, per_hemisphere in FSAVERAGE_SIZES.items():
        if width == per_hemisphere * 2:
            logger.warning(
                "Model config reports space '%s' but the output width (%d) matches '%s'. "
                "Using %d vertices per hemisphere.",
                surface_space,
                width,
                space,
                per_hemisphere,
            )
            return per_hemisphere

    if width % 2 == 0:
        logger.warning(
            "Output width %d matches no known fsaverage resolution; assuming two equal "
            "hemispheres of %d vertices. Verify before using the cortical mapping.",
            width,
            width // 2,
        )
        return width // 2

    raise PostprocessingError(
        f"Prediction width {width} is odd and matches no known fsaverage resolution, so "
        f"it cannot be split into two hemispheres. The cortical mapping is unknown."
    )
