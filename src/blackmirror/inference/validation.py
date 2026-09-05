"""Stimulus ingestion/validation and prediction validation.

Two independent concerns share this module because both answer the same
question at different ends of the pipeline: *is this data usable?*

Scientific-integrity rule: prediction validation **describes**, it never
repairs. NaNs and Infs are counted and reported; the raw array is untouched.
"""

from __future__ import annotations

import mimetypes
import shutil
import subprocess
from pathlib import Path

import numpy as np

from blackmirror.errors import (
    StimulusValidationError,
    UnsupportedStimulusError,
)
from blackmirror.schemas.prediction import PredictionValidation
from blackmirror.schemas.stimulus import SUPPORTED_EXTENSIONS, MediaType, StimulusInput
from blackmirror.utils.hashing import sha256_file
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Refuse absurdly large uploads early rather than after a long hash.
MAX_STIMULUS_BYTES = 2 * 1024**3  # 2 GiB


# ---------------------------------------------------------------------------
# Stimulus
# ---------------------------------------------------------------------------


def probe_media_duration(path: Path) -> float | None:
    """Return media duration in seconds via ffprobe, or None if undeterminable.

    Duration is measured from the file, never inferred from the number of
    prediction rows — that would be circular and would hide dropped segments.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        logger.debug("ffprobe not on PATH; stimulus duration will be unknown.")
        return None

    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("ffprobe failed for %s: %s", path.name, exc)
        return None

    raw = completed.stdout.strip()
    try:
        duration = float(raw)
    except ValueError:
        return None
    return duration if duration > 0 else None


def build_stimulus(
    path: str | Path,
    *,
    supported_media_types: frozenset[str] | None = None,
) -> StimulusInput:
    """Validate a content file and build the typed, content-addressed stimulus.

    Args:
        path: Path to the content file.
        supported_media_types: Narrow the accepted set to what the chosen
            backend can actually process. ``None`` accepts every type
            BlackMirror declares.

    Raises:
        StimulusValidationError: missing, unreadable, empty or oversized file.
        UnsupportedStimulusError: extension or modality not supported.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise StimulusValidationError(f"Stimulus file does not exist: {resolved}")
    if not resolved.is_file():
        raise StimulusValidationError(f"Stimulus path is not a regular file: {resolved}")

    media_type = MediaType.from_extension(resolved.suffix)
    if media_type is None:
        supported = sorted(ext for group in SUPPORTED_EXTENSIONS.values() for ext in group)
        raise UnsupportedStimulusError(
            f"Unsupported stimulus extension '{resolved.suffix}' for {resolved.name}. "
            f"Supported extensions: {supported}"
        )

    if supported_media_types is not None and media_type.value not in supported_media_types:
        raise UnsupportedStimulusError(
            f"Stimulus '{resolved.name}' is {media_type.value}, but the selected backend "
            f"supports only {sorted(supported_media_types)}."
        )

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise StimulusValidationError(f"Cannot stat stimulus file {resolved}: {exc}") from exc

    if size == 0:
        raise StimulusValidationError(f"Stimulus file is empty: {resolved}")
    if size > MAX_STIMULUS_BYTES:
        raise StimulusValidationError(
            f"Stimulus file is {size / 1e9:.2f} GB, above the "
            f"{MAX_STIMULUS_BYTES / 1e9:.2f} GB limit: {resolved}"
        )

    try:
        with resolved.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise StimulusValidationError(f"Stimulus file is not readable: {resolved}") from exc

    if media_type is MediaType.TEXT:
        try:
            text = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StimulusValidationError(
                f"Text stimulus is not valid UTF-8: {resolved}"
            ) from exc
        if not text.strip():
            raise StimulusValidationError(f"Text stimulus contains no content: {resolved}")

    duration = probe_media_duration(resolved) if media_type is not MediaType.TEXT else None
    digest = sha256_file(resolved)

    stimulus = StimulusInput(
        path=resolved,
        filename=resolved.name,
        media_type=media_type,
        mime_type=mimetypes.guess_type(resolved.name)[0],
        file_size_bytes=size,
        duration_seconds=duration,
        sha256=digest,
    )
    logger.info("Stimulus validated: %s", stimulus.summary_line())
    return stimulus


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def validate_predictions(
    array: np.ndarray,
    *,
    expected_vertex_count: int | None = None,
    max_constant_vertex_ratio: float = 0.5,
) -> PredictionValidation:
    """Compute structural and numerical health statistics for a prediction matrix.

    Structural problems set ``valid=False`` and populate ``errors``. Numerical
    anomalies (NaN, Inf, dead vertices, no temporal variance) populate
    ``warnings`` — the run still completes and the raw data is still persisted,
    because suppressing anomalous scientific output is worse than surfacing it.

    Args:
        array: Raw prediction matrix, expected rank 2 as ``[time, vertex]``.
        expected_vertex_count: Vertex count the backend declared. A mismatch is
            a structural error — it means our cortical mapping is wrong.
        max_constant_vertex_ratio: Fraction of zero-variance vertices above
            which we warn.
    """
    errors: list[str] = []
    warnings: list[str] = []

    shape = tuple(int(dim) for dim in array.shape)
    dtype = str(array.dtype)

    if array.ndim != 2:
        errors.append(f"Expected a 2-D [time, vertex] matrix, got {array.ndim}-D {shape}.")
    if not np.issubdtype(array.dtype, np.number):
        errors.append(f"Prediction dtype {dtype} is not numeric.")

    n_time = shape[0] if len(shape) > 0 else 0
    n_vertices = shape[1] if len(shape) > 1 else 0

    if n_time == 0:
        errors.append("Temporal dimension is zero: the model produced no time points.")
    if n_vertices == 0:
        errors.append("Vertex dimension is zero: the model produced no cortical features.")
    if expected_vertex_count is not None and n_vertices not in (0, expected_vertex_count):
        errors.append(
            f"Vertex count {n_vertices} does not match the backend's declared "
            f"{expected_vertex_count}. The cortical mapping cannot be trusted."
        )

    if errors:
        return PredictionValidation(
            valid=False,
            shape=shape,
            dtype=dtype,
            nan_count=0,
            inf_count=0,
            finite_count=0,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    values = array.astype(np.float64, copy=False)
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)
    finite_mask = np.isfinite(values)

    nan_count = int(nan_mask.sum())
    inf_count = int(inf_mask.sum())
    finite_count = int(finite_mask.sum())

    if nan_count:
        warnings.append(
            f"{nan_count} NaN value(s) ({100 * nan_count / values.size:.4f}%). "
            f"Reported, not removed — inspect before using this run."
        )
    if inf_count:
        warnings.append(
            f"{inf_count} infinite value(s) ({100 * inf_count / values.size:.4f}%). "
            f"Reported, not clipped."
        )

    stats: dict[str, float | None] = dict.fromkeys(
        ("min", "max", "mean", "std", "p01", "p50", "p99")
    )
    constant_vertices: int | None = None
    temporal_variance_mean: float | None = None

    if finite_count:
        finite_values = values[finite_mask]
        stats["min"] = float(finite_values.min())
        stats["max"] = float(finite_values.max())
        stats["mean"] = float(finite_values.mean())
        stats["std"] = float(finite_values.std())
        p01, p50, p99 = np.percentile(finite_values, [1.0, 50.0, 99.0])
        stats["p01"], stats["p50"], stats["p99"] = float(p01), float(p50), float(p99)

        if n_time > 1:
            # nanvar over time per vertex; all-NaN columns yield NaN, which we ignore.
            with np.errstate(invalid="ignore"):
                per_vertex_var = np.nanvar(np.where(finite_mask, values, np.nan), axis=0)
            usable = per_vertex_var[np.isfinite(per_vertex_var)]
            if usable.size:
                constant_vertices = int((usable == 0.0).sum())
                temporal_variance_mean = float(usable.mean())

                ratio = constant_vertices / usable.size
                if ratio > max_constant_vertex_ratio:
                    warnings.append(
                        f"{constant_vertices}/{usable.size} vertices "
                        f"({100 * ratio:.1f}%) are constant over time."
                    )
                if temporal_variance_mean == 0.0:
                    warnings.append(
                        "Prediction does not vary over time. Content-driven comparison "
                        "would be meaningless — check preprocessing produced real events."
                    )
    else:
        warnings.append("No finite values in the prediction matrix.")

    if n_time == 1:
        warnings.append(
            "Only one time point was predicted; temporal analysis is not possible. "
            "The stimulus may be shorter than one TR."
        )

    return PredictionValidation(
        valid=True,
        shape=shape,
        dtype=dtype,
        nan_count=nan_count,
        inf_count=inf_count,
        finite_count=finite_count,
        constant_vertices=constant_vertices,
        temporal_variance_mean=temporal_variance_mean,
        warnings=tuple(warnings),
        **stats,  # type: ignore[arg-type]
    )
