"""Visual motion and low-level frame features.

WHAT IT DOES
    Walks the video at a fixed sampling rate and produces continuous time-series:
    how much the picture is changing, how bright it is, how colourful, how
    complex.

WHY IT MATTERS
    These are the cheap, dense, objective signals. Unlike semantic labels they
    have no model risk — brightness is brightness. They become columns of the
    Phase 4 feature matrix and are the natural first candidates for correlation
    against neural time-series, because they are defined at every instant.

HOW IT WORKS
    Frames are decoded at `sample_fps`, downscaled, and compared to their
    predecessor. Motion is the mean absolute difference between consecutive
    grayscale frames, normalised to [0, 1].

    Frame differencing rather than dense optical flow: flow is far more
    expensive and measures *direction*, which nothing downstream currently uses.
    Magnitude of change is what editing pace and scene grouping need.

    Plain frame differencing cannot tell a moving subject from a fade: both
    change every pixel. That is separated here rather than left as a caveat, by
    decomposing each frame pair into two independent quantities:

        luminance_shift   |mean(frame) - mean(previous)|
        motion_structural mean |z(frame) - z(previous)|

    where z() standardises a frame to zero mean and unit contrast. Subtracting
    the mean alone is not enough: a fade to black is *multiplicative*, so
    f2 = a * f1 leaves a residual (a-1) x contrast after mere centring. Dividing
    by the frame's own standard deviation removes that too, so a pure fade
    scores zero structural motion no matter how much the picture darkens, while
    a subject moving under constant light is untouched. Frames with no contrast
    left to standardise are excluded rather than amplified — see
    :func:`_structural_change`.

    Measured on the Sintel trailer at the three fade-to-black transitions, this
    is not a theoretical distinction — see docs/content_features.md.

IN:  a video file
OUT: aligned time-series (times, motion, brightness, contrast, saturation, entropy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Sampling rate for the visual signal track. 4 Hz resolves cuts and gestures
#: while decoding ~6x fewer frames than 24 fps native.
DEFAULT_SAMPLE_FPS = 4.0

#: Frames are downscaled before differencing: motion is a global statistic and
#: full resolution adds cost without changing the answer.
ANALYSIS_WIDTH = 160


@dataclass
class VisualSignals:
    """Dense per-sample visual features on a shared time base."""

    times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    motion: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: Frame-to-frame change with the global luminance shift removed.
    motion_structural: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: Magnitude of the global luminance shift alone — fades, flashes, cuts to black.
    luminance_shift: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    brightness: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: 90th-percentile luminance. A bright subject on a dark ground has a low
    #: mean brightness but a high p90; the mean alone calls that frame "dark".
    brightness_p90: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    contrast: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    saturation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    entropy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "visual_times": self.times,
            "motion": self.motion,
            "motion_structural": self.motion_structural,
            "luminance_shift": self.luminance_shift,
            "brightness": self.brightness,
            "brightness_p90": self.brightness_p90,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "entropy": self.entropy,
        }

    def __len__(self) -> int:
        return int(self.times.size)


def compute_visual_signals(
    video_path: Path,
    *,
    duration: float,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> tuple[VisualSignals, list[str]]:
    """Decode the video once and derive every low-level visual signal."""
    warnings: list[str] = []
    try:
        import cv2
    except ImportError:
        return VisualSignals(), ["OpenCV not installed; visual signals unavailable."]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return VisualSignals(), [f"Could not open {video_path.name} for visual analysis."]

    try:
        native_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if native_fps <= 0:
            warnings.append("Frame rate unreported; falling back to timestamp-based sampling.")
        step = max(1, round(native_fps / sample_fps)) if native_fps > 0 else 1

        times: list[float] = []
        motion: list[float] = []
        motion_structural: list[float] = []
        luminance_shift: list[float] = []
        brightness: list[float] = []
        brightness_p90: list[float] = []
        contrast: list[float] = []
        saturation: list[float] = []
        entropy: list[float] = []

        previous_gray: np.ndarray | None = None
        index = 0
        while True:
            ok = capture.grab()
            if not ok:
                break
            if index % step != 0:
                index += 1
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                index += 1
                continue

            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp <= 0 and native_fps > 0:
                timestamp = index / native_fps
            if timestamp > duration + 1.0:
                break

            small = _downscale(cv2, frame, ANALYSIS_WIDTH)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            if previous_gray is None:
                # No predecessor: report 0 rather than inventing a value.
                motion.append(0.0)
                motion_structural.append(0.0)
                luminance_shift.append(0.0)
            else:
                motion.append(float(np.abs(gray - previous_gray).mean()))
                # Standardising each frame removes both the additive (flash,
                # exposure) and multiplicative (fade) components of a global
                # luminance change, leaving only spatial rearrangement.
                motion_structural.append(_structural_change(gray, previous_gray))
                luminance_shift.append(abs(float(gray.mean() - previous_gray.mean())))
            previous_gray = gray

            times.append(float(timestamp))
            brightness.append(float(gray.mean()))
            brightness_p90.append(float(np.percentile(gray, 90)))
            contrast.append(float(gray.std()))
            saturation.append(float(hsv[:, :, 1].astype(np.float32).mean() / 255.0))
            entropy.append(_entropy(gray))
            index += 1
    finally:
        capture.release()

    if not times:
        return VisualSignals(), [*warnings, "No frames could be decoded for visual analysis."]

    signals = VisualSignals(
        times=np.asarray(times, dtype=np.float32),
        motion=np.asarray(motion, dtype=np.float32),
        motion_structural=np.asarray(motion_structural, dtype=np.float32),
        luminance_shift=np.asarray(luminance_shift, dtype=np.float32),
        brightness=np.asarray(brightness, dtype=np.float32),
        brightness_p90=np.asarray(brightness_p90, dtype=np.float32),
        contrast=np.asarray(contrast, dtype=np.float32),
        saturation=np.asarray(saturation, dtype=np.float32),
        entropy=np.asarray(entropy, dtype=np.float32),
    )
    logger.info("Computed visual signals over %d samples", len(signals))
    return signals, warnings


def _downscale(cv2_module: object, frame: np.ndarray, width: int) -> np.ndarray:
    height, original_width = frame.shape[:2]
    if original_width <= width:
        return frame
    scale = width / float(original_width)
    return cv2_module.resize(  # type: ignore[attr-defined]
        frame, (width, max(1, round(height * scale)))
    )


#: Contrast below which a frame carries no structure worth comparing. Dividing
#: by a near-zero standard deviation amplifies sensor and compression noise
#: without limit: measured on the Sintel trailer, frames mid-fade sit at
#: contrast 0.020 against a clip median of 0.142, and standardising them turned
#: a motion of 0.0009 into an apparent structural change of 0.0366 — noise,
#: magnified. The 10th percentile of real contrast is 0.010, so this floor
#: excludes essentially-blank frames without touching dim but real pictures.
STRUCTURE_CONTRAST_FLOOR = 0.025


def _structural_change(gray: np.ndarray, previous: np.ndarray) -> float:
    """Spatial change between two frames, independent of global luminance.

    Both frames are standardised to zero mean and unit contrast, which removes
    the additive *and* multiplicative components of a luminance change, so a
    pure fade scores exactly zero however dark it gets.

    When either frame falls below the contrast floor the comparison is not
    meaningful and returns 0. Standardising a near-blank frame amplifies noise;
    standardising only one of the pair is worse still, because a structured
    frame against a flat one differs by the full unit-variance scale and
    manufactures a large structural change at precisely the fade boundary it is
    meant to ignore. Such a transition is a luminance event, and
    ``luminance_shift`` is where it is reported.
    """
    spread = float(gray.std())
    previous_spread = float(previous.std())
    if spread < STRUCTURE_CONTRAST_FLOOR or previous_spread < STRUCTURE_CONTRAST_FLOOR:
        return 0.0
    current = (gray - gray.mean()) / spread
    earlier = (previous - previous.mean()) / previous_spread
    return float(np.abs(current - earlier).mean())


def _entropy(gray: np.ndarray, bins: int = 64) -> float:
    """Shannon entropy of the intensity histogram.

    A proxy for visual complexity: a flat colour field scores near 0, a detailed
    textured scene scores high.
    """
    histogram, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    total = histogram.sum()
    if total == 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def resample_to(
    times: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Sample a signal onto another time base by linear interpolation.

    Used to put every modality on the shared fusion grid. Interpolation of a
    *content* signal is safe — unlike neural predictions, these are dense
    measurements of a continuous quantity, and the method is stated.
    """
    if times.size == 0 or values.size == 0 or targets.size == 0:
        return np.zeros(targets.size, dtype=np.float32)
    return np.interp(targets, times, values).astype(np.float32)


def interval_mean(
    times: np.ndarray, values: np.ndarray, start: float, end: float
) -> float | None:
    """Mean of a signal within [start, end), or None if no sample falls inside."""
    if times.size == 0:
        return None
    mask = (times >= start) & (times < end)
    finite = mask & np.isfinite(values)
    if not finite.any():
        return None
    return float(values[finite].mean())
