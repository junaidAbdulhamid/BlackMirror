"""Media inspection and extraction.

The lowest layer of Phase 4: turn a file on disk into the facts and derived
assets every other pipeline needs — container metadata, a normalised audio
track, and representative frames.

Deliberately shells out to ffprobe/ffmpeg rather than decoding in Python. They
are the reference implementations, already required by Phase 1, and they handle
the container zoo correctly.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from blackmirror.content.schemas import VideoMetadata
from blackmirror.errors import BlackMirrorError
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Downstream audio analysis assumes mono at this rate. Fixing it here means
#: every signal computation sees the same representation regardless of source.
AUDIO_SAMPLE_RATE = 16_000


class MediaError(BlackMirrorError):
    """Media could not be inspected or decoded."""


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise MediaError(
            f"`{name}` is not on PATH. Content analysis needs ffmpeg/ffprobe "
            f"(`brew install ffmpeg`)."
        )
    return path


def _run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    except subprocess.CalledProcessError as exc:
        raise MediaError(
            f"{Path(command[0]).name} failed ({exc.returncode}): "
            f"{(exc.stderr or '').strip()[:400]}"
        ) from exc
    except subprocess.SubprocessError as exc:
        raise MediaError(f"{Path(command[0]).name} failed: {exc}") from exc


def probe_media(path: Path) -> VideoMetadata:
    """Read container metadata with ffprobe.

    Everything is read rather than assumed: a stimulus may be audio-only, may
    have variable frame rate, or may not report a frame count.
    """
    completed = _run(
        [
            _tool("ffprobe"), "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned unparsable JSON for {path.name}") from exc

    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _as_float(probe.get("format", {}).get("duration"))
    if duration is None and video is not None:
        duration = _as_float(video.get("duration"))
    if duration is None or duration <= 0:
        raise MediaError(f"Could not determine a positive duration for {path.name}")

    fps = _parse_rate(video.get("avg_frame_rate") if video else None) or _parse_rate(
        video.get("r_frame_rate") if video else None
    )
    frame_count = _as_int(video.get("nb_frames")) if video else None
    if frame_count is None and fps is not None:
        # nb_frames is absent in many containers; derive it so shot detection
        # can still report frame indices.
        frame_count = round(duration * fps)

    return VideoMetadata(
        duration_seconds=duration,
        width=_as_int(video.get("width")) if video else None,
        height=_as_int(video.get("height")) if video else None,
        fps=fps,
        frame_count=frame_count,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=_as_int(audio.get("sample_rate")) if audio else None,
        audio_channels=_as_int(audio.get("channels")) if audio else None,
        has_video=video is not None,
        has_audio=audio is not None,
    )


def extract_audio(source: Path, destination: Path) -> Path:
    """Decode audio to mono 16 kHz PCM WAV.

    Written to the run's working directory rather than the artifact store: it is
    a large, fully reproducible intermediate, and Phase 1's rule is that media
    is referenced, not duplicated into artifacts.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _tool("ffmpeg"), "-y", "-v", "error",
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(destination),
        ]
    )
    if not destination.exists() or destination.stat().st_size == 0:
        raise MediaError(f"Audio extraction produced no data for {source.name}")
    return destination


def extract_frame(source: Path, timestamp: float, destination: Path, *, width: int = 512) -> Path:
    """Grab a single frame at `timestamp`, scaled to a fixed width.

    `-ss` before `-i` seeks by keyframe, which is fast and accurate enough for
    representative frames; downscaling keeps OCR and CLIP cheap.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _tool("ffmpeg"), "-y", "-v", "error",
            "-ss", f"{max(0.0, timestamp):.3f}",
            "-i", str(source),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            str(destination),
        ],
        timeout=120,
    )
    if not destination.exists() or destination.stat().st_size == 0:
        raise MediaError(f"Could not extract a frame at {timestamp:.2f}s from {source.name}")
    return destination


def sample_times(start: float, end: float, count: int) -> list[float]:
    """Evenly spaced interior sample points for an interval.

    Interior rather than edge-inclusive: a frame taken exactly on a cut often
    lands on a dissolve or a black frame, which describes neither shot.
    """
    if count <= 0 or end <= start:
        return []
    if count == 1:
        return [start + (end - start) / 2.0]
    step = (end - start) / (count + 1)
    return [start + step * (i + 1) for i in range(count)]


def _as_float(value: object) -> float | None:
    if not isinstance(value, str | int | float):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: object) -> int | None:
    if not isinstance(value, str | int | float):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(value: object) -> float | None:
    """Parse ffprobe's `num/den` frame-rate notation."""
    if not isinstance(value, str) or "/" not in value:
        return _as_float(value)
    numerator, _, denominator = value.partition("/")
    num = _as_float(numerator)
    den = _as_float(denominator)
    if num is None or den is None or den == 0:
        return None
    rate = num / den
    return rate if rate > 0 else None
