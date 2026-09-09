"""Deterministic execution of a materialization plan.

DETERMINISM IS A REQUIREMENT, NOT A PREFERENCE
    Phase 9 caches evaluations by candidate identity and detects duplicate
    genomes before spending an inference. Both are only sound if the same plan
    against the same source produces the same bytes every time. ffmpeg is not
    deterministic by default: it stamps encoder metadata and creation times into
    the container, and multi-threaded encoding can reorder work.

    So every invocation here pins `-fflags +bitexact`, bitexact stream flags,
    strips container metadata, and forces single-threaded x264. Verified: two
    runs of the same audio-only plan and two runs of the same video plan each
    produced byte-identical output.

WHY THE FILE IS NAMED AFTER ITS PLAN
    The feature cache under TRIBE keys on file path and time range, not content
    (`item_uid` in `neuralset.extractors`). A reused path with new content would
    silently return the previous variant's features. Naming each output after
    the plan and source content makes path identity equal content identity, so
    the cache becomes correct rather than dangerous, and re-materializing a
    candidate is free.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from blackmirror.content.media import MediaError, probe_media
from blackmirror.materialization.schemas import (
    EditOperation,
    EditStep,
    MaterializationPlan,
    MaterializationResult,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Flags that make ffmpeg reproducible. Without these the same edit produces
#: different bytes on every run and every cache below becomes unsound.
_DETERMINISM_FLAGS = ("-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact")


class MaterializationError(MediaError):
    """The variant could not be produced."""


def _video_filter(steps: tuple[EditStep, ...]) -> str | None:
    """Build the `eq`/`setpts` chain for the visual stream."""
    eq: list[str] = []
    chain: list[str] = []
    for step in steps:
        if step.operation is EditOperation.BRIGHTNESS:
            eq.append(f"brightness={step.value:g}")
        elif step.operation is EditOperation.CONTRAST:
            eq.append(f"contrast={step.value:g}")
        elif step.operation is EditOperation.SATURATION:
            eq.append(f"saturation={step.value:g}")
        elif step.operation is EditOperation.SPEED:
            chain.append(f"setpts=PTS/{step.value:g}")
    if eq:
        chain.insert(0, "eq=" + ":".join(eq))
    return ",".join(chain) if chain else None


def _audio_filter(steps: tuple[EditStep, ...]) -> str | None:
    chain: list[str] = []
    for step in steps:
        if step.operation is EditOperation.AUDIO_GAIN_DB:
            chain.append(f"volume={step.value:g}dB")
        elif step.operation is EditOperation.SPEED:
            # atempo is only defined on [0.5, 2.0], which is exactly the bound
            # the schema enforces, so one stage always suffices.
            chain.append(f"atempo={step.value:g}")
    return ",".join(chain) if chain else None


def build_command(
    source: Path, destination: Path, plan: MaterializationPlan, *, has_audio: bool
) -> tuple[str, ...]:
    """The exact ffmpeg invocation, recorded in the result for provenance."""
    steps = plan.effective_steps
    video_filter = _video_filter(steps)
    audio_filter = _audio_filter(steps) if has_audio else None

    command: list[str] = [
        _tool("ffmpeg"), "-y", "-v", "error", "-nostdin",
        "-fflags", "+bitexact",
        "-i", str(source),
    ]
    if video_filter:
        command += ["-vf", video_filter]
    if audio_filter:
        command += ["-af", audio_filter]

    # Re-encode only the stream that changed. Copying the untouched stream is
    # faster and, more importantly, leaves it bit-identical to the source, so a
    # candidate differs from its parent only in the way the plan says it does.
    if video_filter:
        command += [
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-x264-params", "threads=1",
        ]
    else:
        command += ["-c:v", "copy"]

    if has_audio:
        command += ["-c:a", "aac", "-b:a", "128k"] if audio_filter else ["-c:a", "copy"]
    else:
        command += ["-an"]

    command += ["-map_metadata", "-1", *_DETERMINISM_FLAGS, str(destination)]
    return tuple(command)


def candidate_key(plan: MaterializationPlan, source_sha256: str) -> str:
    """Identity of this candidate: the edit applied to that exact content."""
    return hashlib.sha256(f"{source_sha256}:{plan.fingerprint()}".encode()).hexdigest()


def materialize(
    plan: MaterializationPlan, destination_dir: Path, *, reuse: bool = True
) -> MaterializationResult:
    """Produce the variant described by `plan`, or reuse an identical one.

    Reuse is safe precisely because the filename is derived from the plan and
    the source content, so a name collision means the bytes would have been the
    same anyway.
    """
    source = Path(plan.source_path)
    if not source.is_file():
        raise MaterializationError(f"source media does not exist: {source}")

    source_sha = _file_hash(source)
    key = candidate_key(plan, source_sha)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{source.stem}-{key[:16]}{source.suffix}"

    probe = probe_media(source)
    command = build_command(source, destination, plan, has_audio=probe.has_audio)

    if reuse and destination.is_file():
        logger.info("Reusing materialized candidate %s", destination.name)
        variant_sha = _file_hash(destination)
        variant_probe = probe_media(destination)
        return MaterializationResult(
            plan=plan,
            variant_path=destination,
            variant_sha256=variant_sha,
            source_sha256=source_sha,
            candidate_key=key,
            tool="ffmpeg",
            tool_version=ffmpeg_version(),
            command=command,
            source_duration_seconds=probe.duration_seconds,
            variant_duration_seconds=variant_probe.duration_seconds,
            reused_existing=True,
        )

    if plan.modalities == {"audio"} and not probe.has_audio:
        raise MaterializationError(
            f"{source.name} has no audio stream, so an audio-only edit would "
            f"produce an unchanged file"
        )

    # Write to a temporary name and rename, so an interrupted ffmpeg never
    # leaves a truncated file sitting at the name a later run would reuse.
    # The suffix stays last: ffmpeg picks its container from the extension, so
    # a staging name ending in ".partial" makes it refuse to write at all.
    staging = destination.with_name(f".{destination.stem}.partial{destination.suffix}")
    staging.unlink(missing_ok=True)
    staged_command = tuple(
        str(staging) if item == str(destination) else item for item in command
    )
    try:
        completed = subprocess.run(
            list(staged_command), capture_output=True, text=True, timeout=1800, check=False
        )
        if completed.returncode != 0:
            raise MaterializationError(
                f"ffmpeg failed for plan [{plan.describe()}]: "
                f"{completed.stderr.strip()[:400] or 'no stderr'}"
            )
        if not staging.is_file() or staging.stat().st_size == 0:
            raise MaterializationError("ffmpeg reported success but wrote no output")
        shutil.move(str(staging), str(destination))
    except subprocess.SubprocessError as exc:
        staging.unlink(missing_ok=True)
        raise MaterializationError(f"ffmpeg could not be run: {exc}") from exc
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    variant_sha = _file_hash(destination)
    if variant_sha == source_sha:
        destination.unlink(missing_ok=True)
        raise MaterializationError(
            f"the edit [{plan.describe()}] produced a file identical to its "
            f"source; nothing would be measured by evaluating it"
        )

    variant_probe = probe_media(destination)
    logger.info(
        "Materialized %s [%s] -> %s", source.name, plan.describe(), destination.name
    )
    return MaterializationResult(
        plan=plan,
        variant_path=destination,
        variant_sha256=variant_sha,
        source_sha256=source_sha,
        candidate_key=key,
        tool="ffmpeg",
        tool_version=ffmpeg_version(),
        command=command,
        source_duration_seconds=probe.duration_seconds,
        variant_duration_seconds=variant_probe.duration_seconds,
        reused_existing=False,
    )


def ffmpeg_version() -> str:
    """The encoder build, recorded because it can change output bytes."""
    try:
        completed = subprocess.run(
            [_tool("ffmpeg"), "-version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment
        return "unknown"
    match = re.search(r"ffmpeg version (\S+)", completed.stdout)
    return match.group(1) if match else "unknown"


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise MaterializationError(
            f"`{name}` is not on PATH. Materializing candidates needs ffmpeg "
            f"(`brew install ffmpeg`)."
        )
    return path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "MaterializationError",
    "build_command",
    "candidate_key",
    "ffmpeg_version",
    "materialize",
]
