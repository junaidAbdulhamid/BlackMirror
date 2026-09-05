"""Working directory for reproducible content-analysis intermediates.

WHY THIS EXISTS
    Phase 1's rule is that artifacts *reference* media rather than duplicating
    it — a run directory records a path and a SHA-256, never a copy. Decoded
    audio and extracted OCR frames broke that rule: they are derived media, they
    can be regenerated from the source at any time, and for a 10-minute video
    the audio alone is ~19 MB per run.

    They are also expensive enough to be worth keeping, so deleting them
    outright would trade one problem for another.

    The resolution is that they are a *cache*, not an artifact. They live under
    the model cache directory, keyed by the stimulus SHA-256, so:

      * the artifact store contains no duplicated media
      * two runs over the same stimulus share one decode
      * the whole directory is disposable — deleting it costs only time

    Keyframes are the exception and stay in the run directory: they are
    referenced by `SceneSegment.keyframe_path`, served by the API, and shown in
    the UI, so they are genuine artifacts rather than intermediates.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

WORKSPACE_DIR = "content_workspace"


@dataclass(frozen=True)
class ContentWorkspace:
    """Regenerable intermediates for one stimulus, shared across runs."""

    root: Path
    stimulus_sha256: str

    @classmethod
    def for_stimulus(cls, cache_root: Path, stimulus_sha256: str) -> ContentWorkspace:
        # Sharded by hash prefix so a large cache does not become one flat
        # directory with thousands of entries.
        root = cache_root / WORKSPACE_DIR / stimulus_sha256[:2] / stimulus_sha256[:16]
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root, stimulus_sha256=stimulus_sha256)

    @property
    def audio_path(self) -> Path:
        """Decoded mono 16 kHz audio. Regenerable from the source at any time."""
        return self.root / "audio.wav"

    @property
    def ocr_frame_dir(self) -> Path:
        """Densely-sampled frames used only for OCR; not referenced by artifacts."""
        directory = self.root / "ocr_frames"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def has_audio(self) -> bool:
        return self.audio_path.exists() and self.audio_path.stat().st_size > 0

    def size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def clear(self) -> None:
        """Discard everything. Safe: every file here can be regenerated."""
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            logger.info("Cleared content workspace %s", self.root)


def clear_all(cache_root: Path) -> int:
    """Remove every stimulus workspace. Returns the bytes reclaimed."""
    root = cache_root / WORKSPACE_DIR
    if not root.exists():
        return 0
    reclaimed = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    shutil.rmtree(root, ignore_errors=True)
    logger.info("Cleared all content workspaces (%.1f MB)", reclaimed / 1e6)
    return reclaimed
