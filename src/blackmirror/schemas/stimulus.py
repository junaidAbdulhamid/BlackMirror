"""Stimulus contract: the typed representation of a piece of content.

A ``StimulusInput`` is created by :func:`build_stimulus`, which validates the
file and computes its SHA-256. Constructing one by hand does not perform I/O —
that keeps the schema pure and unit-testable.
"""

from __future__ import annotations

import datetime as dt
import uuid
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

#: Extensions accepted per media type.
#:
#: Copied from ``tribev2.demo_utils.VALID_SUFFIXES``. BlackMirror declares
#: support for exactly what the backend can process — no more. If a backend
#: supports fewer formats it narrows this at validation time.
SUPPORTED_EXTENSIONS: dict[str, frozenset[str]] = {
    "video": frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm"}),
    "audio": frozenset({".wav", ".mp3", ".flac", ".ogg"}),
    "text": frozenset({".txt"}),
}


class MediaType(StrEnum):
    """Modalities Phase 1 can actually process end-to-end."""

    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"

    @classmethod
    def from_extension(cls, suffix: str) -> MediaType | None:
        """Map a file extension to a media type, or None if unsupported."""
        lowered = suffix.lower()
        for media_type, extensions in SUPPORTED_EXTENSIONS.items():
            if lowered in extensions:
                return cls(media_type)
        return None


class StimulusInput(BaseModel):
    """A validated, content-addressed piece of stimulus content.

    The SHA-256 is what makes an experiment reproducible: it proves which exact
    bytes produced a prediction, independent of filename or path.
    """

    model_config = ConfigDict(frozen=True)

    stimulus_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    path: Path = Field(description="Absolute path to the source file. Never copied into artifacts.")
    filename: str
    media_type: MediaType
    mime_type: str | None = Field(
        default=None,
        description="Best-effort guess from the extension; None when undetermined.",
    )
    file_size_bytes: int = Field(ge=0)
    duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Stimulus duration where it could be determined (ffprobe for media, None for "
            "text). Never inferred from the prediction — that would be circular."
        ),
    )
    sha256: str = Field(min_length=64, max_length=64)
    created_at: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC),
        description="When BlackMirror ingested this stimulus (UTC), not the file's mtime.",
    )

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]

    def summary_line(self) -> str:
        duration = (
            f"{self.duration_seconds:.2f}s" if self.duration_seconds is not None else "unknown"
        )
        return (
            f"{self.filename} ({self.media_type.value}, {duration}, "
            f"{self.file_size_bytes / 1e6:.2f} MB, sha256={self.short_sha})"
        )
