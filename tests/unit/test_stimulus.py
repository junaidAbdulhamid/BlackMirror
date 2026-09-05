"""Stimulus ingestion: the gate between arbitrary files and the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from blackmirror.errors import StimulusValidationError, UnsupportedStimulusError
from blackmirror.inference.validation import build_stimulus
from blackmirror.schemas.stimulus import MediaType


def test_media_type_from_extension_is_case_insensitive() -> None:
    assert MediaType.from_extension(".MP4") is MediaType.VIDEO
    assert MediaType.from_extension(".wav") is MediaType.AUDIO
    assert MediaType.from_extension(".txt") is MediaType.TEXT
    assert MediaType.from_extension(".pdf") is None


def test_build_stimulus_populates_the_contract(video_file: Path) -> None:
    stimulus = build_stimulus(video_file)
    assert stimulus.media_type is MediaType.VIDEO
    assert stimulus.filename == "clip.mp4"
    assert stimulus.file_size_bytes == video_file.stat().st_size
    assert len(stimulus.sha256) == 64
    assert stimulus.path.is_absolute()


def test_build_stimulus_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StimulusValidationError, match="does not exist"):
        build_stimulus(tmp_path / "nope.mp4")


def test_build_stimulus_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(StimulusValidationError, match="not a regular file"):
        build_stimulus(tmp_path)


def test_build_stimulus_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.mp4"
    path.touch()
    with pytest.raises(StimulusValidationError, match="empty"):
        build_stimulus(path)


def test_build_stimulus_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(UnsupportedStimulusError, match="Unsupported stimulus extension"):
        build_stimulus(path)


def test_build_stimulus_respects_backend_media_types(video_file: Path) -> None:
    """A backend must be able to refuse a modality it cannot process."""
    with pytest.raises(UnsupportedStimulusError, match="supports only"):
        build_stimulus(video_file, supported_media_types=frozenset({"text"}))


def test_build_stimulus_rejects_whitespace_only_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.txt"
    path.write_text("   \n\t  ", encoding="utf-8")
    with pytest.raises(StimulusValidationError, match="no content"):
        build_stimulus(path)


def test_build_stimulus_accepts_text(text_file: Path) -> None:
    stimulus = build_stimulus(text_file)
    assert stimulus.media_type is MediaType.TEXT
    # Text has no measurable playback duration; None beats a fabricated number.
    assert stimulus.duration_seconds is None


def test_identical_content_hashes_identically(tmp_path: Path, video_file: Path) -> None:
    """Renaming a file must not change its identity — that is the point of the hash."""
    copy = tmp_path / "renamed.mp4"
    copy.write_bytes(video_file.read_bytes())
    assert build_stimulus(copy).sha256 == build_stimulus(video_file).sha256
