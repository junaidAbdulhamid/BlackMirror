"""Shared test fixtures.

Every fixture here is model-free: the unit suite must run without the TRIBE
checkpoint, gated HuggingFace access, or a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from blackmirror.config.settings import BackendName, DevicePreference, Settings
from blackmirror.inference.backend import RawPrediction, SegmentRecord
from blackmirror.schemas.stimulus import MediaType, StimulusInput
from blackmirror.utils.hashing import sha256_file

# Small enough to keep tests fast; the shape semantics are what matter.
TEST_VERTICES_PER_HEMISPHERE = 8
TEST_VERTEX_COUNT = TEST_VERTICES_PER_HEMISPHERE * 2


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointed entirely at a temp directory, using the mock backend."""
    return Settings(
        backend=BackendName.MOCK,
        device=DevicePreference.CPU,
        artifact_dir=tmp_path / "artifacts",
        model_cache_dir=tmp_path / "cache",
        log_level="WARNING",
    )


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    """A non-empty file with a supported video extension.

    Contents are arbitrary bytes: stimulus validation checks existence, size,
    readability and extension — it does not decode the container.
    """
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x01\x02\x03" * 256)
    return path


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    path = tmp_path / "script.txt"
    path.write_text("A short script used as a text stimulus.", encoding="utf-8")
    return path


@pytest.fixture
def stimulus(video_file: Path) -> StimulusInput:
    return StimulusInput(
        path=video_file,
        filename=video_file.name,
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        file_size_bytes=video_file.stat().st_size,
        duration_seconds=12.0,
        sha256=sha256_file(video_file),
    )


@pytest.fixture
def raw_prediction() -> RawPrediction:
    """A small, contiguous, well-formed prediction."""
    n_time = 6
    rng = np.random.default_rng(0)
    array = rng.normal(size=(n_time, TEST_VERTEX_COUNT)).astype(np.float32)
    segments = [
        SegmentRecord(
            index=i,
            start_seconds=float(i),
            duration_seconds=1.0,
            n_events=2,
            event_types=("Word",),
        )
        for i in range(n_time)
    ]
    return RawPrediction(
        array=array,
        segments=segments,
        tr_seconds=1.0,
        surface_space="fsaverage5",
        vertices_per_hemisphere=TEST_VERTICES_PER_HEMISPHERE,
        hemodynamic_offset_seconds=5.0,
    )


@pytest.fixture
def gappy_prediction() -> RawPrediction:
    """A prediction whose event-free segments were dropped.

    This models TRIBE's default ``remove_empty_segments=True`` behaviour, which
    is the case downstream code is most likely to get wrong.
    """
    starts = [0.0, 1.0, 2.0, 7.0, 8.0]
    array = np.arange(len(starts) * TEST_VERTEX_COUNT, dtype=np.float32).reshape(
        len(starts), TEST_VERTEX_COUNT
    )
    segments = [
        SegmentRecord(index=i, start_seconds=s, duration_seconds=1.0, n_events=1)
        for i, s in enumerate(starts)
    ]
    return RawPrediction(
        array=array,
        segments=segments,
        tr_seconds=1.0,
        surface_space="fsaverage5",
        vertices_per_hemisphere=TEST_VERTICES_PER_HEMISPHERE,
        hemodynamic_offset_seconds=5.0,
        segments_were_filtered=True,
    )


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop a developer's real .env or HF token from leaking into tests."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name in list(dict(__import__("os").environ)):
        if name.startswith("BLACKMIRROR_"):
            monkeypatch.delenv(name, raising=False)
    yield
