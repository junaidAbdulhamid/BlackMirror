"""Content pipeline components: caching identity, metrics, features, OCR dedup.

No model downloads and no video decoding — these test the logic that surrounds
the models, which is where the bugs that matter live.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from blackmirror.content.audio.signals import AudioSignals, _merge_short_segments, _runs_to_segments
from blackmirror.content.features import FEATURE_NAMES, build_feature_matrix, summarise_columns
from blackmirror.content.metrics import _covered, compute_metrics
from blackmirror.content.pipeline import (
    ContentAnalysisConfig,
    _align_to_neural,
    _upstream_fingerprints,
    compute_cache_key,
)
from blackmirror.content.schemas import (
    AnalysisSource,
    AudioSegment,
    AudioSegmentType,
    CallToAction,
    ContentEvent,
    ContentEventType,
    Provenance,
    SceneSegment,
    ShotSegment,
    TextOverlay,
    TranscriptSegment,
    WordTiming,
)
from blackmirror.content.timeline import ContentTimeline
from blackmirror.content.visual.motion import VisualSignals, interval_mean, resample_to
from blackmirror.content.visual.ocr import FrameText, _merge, _normalise, _similar
from blackmirror.content.visual.shots import editing_pace

DSP = Provenance(source=AnalysisSource.SIGNAL_DSP)
ASR = Provenance(source=AnalysisSource.TRIBE_EVENTS)
OCR = Provenance(source=AnalysisSource.RAPID_OCR)


# --- Caching identity -----------------------------------------------------


class TestCacheKey:
    BASE: ClassVar[dict[str, Any]] = {
        "stimulus_sha256": "a" * 64,
        "models": {"visual_semantics": "clip", "ocr": "rapidocr"},
        "config": ContentAnalysisConfig(),
    }

    def test_identical_inputs_give_identical_keys(self) -> None:
        assert compute_cache_key(**self.BASE) == compute_cache_key(**self.BASE)

    def test_different_stimulus_changes_the_key(self) -> None:
        other = {**self.BASE, "stimulus_sha256": "b" * 64}
        assert compute_cache_key(**other) != compute_cache_key(**self.BASE)

    def test_different_model_changes_the_key(self) -> None:
        """Swapping a model must invalidate the cache — the results differ."""
        other = {**self.BASE, "models": {"visual_semantics": "other", "ocr": "rapidocr"}}
        assert compute_cache_key(**other) != compute_cache_key(**self.BASE)

    def test_different_configuration_changes_the_key(self) -> None:
        other = {**self.BASE, "config": ContentAnalysisConfig(keyframes_per_shot=1)}
        assert compute_cache_key(**other) != compute_cache_key(**self.BASE)

    def test_model_ordering_does_not_change_the_key(self) -> None:
        other = {**self.BASE, "models": {"ocr": "rapidocr", "visual_semantics": "clip"}}
        assert compute_cache_key(**other) == compute_cache_key(**self.BASE)

    def test_upstream_analytics_change_the_key(self, tmp_path: Path) -> None:
        analytics = tmp_path / "analytics"
        analytics.mkdir()
        metadata = analytics / "metadata.json"
        metadata.write_text('{"events": []}', encoding="utf-8")
        before = compute_cache_key(
            **self.BASE, upstream_fingerprints=_upstream_fingerprints(tmp_path)
        )
        metadata.write_text('{"events": [{"event_id": "new"}]}', encoding="utf-8")
        after = compute_cache_key(
            **self.BASE, upstream_fingerprints=_upstream_fingerprints(tmp_path)
        )
        assert before != after


# --- Metrics --------------------------------------------------------------


class TestMetrics:
    def test_overlapping_spans_are_merged_not_summed(self) -> None:
        """Two overlapping speech segments cannot exceed the stimulus length."""
        assert _covered([(0, 5), (3, 8)], 10) == 8.0
        assert _covered([(0, 5), (5, 10)], 10) == 10.0
        assert _covered([], 10) == 0.0

    def test_spans_are_clamped_to_the_stimulus(self) -> None:
        assert _covered([(-2, 5), (8, 20)], 10) == 7.0

    def test_metrics_from_a_small_stimulus(self) -> None:
        metrics = compute_metrics(
            duration=10.0,
            shots=[
                ShotSegment(index=0, start_time=0, end_time=4, duration=4),
                ShotSegment(index=1, start_time=4, end_time=10, duration=6),
            ],
            scenes=[SceneSegment(index=0, start_time=0, end_time=10, duration=10)],
            transcript=[
                TranscriptSegment(
                    index=0, start_time=0, end_time=5, text="one two three",
                    words=(
                        WordTiming(text="one", start_time=0, end_time=1),
                        WordTiming(text="two", start_time=1, end_time=2),
                        WordTiming(text="three", start_time=2, end_time=3),
                    ),
                    provenance=ASR,
                )
            ],
            audio_segments=[
                AudioSegment(segment_type=AudioSegmentType.MUSIC, start_time=0, end_time=10,
                             mean_energy=0.2, provenance=DSP)
            ],
            overlays=[TextOverlay(text="HI", start_time=1, end_time=2, provenance=OCR)],
            calls_to_action=[
                CallToAction(text="Buy now", start_time=8, end_time=9, modality="speech",
                             confidence=0.9, provenance=ASR)
            ],
            events=[],
            visual_signals=VisualSignals(),
            audio_signals=AudioSignals(),
        )
        assert metrics.shot_count == 2
        assert metrics.mean_shot_duration == 5.0
        # Two shots is ONE cut, over 10s -> 6 cuts/minute.
        assert metrics.cuts_per_minute == 6.0
        assert metrics.speech_fraction == 0.5
        assert metrics.music_fraction == 1.0
        assert metrics.word_count == 3
        assert metrics.cta_count == 1
        assert metrics.first_cta_time == 8.0

    def test_absent_signals_report_none_not_zero(self) -> None:
        """A metric that cannot be derived must be absent, never guessed."""
        metrics = compute_metrics(
            duration=5.0, shots=[], scenes=[], transcript=[], audio_segments=[],
            overlays=[], calls_to_action=[], events=[],
            visual_signals=VisualSignals(), audio_signals=AudioSignals(),
        )
        assert metrics.mean_motion is None
        assert metrics.mean_audio_energy is None
        assert metrics.mean_shot_duration is None
        assert metrics.shot_count == 0

    def test_editing_pace_counts_cuts_not_shots(self) -> None:
        shots = [
            ShotSegment(index=i, start_time=i * 2.0, end_time=(i + 1) * 2.0, duration=2.0)
            for i in range(5)
        ]
        pace = editing_pace(shots, 10.0)
        assert pace["shot_count"] == 5.0
        assert pace["mean_shot_duration"] == 2.0
        assert pace["cuts_per_minute"] == pytest.approx(24.0)

    def test_median_resists_one_long_shot(self) -> None:
        shots = [
            ShotSegment(index=0, start_time=0, end_time=1, duration=1),
            ShotSegment(index=1, start_time=1, end_time=2, duration=1),
            ShotSegment(index=2, start_time=2, end_time=32, duration=30),
        ]
        pace = editing_pace(shots, 32.0)
        assert pace["median_shot_duration"] == 1.0
        assert pace["mean_shot_duration"] > 10.0

    def test_audio_runs_cover_the_stimulus_origin_and_merge_without_gaps(self) -> None:
        times = np.array([0.0125, 0.0225, 0.0325], dtype=np.float32)
        kinds = np.array(["music", "silence", "music"], dtype=object)
        segments = _merge_short_segments(
            _runs_to_segments(times, kinds, np.ones(3), duration=0.05)
        )
        assert segments[0].start_time == 0.0
        assert segments[-1].end_time == 0.05
        assert all(
            left.end_time == right.start_time
            for left, right in pairwise(segments)
        )


# --- Feature matrix -------------------------------------------------------


class TestFeatureMatrix:
    def test_shape_and_column_order(self) -> None:
        matrix, names, times = build_feature_matrix(
            duration=4.0,
            visual_signals=VisualSignals(
                times=np.array([0, 1, 2, 3], dtype=np.float32),
                motion=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                brightness=np.zeros(4, dtype=np.float32),
                contrast=np.zeros(4, dtype=np.float32),
                saturation=np.zeros(4, dtype=np.float32),
                entropy=np.zeros(4, dtype=np.float32),
            ),
            audio_signals=AudioSignals(),
            events=[],
            shot_times=[2.0],
            rate_hz=4.0,
        )
        assert matrix.shape == (16, len(FEATURE_NAMES))
        assert names == FEATURE_NAMES
        assert times.size == 16
        # Column order is a contract Phase 5 will rely on when diffing runs.
        assert names[0] == "motion"

    def test_shot_changes_are_impulses(self) -> None:
        matrix, names, _ = build_feature_matrix(
            duration=4.0,
            visual_signals=VisualSignals(),
            audio_signals=AudioSignals(),
            events=[],
            shot_times=[2.0],
            rate_hz=4.0,
        )
        column = matrix[:, names.index("shot_change")]
        assert column.sum() == 1.0

    def test_categorical_columns_are_binary(self) -> None:
        events = [
            ContentEvent(
                event_id="a", event_type=ContentEventType.SPEECH_SEGMENT,
                start_time=0, end_time=4, speech_text="hello",
                on_screen_text=("SALE",), audio_description="music",
            )
        ]
        matrix, names, _ = build_feature_matrix(
            duration=4.0, visual_signals=VisualSignals(), audio_signals=AudioSignals(),
            events=events, shot_times=[], rate_hz=2.0,
        )
        for name in ("speech_present", "music_present", "text_present"):
            column = matrix[:, names.index(name)]
            assert set(np.unique(column)) <= {0.0, 1.0}
            assert column.max() == 1.0

    def test_zero_duration_yields_an_empty_matrix(self) -> None:
        matrix, _, times = build_feature_matrix(
            duration=0.0, visual_signals=VisualSignals(), audio_signals=AudioSignals(),
            events=[], shot_times=[],
        )
        assert matrix.shape[0] == 0
        assert times.size == 0

    def test_column_summary(self) -> None:
        matrix = np.array([[0.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        summary = summarise_columns(matrix, ("a", "b"))
        assert summary["a"]["max"] == 1.0
        assert summary["b"]["mean"] == 2.0


# --- Signal helpers -------------------------------------------------------


class TestSignalHelpers:
    def test_resample_is_linear_interpolation(self) -> None:
        out = resample_to(
            np.array([0.0, 2.0]), np.array([0.0, 10.0]), np.array([0.0, 1.0, 2.0])
        )
        np.testing.assert_allclose(out, [0.0, 5.0, 10.0])

    def test_resample_handles_empty_input(self) -> None:
        out = resample_to(np.zeros(0), np.zeros(0), np.array([1.0, 2.0]))
        assert out.shape == (2,)
        assert not out.any()

    def test_interval_mean_uses_samples_inside(self) -> None:
        times = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert interval_mean(times, values, 1.0, 3.0) == 2.5

    def test_interval_mean_does_not_invent_coverage(self) -> None:
        """An arbitrarily distant sample is not evidence for an empty interval."""
        times = np.array([0.0, 10.0])
        values = np.array([5.0, 50.0])
        assert interval_mean(times, values, 1.0, 1.2) is None


def test_interval_contract_rejects_reversed_and_inconsistent_ranges() -> None:
    with pytest.raises(ValueError, match="end_time"):
        TextOverlay(text="x", start_time=2, end_time=1, provenance=OCR)
    with pytest.raises(ValueError, match="duration"):
        ShotSegment(index=0, start_time=0, end_time=2, duration=1)


# --- OCR deduplication ----------------------------------------------------


class TestOcrDeduplication:
    def test_normalisation_strips_case_and_punctuation(self) -> None:
        assert _normalise("  50%  OFF! ") == "50 off"

    def test_ocr_jitter_is_still_similar(self) -> None:
        assert _similar(_normalise("50% OFF"), _normalise("5O% OFF")) > 0.7

    def test_consecutive_repeats_become_one_overlay(self) -> None:
        """Six detections of one title must not become six overlays."""
        detections = [
            FrameText(time=t, text="50% OFF", confidence=0.9) for t in (1.0, 1.5, 2.0, 2.5)
        ]
        overlays = _merge(detections, duration=10.0, frame_span=0.5)
        assert len(overlays) == 1
        assert overlays[0].occurrences == 4
        assert overlays[0].start_time == 1.0
        assert overlays[0].end_time == 3.0

    def test_the_same_text_much_later_is_a_new_overlay(self) -> None:
        detections = [
            FrameText(time=1.0, text="BUY NOW", confidence=0.9),
            FrameText(time=9.0, text="BUY NOW", confidence=0.9),
        ]
        assert len(_merge(detections, duration=12.0, frame_span=0.5)) == 2

    def test_highest_confidence_rendering_wins(self) -> None:
        detections = [
            FrameText(time=1.0, text="5O% OFF", confidence=0.6),
            FrameText(time=1.5, text="50% OFF", confidence=0.95),
        ]
        overlays = _merge(detections, duration=5.0, frame_span=0.5)
        assert overlays[0].text == "50% OFF"

    def test_distinct_text_is_kept_separate(self) -> None:
        detections = [
            FrameText(time=1.0, text="50% OFF", confidence=0.9),
            FrameText(time=1.2, text="FREE SHIPPING", confidence=0.9),
        ]
        assert len(_merge(detections, duration=5.0, frame_span=0.5)) == 2


# --- Storage round-trip ---------------------------------------------------


def test_content_store_round_trip(tmp_path: Path) -> None:
    from blackmirror.content.schemas import (
        ContentAnalysisMetadata,
        ContentAnalysisResult,
        ContentMetrics,
        VideoMetadata,
    )
    from blackmirror.content.storage import ContentStore

    store = ContentStore(tmp_path)
    result = ContentAnalysisResult(
        run_id="run-1",
        stimulus_id="s1",
        stimulus_filename="clip.mp4",
        media=VideoMetadata(duration_seconds=4.0),
        metrics=ContentMetrics(duration_seconds=4.0, shot_count=1, scene_count=1),
        metadata=ContentAnalysisMetadata(
            analysis_version="1.0", stimulus_sha256="a" * 64, cache_key="k1"
        ),
    )
    store.write(result, {"features": np.zeros((4, 3), dtype=np.float32)})

    assert store.exists("run-1")
    reloaded = store.read("run-1")
    assert reloaded.run_id == "run-1"
    assert store.read_arrays("run-1")["features"].shape == (4, 3)

    # Cache hit only when the key matches.
    assert store.cached_result("run-1", "k1") is not None
    assert store.cached_result("run-1", "different") is None


def test_alignment_reads_real_phase3_array_names(tmp_path: Path) -> None:
    analytics = tmp_path / "runs" / "run-1" / "analytics"
    analytics.mkdir(parents=True)
    (analytics / "metadata.json").write_text('{"events": []}', encoding="utf-8")
    times = np.arange(12, dtype=np.float64)
    np.savez_compressed(
        analytics / "timeseries.npz",
        times=times,
        global_mean=np.sin(times),
        change_magnitude=np.cos(times),
    )
    content_times = np.linspace(0, 11, 48, dtype=np.float32)
    matrix = np.sin(content_times)[:, None]
    _, correlations = _align_to_neural(
        artifact_root=tmp_path,
        run_id="run-1",
        timeline=ContentTimeline([]),
        matrix=matrix,
        feature_names=("motion",),
        feature_times=content_times,
        config=ContentAnalysisConfig(),
        warnings=[],
    )
    assert correlations
    assert {item.neural_metric for item in correlations} <= {
        "global_mean",
        "change_magnitude",
    }
