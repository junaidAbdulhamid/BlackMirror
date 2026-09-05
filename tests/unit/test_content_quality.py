"""Abstention, evidence-based labelling, and the workspace contract.

These cover the fixes made after the Phase 4 review: every one of them exists
because a measured failure showed the previous behaviour was confidently wrong
rather than merely imprecise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blackmirror.content.audio.diarization import (
    MIN_SEPARATION,
    _cluster,
    _greedy_cluster,
    _separation,
)
from blackmirror.content.audio.tagging import (
    AudioTagWindow,
    events_from_tags,
    probability_series,
    segments_from_tags,
)
from blackmirror.content.language.semantics import (
    _marker_role,
    _positional_prior,
    _structural_role,
)
from blackmirror.content.schemas import (
    AnalysisSource,
    AudioSegmentType,
    FrameKind,
    Provenance,
    StructuralRole,
    TranscriptSegment,
)
from blackmirror.content.visual.objects import Detection, objects_in_interval
from blackmirror.content.visual.semantics import (
    SETTING_MARGIN_ABSTAIN,
    classify_frame_kind,
    margin_confidence,
)
from blackmirror.content.workspace import ContentWorkspace, clear_all

ASR = Provenance(source=AnalysisSource.TRIBE_EVENTS)


# --- Visual abstention ----------------------------------------------------


class TestFrameKind:
    """Frame kind comes from measured evidence, not from a CLIP prompt.

    Measured on real frames: a title card scored entropy 0.26 / saturation 0.00,
    a BLACK frame 0.00 / 0.00, and a DARK PHOTOGRAPHIC frame 0.47 / 0.25.
    Entropy alone cannot separate the last two — saturation and OCR text can.
    """

    def test_title_card_with_text_is_graphic(self) -> None:
        assert (
            classify_frame_kind(entropy=0.26, saturation=0.0, has_text=True)
            is FrameKind.GRAPHIC
        )

    def test_black_frame_is_blank(self) -> None:
        assert (
            classify_frame_kind(entropy=0.0, saturation=0.0, has_text=False)
            is FrameKind.BLANK
        )

    def test_dark_photographic_frame_is_not_mistaken_for_a_graphic(self) -> None:
        """The case naive entropy thresholding gets wrong."""
        assert (
            classify_frame_kind(entropy=0.47, saturation=0.252, has_text=False)
            is FrameKind.PHOTOGRAPHIC
        )

    def test_detailed_frame_is_photographic(self) -> None:
        assert (
            classify_frame_kind(entropy=4.72, saturation=0.386, has_text=False)
            is FrameKind.PHOTOGRAPHIC
        )

    def test_monochrome_low_detail_is_graphic_even_without_ocr(self) -> None:
        assert (
            classify_frame_kind(entropy=0.5, saturation=0.005, has_text=False)
            is FrameKind.GRAPHIC
        )

    def test_missing_evidence_is_unknown_not_a_guess(self) -> None:
        assert (
            classify_frame_kind(entropy=None, saturation=None, has_text=False)
            is FrameKind.UNKNOWN
        )


class TestMarginConfidence:
    """Confidence is the top-two cosine margin, not softmax.

    Softmax must sum to 1 over a closed label set, so it reports high confidence
    for a forced choice among equally-bad options — which is exactly how the
    Sintel title card became "a store or shop" at 0.62.
    """

    def test_below_the_abstain_threshold_is_zero(self) -> None:
        assert margin_confidence(0.0026) == 0.0
        assert margin_confidence(0.0144) == 0.0

    def test_above_the_threshold_is_positive_but_modest(self) -> None:
        confidence = margin_confidence(0.0248)
        assert 0.0 < confidence < 0.5

    def test_confidence_is_monotonic_in_margin(self) -> None:
        values = [margin_confidence(m) for m in (0.02, 0.03, 0.04, 0.05, 0.09)]
        assert values == sorted(values)

    def test_confidence_saturates_at_one(self) -> None:
        assert margin_confidence(10.0) == 1.0

    def test_threshold_sits_between_the_measured_cases(self) -> None:
        # Guards the calibration: correct label 0.0248, forced guesses <= 0.0144.
        assert 0.0144 < SETTING_MARGIN_ABSTAIN < 0.0248


# --- Audio tagging --------------------------------------------------------


class TestAudioTagging:
    @staticmethod
    def window(start: float, end: float, **labels: float) -> AudioTagWindow:
        return AudioTagWindow(start=start, end=end, probabilities=dict(labels))

    def test_music_and_speech_are_independent_labels(self) -> None:
        """AudioSet is multi-label: music and speech genuinely co-occur."""
        segments, _ = segments_from_tags(
            [self.window(0, 10, Music=0.87, Speech=0.84)], duration=10
        )
        assert segments[0].segment_type is AudioSegmentType.MUSIC
        assert segments[0].confidence == pytest.approx(0.87)

    def test_a_trained_tag_reports_calibrated_confidence(self) -> None:
        """The DSP heuristic it replaces could report none at all."""
        segments, _ = segments_from_tags([self.window(0, 10, Music=0.61)], duration=10)
        assert segments[0].confidence is not None
        assert segments[0].provenance.source is AnalysisSource.AUDIO_TAGGING

    def test_silence_requires_music_and_speech_to_be_absent(self) -> None:
        segments, _ = segments_from_tags(
            [self.window(0, 10, Silence=0.9, Music=0.05)], duration=10
        )
        assert segments[0].segment_type is AudioSegmentType.SILENCE

    def test_transcript_intervals_take_precedence_for_speech(self) -> None:
        """ASR word boundaries beat a windowed probability."""
        segments, _ = segments_from_tags(
            [self.window(0, 10, Music=0.9)],
            duration=10,
            speech_intervals=[(2.0, 4.0)],
        )
        speech = [s for s in segments if s.segment_type is AudioSegmentType.SPEECH]
        assert len(speech) == 1
        assert (speech[0].start_time, speech[0].end_time) == (2.0, 4.0)
        assert speech[0].provenance.source is AnalysisSource.TRIBE_EVENTS

    def test_medium_labels_are_not_reported_as_events(self) -> None:
        """'Music' is what the audio IS, not something that happened."""
        events = events_from_tags(
            [self.window(0, 10, Music=0.9, Speech=0.8, Applause=0.7)], duration=10
        )
        assert [e["label"] for e in events] == ["Applause"]

    def test_broad_parent_labels_are_dropped(self) -> None:
        events = events_from_tags(
            [self.window(0, 10, **{"Sound effect": 0.9, "Applause": 0.6})], duration=10
        )
        assert [e["label"] for e in events] == ["Applause"]

    def test_below_threshold_labels_are_ignored(self) -> None:
        assert events_from_tags([self.window(0, 10, Applause=0.1)], duration=10) == []

    def test_overlapping_windows_combine_by_maximum(self) -> None:
        """A sound in half a window is present; averaging would dilute it."""
        windows = [self.window(0, 10, Music=0.2), self.window(5, 15, Music=0.9)]
        series = probability_series(windows, "Music", np.array([7.0]))
        assert series[0] == pytest.approx(0.9)

    def test_no_windows_degrades_with_a_warning(self) -> None:
        segments, warnings = segments_from_tags([], duration=10)
        assert segments == []
        assert any("DSP" in w for w in warnings)


# --- Diarization abstention ----------------------------------------------


class TestDiarizationAbstention:
    """Measured on the Sintel trailer: two speakers, four clusters.

    Same-speaker male utterances scored 0.594 apart while same-speaker female
    utterances scored 0.811 — indistinguishable from the 0.818 between
    *different* speakers, because the dialogue sits under an orchestral score.
    """

    def test_separated_clusters_score_high(self) -> None:
        matrix = np.array([[1, 0, 0], [0.98, 0.2, 0], [0, 1, 0], [0.1, 0.99, 0]], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        labels = _cluster(matrix, 0.55)
        assert _separation(matrix, labels) > 0.8

    def test_structureless_vectors_score_at_zero(self) -> None:
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(4, 64)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        assert _separation(matrix, _cluster(matrix, 0.55)) < MIN_SEPARATION

    def test_single_cluster_is_trivially_separated(self) -> None:
        matrix = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (3, 1))
        assert _separation(matrix, [0, 0, 0]) == 1.0

    def test_greedy_fallback_matches_on_easy_input(self) -> None:
        matrix = np.array([[1, 0], [0.99, 0.1], [0, 1]], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        labels = _greedy_cluster(matrix, 0.55)
        assert labels[0] == labels[1] != labels[2]


# --- Structural role from evidence ---------------------------------------


class TestStructuralRole:
    @staticmethod
    def segment(text: str, start: float, end: float) -> TranscriptSegment:
        return TranscriptSegment(
            index=0, start_time=start, end_time=end, text=text, provenance=ASR
        )

    def test_discourse_marker_beats_clock_position(self) -> None:
        """The case position alone gets wrong: a conclusion opening the video."""
        segment = self.segment("Finally, remember to keep it safe.", 0.0, 3.0)
        assert _positional_prior(segment, 15.0) is StructuralRole.INTRODUCTION
        role, reason = _structural_role(segment, 15.0)
        assert role is StructuralRole.CONCLUSION
        assert "discourse marker" in reason

    def test_intro_marker_is_recognised(self) -> None:
        role, _ = _structural_role(self.segment("In this video I show you.", 9.0, 12.0), 15.0)
        assert role is StructuralRole.INTRODUCTION

    def test_call_to_action_implies_conclusion(self) -> None:
        role, reason = _structural_role(
            self.segment("Grab yours from the site.", 1.0, 3.0), 15.0, is_cta=True
        )
        assert role is StructuralRole.CONCLUSION
        assert "call to action" in reason

    def test_marker_must_open_the_utterance(self) -> None:
        """'we finally arrived' is narrative, not a concluding marker."""
        assert _marker_role("we finally arrived at the house") is None

    def test_position_is_the_documented_fallback(self) -> None:
        role, reason = _structural_role(
            self.segment("The mountain stood against the sky.", 13.0, 15.0), 15.0
        )
        assert role is StructuralRole.CONCLUSION
        assert "position" in reason

    def test_embedding_evidence_needs_a_clear_margin(self) -> None:
        """A near-tie is a coin flip and must not outrank the prior."""
        segment = self.segment("The mountain stood against the sky.", 0.0, 2.0)
        _, reason = _structural_role(
            segment,
            15.0,
            embedding_scores={
                StructuralRole.INTRODUCTION: 0.50,
                StructuralRole.DEVELOPMENT: 0.49,
                StructuralRole.CONCLUSION: 0.48,
            },
        )
        assert "position" in reason


# --- Object appearances ---------------------------------------------------


class TestObjectInterval:
    def test_labels_are_ranked_by_confidence(self) -> None:
        detections = [
            Detection(time=1.0, label="person", confidence=0.9, box=(0, 0, 1, 1)),
            Detection(time=1.0, label="dog", confidence=0.6, box=(0, 0, 1, 1)),
        ]
        assert objects_in_interval(detections, 0.0, 2.0) == ("person", "dog")

    def test_only_detections_inside_the_interval_count(self) -> None:
        detections = [Detection(time=9.0, label="person", confidence=0.9, box=(0, 0, 1, 1))]
        assert objects_in_interval(detections, 0.0, 2.0) == ()

    def test_repeated_labels_are_deduplicated(self) -> None:
        detections = [
            Detection(time=t, label="person", confidence=c, box=(0, 0, 1, 1))
            for t, c in ((1.0, 0.7), (1.5, 0.95))
        ]
        assert objects_in_interval(detections, 0.0, 2.0) == ("person",)


# --- Workspace (artifact-hygiene debt) ------------------------------------


class TestWorkspace:
    """Regenerable intermediates must live in the cache, not the run directory.

    Phase 1's rule is that artifacts reference media rather than duplicating it;
    a decoded audio track is ~19 MB for a 10-minute video.
    """

    def test_workspace_is_keyed_by_stimulus_hash(self, tmp_path: Path) -> None:
        first = ContentWorkspace.for_stimulus(tmp_path, "a" * 64)
        again = ContentWorkspace.for_stimulus(tmp_path, "a" * 64)
        other = ContentWorkspace.for_stimulus(tmp_path, "b" * 64)
        assert first.root == again.root
        assert first.root != other.root

    def test_workspace_is_outside_the_artifact_run_directory(self, tmp_path: Path) -> None:
        workspace = ContentWorkspace.for_stimulus(tmp_path / "cache", "a" * 64)
        assert "runs" not in workspace.root.parts
        assert workspace.audio_path.name == "audio.wav"

    def test_two_runs_share_one_decode(self, tmp_path: Path) -> None:
        workspace = ContentWorkspace.for_stimulus(tmp_path, "a" * 64)
        workspace.audio_path.write_bytes(b"x" * 32)
        assert workspace.has_audio()
        assert ContentWorkspace.for_stimulus(tmp_path, "a" * 64).has_audio()

    def test_everything_is_disposable(self, tmp_path: Path) -> None:
        workspace = ContentWorkspace.for_stimulus(tmp_path, "a" * 64)
        workspace.audio_path.write_bytes(b"x" * 100)
        assert workspace.size_bytes() == 100
        workspace.clear()
        assert not workspace.has_audio()

    def test_clear_all_reclaims_every_stimulus(self, tmp_path: Path) -> None:
        for digest in ("a" * 64, "b" * 64):
            ContentWorkspace.for_stimulus(tmp_path, digest).audio_path.write_bytes(b"x" * 50)
        assert clear_all(tmp_path) == 100
        assert clear_all(tmp_path) == 0
