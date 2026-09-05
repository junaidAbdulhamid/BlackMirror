"""Tests for the Phase 4 limitation fixes.

Each test names the failure it prevents. Several assert an *abstention* or a
*warning* rather than an answer: where a model cannot know, saying so is the
correct behaviour and is worth protecting with a test.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from blackmirror.content.alignment import _benjamini_hochberg, explore_correlations
from blackmirror.content.audio.tagging import (
    MODEL_SAMPLE_RATE,
    AudioTagWindow,
    _resample,
    segments_from_tags,
)
from blackmirror.content.schemas import AudioSegmentType, TransitionKind
from blackmirror.content.visual.motion import VisualSignals, _structural_change
from blackmirror.content.visual.objects import (
    Detection,
    _deduplicate,
    _iou,
    queries_from_transcript,
)
from blackmirror.content.visual.shots import (
    _Boundary,
    _kind_from_detectors,
    _merge_short,
    _unify,
)


class TestShotTransitions:
    """Frame-difference detection alone misses fades to black."""

    def test_fade_wins_over_cut_because_luminance_is_positive_evidence(self) -> None:
        assert _kind_from_detectors(("content", "threshold")) is TransitionKind.FADE
        assert _kind_from_detectors(("adaptive", "content", "threshold")) is TransitionKind.FADE

    def test_adaptive_plus_content_is_a_cut_not_a_gradual(self) -> None:
        """AdaptiveDetector also fires on hard cuts, so it cannot outvote content.

        Ranking kinds instead mislabelled the measured 124.6-luminance hard cut
        at 16.50 s in the Sintel trailer as "gradual".
        """
        assert _kind_from_detectors(("adaptive", "content")) is TransitionKind.CUT

    def test_adaptive_alone_is_gradual(self) -> None:
        assert _kind_from_detectors(("adaptive",)) is TransitionKind.GRADUAL

    def test_coincident_detections_merge_to_one_boundary(self) -> None:
        found = [
            _Boundary(10.80, 259, TransitionKind.FADE, "threshold"),
            _Boundary(10.95, 263, TransitionKind.CUT, "content"),
            _Boundary(20.00, 480, TransitionKind.CUT, "content"),
        ]
        unified = _unify(found, duration=30.0)
        times = [round(b.seconds, 2) for b in unified]
        assert times == [0.0, 10.8, 20.0], "0.15 s apart is one transition, not two"
        assert unified[1].kind is TransitionKind.FADE
        assert unified[1].seconds == 10.80, "a fade begins before content change is measurable"

    def test_timeline_always_opens_at_zero(self) -> None:
        unified = _unify([_Boundary(5.0, 120, TransitionKind.CUT, "content")], duration=30.0)
        assert unified[0].seconds == 0.0
        assert unified[0].kind is TransitionKind.START, "the opening is not a transition"

    def test_merging_a_short_shot_keeps_the_earlier_transition(self) -> None:
        spans = [
            (0.0, 5.0, 0, 120, TransitionKind.FADE, ("threshold",)),
            (5.0, 5.1, 120, 122, TransitionKind.CUT, ("content",)),
        ]
        merged = _merge_short(spans, 0.25)
        assert len(merged) == 1
        assert merged[0][1] == 5.1
        assert merged[0][4] is TransitionKind.FADE


class TestAudioSampleRate:
    """AST rejects any rate but 16 kHz, and used to fail silently."""

    def test_resample_reaches_the_models_rate(self) -> None:
        source = np.sin(np.linspace(0, 100, 44_100)).astype(np.float32)
        out, rate = _resample(source, 44_100, MODEL_SAMPLE_RATE)
        assert rate == MODEL_SAMPLE_RATE
        assert abs(out.size - 16_000) <= 2, "one second in, one second out"

    def test_resampling_preserves_signal_energy(self) -> None:
        source = np.sin(2 * np.pi * 220 * np.linspace(0, 1, 44_100)).astype(np.float32)
        out, _ = _resample(source, 44_100, MODEL_SAMPLE_RATE)
        rms_in = float(np.sqrt(np.mean(source**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        assert rms_out == pytest.approx(rms_in, rel=0.1)


class TestTagSegmentation:
    """Segments must partition the timeline, and singing is not speech."""

    @staticmethod
    def _windows(probabilities: dict[str, float]) -> list[AudioTagWindow]:
        return [AudioTagWindow(start=0.0, end=10.0, probabilities=probabilities)]

    def test_segments_do_not_overlap(self) -> None:
        """One segment per overlapping window produced near-duplicate spans."""
        windows = [
            AudioTagWindow(start=float(i), end=float(i) + 10.0, probabilities={"Music": 0.9})
            for i in range(5)
        ]
        segments, _ = segments_from_tags(windows, duration=10.0)
        for earlier, later in pairwise(segments):
            assert earlier.end_time <= later.start_time

    def test_segments_cover_the_whole_stimulus(self) -> None:
        segments, _ = segments_from_tags(
            self._windows({"Music": 0.9}), duration=10.0
        )
        covered = sum(s.end_time - s.start_time for s in segments)
        assert covered == pytest.approx(10.0, abs=0.3)

    def test_sung_vocals_are_not_reported_as_speech(self) -> None:
        """Speech and Singing both fire on a sung vocal; speech must clear singing."""
        segments, warnings = segments_from_tags(
            self._windows({"Speech": 0.5, "Singing": 0.8, "Music": 0.9}), duration=10.0
        )
        assert all(s.segment_type is not AudioSegmentType.SPEECH for s in segments)
        assert any("sung vocals" in w for w in warnings)

    def test_clear_speech_is_still_speech(self) -> None:
        segments, _ = segments_from_tags(
            self._windows({"Speech": 0.9, "Singing": 0.05, "Music": 0.1}), duration=10.0
        )
        assert any(s.segment_type is AudioSegmentType.SPEECH for s in segments)


class TestOpenVocabularyDetection:
    """A closed COCO vocabulary could only answer with its 91 classes."""

    def test_overlapping_boxes_of_one_label_collapse(self) -> None:
        """OWLv2 has no NMS; three 'a mountain' boxes were measured on one frame."""
        boxes = [
            Detection(1.0, "a mountain", 0.9, (0.0, 0.0, 100.0, 100.0)),
            Detection(1.0, "a mountain", 0.8, (5.0, 5.0, 102.0, 102.0)),
            Detection(1.0, "a person", 0.7, (0.0, 0.0, 100.0, 100.0)),
        ]
        kept = _deduplicate(boxes)
        assert len(kept) == 2, "same label + overlap is one object; a different label is not"
        assert kept[0].confidence == 0.9, "the most confident box survives"

    def test_distant_boxes_of_one_label_are_kept(self) -> None:
        boxes = [
            Detection(1.0, "a person", 0.9, (0.0, 0.0, 50.0, 50.0)),
            Detection(1.0, "a person", 0.8, (200.0, 200.0, 250.0, 250.0)),
        ]
        assert len(_deduplicate(boxes)) == 2

    def test_iou_is_zero_for_disjoint_boxes(self) -> None:
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_repeated_words_become_queries(self) -> None:
        text = "The dragon flew. The dragon roared. A sword fell, and the sword broke."
        assert queries_from_transcript(text) == ("a dragon", "a sword")

    def test_a_word_said_once_is_not_asked_about(self) -> None:
        assert queries_from_transcript("A solitary dragon appeared") == ()

    def test_stopwords_never_become_queries(self) -> None:
        queries = queries_from_transcript("that that this this with with from from")
        assert queries == ()


class TestMotionDecomposition:
    """Frame differencing cannot tell a moving subject from a fade."""

    def test_a_multiplicative_fade_scores_exactly_zero(self) -> None:
        """A fade to black is f2 = a*f1: every pixel changes, nothing moves.

        Mean-centring alone does NOT fix this — it leaves a residual of
        (a-1) x contrast — which is why the frames are standardised.
        """
        frame = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        for alpha in (0.5, 0.2, 0.05):
            faded = (frame * alpha).astype(np.float32)
            assert float(np.abs(faded - frame).mean()) > 0.1, "differencing sees a fade"
            assert _structural_change(faded, frame) == pytest.approx(0.0, abs=1e-6)

    def test_an_additive_flash_scores_exactly_zero(self) -> None:
        frame = np.random.default_rng(1).random((32, 32)).astype(np.float32)
        brighter = (frame + 0.2).astype(np.float32)
        assert _structural_change(brighter, frame) == pytest.approx(0.0, abs=1e-6)

    def test_real_movement_is_not_suppressed(self) -> None:
        frame = np.zeros((32, 32), dtype=np.float32)
        frame[8:16, 8:16] = 1.0
        moved = np.zeros((32, 32), dtype=np.float32)
        moved[8:16, 16:24] = 1.0  # same object, different place, same light
        assert _structural_change(moved, frame) > 0.5

    def test_a_blank_frame_yields_no_structural_motion(self) -> None:
        """Standardising a near-blank frame amplifies noise without limit, and
        comparing a structured frame against a flat one manufactures a large
        change at exactly the fade boundary it should ignore."""
        frame = np.random.default_rng(2).random((32, 32)).astype(np.float32)
        blank = np.full((32, 32), 0.001, dtype=np.float32)
        assert _structural_change(blank, frame) == 0.0
        assert _structural_change(blank, blank) == 0.0

    def test_signal_dict_exposes_the_decomposition(self) -> None:
        signals = VisualSignals(
            times=np.zeros(2, dtype=np.float32),
            motion=np.zeros(2, dtype=np.float32),
            motion_structural=np.zeros(2, dtype=np.float32),
            luminance_shift=np.zeros(2, dtype=np.float32),
            brightness=np.zeros(2, dtype=np.float32),
            brightness_p90=np.zeros(2, dtype=np.float32),
        )
        keys = signals.as_dict()
        assert "motion_structural" in keys and "luminance_shift" in keys
        assert "brightness_p90" in keys


class TestFalseDiscoveryRate:
    """Bonferroni alone is the wrong instrument for exploratory work."""

    def test_matches_a_hand_checked_example(self) -> None:
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
        q = _benjamini_hochberg(p_values)
        assert q[0.001] == pytest.approx(0.01)
        assert q[0.008] == pytest.approx(0.04)
        assert q[0.039] == pytest.approx(0.084, abs=1e-3)

    def test_q_values_never_fall_below_their_p_value(self) -> None:
        p_values = [0.001, 0.02, 0.3, 0.5, 0.9]
        q = _benjamini_hochberg(p_values)
        assert all(q[p] >= p - 1e-9 for p in p_values)

    def test_q_values_are_monotone_in_p(self) -> None:
        p_values = [0.001, 0.02, 0.3, 0.5, 0.9]
        q = _benjamini_hochberg(p_values)
        ordered = [q[p] for p in sorted(p_values)]
        assert ordered == sorted(ordered)

    def test_empty_input_is_not_an_error(self) -> None:
        assert _benjamini_hochberg([]) == {}

    def test_a_planted_signal_survives_correction(self) -> None:
        times = np.linspace(0, 60, 240).astype(np.float32)
        neural_times = np.linspace(0, 60, 60)
        rng = np.random.default_rng(0)
        matrix = np.stack(
            [np.sin(times / 3.0), rng.normal(size=240).astype(np.float32)], axis=1
        ).astype(np.float32)
        results, warnings = explore_correlations(
            matrix, ("signal", "noise"), times,
            {"global_mean": np.sin(neural_times / 3.0)}, neural_times,
        )
        best = next(r for r in results if r.content_feature == "signal")
        assert best.q_value_bh is not None and best.q_value_bh < 0.05
        assert any("survive FDR correction" in w for w in warnings)

    def test_pure_noise_survives_nothing(self) -> None:
        rng = np.random.default_rng(7)
        times = np.linspace(0, 60, 240).astype(np.float32)
        neural_times = np.linspace(0, 60, 60)
        results, warnings = explore_correlations(
            rng.normal(size=(240, 4)).astype(np.float32), tuple("abcd"), times,
            {"global_mean": rng.normal(size=60)}, neural_times,
        )
        assert results, "the test is vacuous if nothing was computed"
        assert not [r for r in results if r.q_value_bh is not None and r.q_value_bh < 0.05]
        assert any("No correlation survives correction" in w for w in warnings)

    def test_correction_counts_every_test_not_just_the_winners(self) -> None:
        """Correcting only the reported best would understate multiplicity by
        the number of offsets searched."""
        times = np.linspace(0, 60, 240).astype(np.float32)
        neural_times = np.linspace(0, 60, 60)
        rng = np.random.default_rng(3)
        results, _ = explore_correlations(
            rng.normal(size=(240, 3)).astype(np.float32), ("a", "b", "c"), times,
            {"global_mean": rng.normal(size=60)}, neural_times,
        )
        assert results
        assert results[0].comparisons > len(results)


class TestNullPromptAbstention:
    """A closed label set cannot say "none of these fit"; a null bank can.

    The top-two margin measures whether the model can separate its labels. It
    says nothing about whether any label fits at all, which is a different
    failure. Ground-truthed on eight Sintel frames: the margin test alone
    reported "a desert" for a close-up of a person in a cave (margin 0.0192),
    and the null test alone reported a setting for the Sintel title card.
    Applying both leaves one report -- the frame that really is a desert.
    """

    def test_the_two_tests_are_not_redundant(self) -> None:
        from blackmirror.content.visual.semantics import (
            NULL_MARGIN_ABSTAIN,
            SETTING_MARGIN_ABSTAIN,
        )

        # Measured values from the ground-truthed frames.
        person_in_cave = {"margin": 0.0192, "null_gap": 0.0069}
        title_card = {"margin": 0.0133, "null_gap": 0.0036}
        real_desert = {"margin": 0.0626, "null_gap": 0.0676}

        def reports(frame: dict[str, float]) -> bool:
            return (
                frame["margin"] >= SETTING_MARGIN_ABSTAIN
                and frame["null_gap"] >= NULL_MARGIN_ABSTAIN
            )

        # Each wrong frame is caught by a different one of the two tests.
        assert person_in_cave["margin"] >= SETTING_MARGIN_ABSTAIN, "margin passes it"
        assert person_in_cave["null_gap"] < NULL_MARGIN_ABSTAIN, "null test catches it"
        assert title_card["null_gap"] < NULL_MARGIN_ABSTAIN
        assert title_card["margin"] < SETTING_MARGIN_ABSTAIN, "margin catches it"

        assert not reports(person_in_cave)
        assert not reports(title_card)
        assert reports(real_desert), "the one correct label must survive both tests"

    def test_null_prompts_describe_nothing_visual(self) -> None:
        """If a control phrase were visually meaningful it would score highly on
        matching images and the bar would move with the content."""
        from blackmirror.content.visual.semantics import LABEL_GROUPS, NULL_PROMPTS

        assert len(NULL_PROMPTS) >= 8, "a single control phrase is a noisy maximum"
        settings = {s.lower() for s in LABEL_GROUPS["setting"]}
        assert not (settings & {p.lower() for p in NULL_PROMPTS})


class TestOcrScriptSelection:
    """The default recognition model silently drops characters it lacks."""

    def test_only_verified_scripts_are_offered(self) -> None:
        from blackmirror.content.visual.ocr import OCR_SCRIPTS

        assert set(OCR_SCRIPTS) == {"default", "japanese"}, (
            "Korean was measured to return nothing and is deliberately absent; "
            "Cyrillic, Arabic, Devanagari and Hebrew have no published model."
        )

    def test_v1_models_declare_a_32_pixel_input(self) -> None:
        """Passing the v4 height to a v1 model raises rather than degrading."""
        from blackmirror.content.visual.ocr import OCR_SCRIPTS

        assert OCR_SCRIPTS["default"][1] == (3, 48, 320)
        assert OCR_SCRIPTS["japanese"][1] == (3, 32, 320)

    def test_an_unknown_script_is_rejected_at_construction(self) -> None:
        from blackmirror.content.visual.ocr import OcrReader

        with pytest.raises(ValueError, match="unknown OCR script"):
            OcrReader(script="cyrillic")

    def test_the_default_is_unchanged(self) -> None:
        from blackmirror.content.visual.ocr import OcrReader

        assert OcrReader().script == "default"


class TestAudioEventMerging:
    """Overlapping windows multiplied every sound by the window overlap."""

    @staticmethod
    def _run(probabilities: float, count: int, hop: float = 1.0) -> list[AudioTagWindow]:
        return [
            AudioTagWindow(
                start=i * hop, end=i * hop + 10.24, probabilities={"Applause": probabilities}
            )
            for i in range(count)
        ]

    def test_one_sound_is_one_event(self) -> None:
        from blackmirror.content.audio.tagging import events_from_tags

        events = events_from_tags(self._run(0.6, 6), duration=20.0)
        assert len(events) == 1, "six overlapping windows described one applause"

    def test_a_gap_separates_two_events(self) -> None:
        from blackmirror.content.audio.tagging import events_from_tags

        windows = [
            AudioTagWindow(start=0.0, end=2.0, probabilities={"Applause": 0.6}),
            AudioTagWindow(start=9.0, end=11.0, probabilities={"Applause": 0.7}),
        ]
        assert len(events_from_tags(windows, duration=12.0)) == 2

    def test_the_event_is_timed_at_its_probability_peak(self) -> None:
        """A window start says only 'somewhere in the next 10.24 s'."""
        from blackmirror.content.audio.tagging import events_from_tags

        windows = [
            AudioTagWindow(start=0.0, end=10.24, probabilities={"Applause": 0.40}),
            AudioTagWindow(start=1.0, end=11.24, probabilities={"Applause": 0.95}),
            AudioTagWindow(start=2.0, end=12.24, probabilities={"Applause": 0.42}),
        ]
        event = events_from_tags(windows, duration=20.0)[0]
        assert event["confidence"] == 0.95
        assert event["peak_time"] == pytest.approx((1.0 + 11.24) / 2)
        # The merged span still exposes the real uncertainty.
        assert event["start_time"] == 0.0
        assert event["end_time"] == pytest.approx(12.24)

    def test_medium_labels_are_not_events(self) -> None:
        from blackmirror.content.audio.tagging import events_from_tags

        windows = [AudioTagWindow(start=0.0, end=10.0, probabilities={"Music": 0.99})]
        assert events_from_tags(windows, duration=10.0) == []


class TestOverlapScoring:
    """Diarization cannot see simultaneous speech; this scores the risk.

    The subprocess is stubbed: the point is the boundary contract and the
    failure handling, neither of which needs a 3 GB second environment.
    """

    @staticmethod
    def _payload(values: list[float | None]) -> str:
        import json

        return json.dumps(
            {
                "model_id": "pyannote/segmentation-3.0",
                "spans": [
                    {"start": float(i), "end": float(i) + 1.0, "overlap_probability": v}
                    for i, v in enumerate(values)
                ],
            }
        )

    def _run(self, monkeypatch, returncode: int, stdout: str, stderr: str = ""):
        import subprocess as sp

        from blackmirror.content.audio import overlap as mod

        monkeypatch.setattr(mod, "environment_available", lambda root: True)
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            lambda *a, **k: sp.CompletedProcess(a[0], returncode, stdout, stderr),
        )
        return mod.score_overlap(
            Path("/tmp/x.wav"), [(0.0, 1.0), (1.0, 2.0)], root=Path(".")
        )

    def test_scores_are_parsed_and_flagged(self, monkeypatch) -> None:
        from blackmirror.content.audio.overlap import OVERLAP_REVIEW_THRESHOLD

        scores, warnings = self._run(monkeypatch, 0, self._payload([0.02, 0.09]))
        assert [s.probability for s in scores] == [0.02, 0.09]
        assert not scores[0].flagged
        assert scores[1].flagged and scores[1].probability >= OVERLAP_REVIEW_THRESHOLD
        assert any("prompt to listen, not as a finding" in w for w in warnings)

    def test_no_flag_means_no_warning(self, monkeypatch) -> None:
        scores, warnings = self._run(monkeypatch, 0, self._payload([0.01, 0.02]))
        assert not any(s.flagged for s in scores)
        assert warnings == []

    def test_an_unscorable_span_is_none_not_zero(self, monkeypatch) -> None:
        """Zero would read as 'definitely one speaker'."""
        scores, _ = self._run(monkeypatch, 0, self._payload([None, 0.03]))
        assert scores[0].probability is None
        assert not scores[0].flagged

    def test_a_gated_repo_failure_is_named(self, monkeypatch) -> None:
        scores, warnings = self._run(
            monkeypatch, 1, "", "huggingface_hub.errors.GatedRepoError: 401 Client Error"
        )
        assert scores == []
        assert any("gated" in w and "hf auth login" in w for w in warnings)

    def test_unreadable_output_does_not_raise(self, monkeypatch) -> None:
        scores, warnings = self._run(monkeypatch, 0, "not json at all")
        assert scores == []
        assert any("unreadable" in w for w in warnings)

    def test_a_missing_environment_explains_itself(self, monkeypatch) -> None:
        from blackmirror.content.audio import overlap as mod

        monkeypatch.setattr(mod, "environment_available", lambda root: False)
        scores, warnings = mod.score_overlap(
            Path("/tmp/x.wav"), [(0.0, 1.0)], root=Path(".")
        )
        assert scores == []
        assert any(".venv-diarization" in w for w in warnings)

    def test_no_spans_is_not_an_error(self) -> None:
        from blackmirror.content.audio.overlap import score_overlap

        assert score_overlap(Path("/tmp/x.wav"), [], root=Path(".")) == ([], [])
