"""ContentTimeline, fusion, and content↔neural alignment.

These are the pure-logic cores of Phase 4. They use small deterministic data so
a failure points at the algorithm rather than at a model download.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from blackmirror.content.alignment import (
    ContentNeuralTimeMapper,
    associate_neural_events,
    explore_correlations,
    rank_associations,
)
from blackmirror.content.fusion import build_boundaries, fuse
from blackmirror.content.schemas import (
    AnalysisSource,
    AudioSegment,
    AudioSegmentType,
    ContentEvent,
    ContentEventType,
    Provenance,
    ShotSegment,
    TranscriptSegment,
)
from blackmirror.content.timeline import ContentTimeline

DSP = Provenance(source=AnalysisSource.SIGNAL_DSP)
ASR = Provenance(source=AnalysisSource.TRIBE_EVENTS)


def event(identifier: str, start: float, end: float, **extra: object) -> ContentEvent:
    return ContentEvent(
        event_id=identifier,
        event_type=ContentEventType.FUSED_INTERVAL,
        start_time=start,
        end_time=end,
        **extra,  # type: ignore[arg-type]
    )


# --- ContentTimeline ------------------------------------------------------


class TestContentTimeline:
    @pytest.fixture
    def timeline(self) -> ContentTimeline:
        return ContentTimeline([event("A", 0, 5), event("B", 5, 10), event("C", 10, 15)])

    def test_lookup_uses_half_open_intervals(self, timeline: ContentTimeline) -> None:
        """Matches Phase 2's coverage convention exactly."""
        assert timeline.get_event_at(0).event_id == "A"
        assert timeline.get_event_at(4.999).event_id == "A"
        # The boundary belongs to the NEXT event, not the previous one.
        assert timeline.get_event_at(5).event_id == "B"

    def test_final_instant_resolves(self, timeline: ContentTimeline) -> None:
        """Scrubbing to the very end must not fall off the timeline."""
        assert timeline.get_event_at(15).event_id == "C"

    def test_outside_the_timeline_is_none(self, timeline: ContentTimeline) -> None:
        assert timeline.get_event_at(-1) is None
        assert timeline.get_event_at(20) is None

    def test_range_query_returns_every_overlap(self, timeline: ContentTimeline) -> None:
        assert [e.event_id for e in timeline.get_events_between(4, 11)] == ["A", "B", "C"]
        assert [e.event_id for e in timeline.get_events_between(6, 7)] == ["B"]

    def test_empty_range_returns_nothing(self, timeline: ContentTimeline) -> None:
        assert timeline.get_events_between(5, 5) == []
        assert timeline.get_events_between(9, 3) == []

    def test_window_is_centred(self, timeline: ContentTimeline) -> None:
        assert [e.event_id for e in timeline.window(7, 2)] == ["B"]
        assert [e.event_id for e in timeline.window(5, 2)] == ["A", "B"]

    def test_empty_timeline_is_safe(self) -> None:
        empty = ContentTimeline([])
        assert len(empty) == 0
        assert empty.get_event_at(1) is None
        assert empty.duration == 0.0

    def test_accessors_return_structured_state(self) -> None:
        timeline = ContentTimeline(
            [event("A", 0, 5, speech_text="hello", visual_description="a gym", objects=("person",))]
        )
        assert timeline.get_speech_at(2) == "hello"
        assert timeline.get_visual_state_at(2)["description"] == "a gym"
        assert timeline.get_visual_state_at(2)["objects"] == ["person"]
        assert timeline.get_speech_at(9) is None


# --- Fusion ---------------------------------------------------------------


class TestFusion:
    def test_boundaries_are_the_union_of_every_modality(self) -> None:
        boundaries = build_boundaries(
            duration=10.0,
            shots=[
                ShotSegment(index=0, start_time=0, end_time=6, duration=6),
                ShotSegment(index=1, start_time=6, end_time=10, duration=4),
            ],
            transcript=[
                TranscriptSegment(index=0, start_time=3, end_time=5, text="hi", provenance=ASR)
            ],
            audio_segments=[
                AudioSegment(
                    segment_type=AudioSegmentType.MUSIC,
                    start_time=0,
                    end_time=10,
                    mean_energy=0.1,
                    provenance=DSP,
                )
            ],
            overlays=[],
        )
        assert boundaries == [0.0, 3.0, 5.0, 6.0, 10.0]

    def test_boundaries_are_deduplicated_and_clamped(self) -> None:
        boundaries = build_boundaries(
            duration=5.0,
            shots=[ShotSegment(index=0, start_time=0, end_time=5, duration=5)],
            transcript=[
                # Slightly off a shot boundary, and running past the end.
                TranscriptSegment(index=0, start_time=0.01, end_time=9.0, text="x", provenance=ASR)
            ],
            audio_segments=[],
            overlays=[],
        )
        assert boundaries[0] == 0.0
        assert boundaries[-1] == 5.0
        assert all(0.0 <= point <= 5.0 for point in boundaries)

    def test_fused_interval_carries_every_overlapping_modality(self) -> None:
        """The Step 61 case: visual 2-6, speech 3-5, music 0-10."""
        from blackmirror.content.audio.signals import AudioSignals
        from blackmirror.content.visual.motion import VisualSignals

        events, _ = fuse(
            duration=10.0,
            shots=[
                ShotSegment(index=0, start_time=0, end_time=2, duration=2),
                ShotSegment(index=1, start_time=2, end_time=6, duration=4),
                ShotSegment(index=2, start_time=6, end_time=10, duration=4),
            ],
            scenes=[],
            attributes=[],
            overlays=[],
            audio_segments=[
                AudioSegment(
                    segment_type=AudioSegmentType.MUSIC,
                    start_time=0,
                    end_time=10,
                    mean_energy=0.2,
                    provenance=DSP,
                )
            ],
            transcript=[
                TranscriptSegment(
                    index=0, start_time=3, end_time=5, text="keep going", provenance=ASR
                )
            ],
            language=[],
            calls_to_action=[],
            visual_signals=VisualSignals(),
            audio_signals=AudioSignals(),
        )

        # The 3-5s interval must contain the speech AND the music.
        middle = next(e for e in events if e.start_time == 3.0)
        assert middle.speech_text == "keep going"
        assert middle.audio_description == "music"
        assert middle.shot_indices == (1,)
        assert "speech" in middle.modalities and "audio" in middle.modalities

    def test_events_partition_the_stimulus_without_gaps(self) -> None:
        from blackmirror.content.audio.signals import AudioSignals
        from blackmirror.content.visual.motion import VisualSignals

        events, _ = fuse(
            duration=10.0,
            shots=[
                ShotSegment(index=0, start_time=0, end_time=4, duration=4),
                ShotSegment(index=1, start_time=4, end_time=10, duration=6),
            ],
            scenes=[],
            attributes=[],
            overlays=[],
            audio_segments=[],
            transcript=[
                TranscriptSegment(index=0, start_time=2, end_time=6, text="x", provenance=ASR)
            ],
            language=[],
            calls_to_action=[],
            visual_signals=VisualSignals(),
            audio_signals=AudioSignals(),
        )
        assert events[0].start_time == 0.0
        assert events[-1].end_time == 10.0
        for previous, following in itertools.pairwise(events):
            assert previous.end_time == following.start_time


# --- Content ↔ neural alignment -------------------------------------------


class TestTimeMapper:
    @pytest.fixture
    def mapper(self) -> ContentNeuralTimeMapper:
        # A deliberately gappy neural timeline, as TRIBE actually produces.
        return ContentNeuralTimeMapper(np.array([0.0, 1.0, 2.0, 7.0, 8.0]), tr_seconds=1.0)

    def test_nearest_row_always_resolves(self, mapper: ContentNeuralTimeMapper) -> None:
        assert mapper.nearest_row(0.4) == 0
        assert mapper.nearest_row(6.6) == 3
        assert mapper.nearest_row(100) == 4

    def test_covering_row_respects_gaps(self, mapper: ContentNeuralTimeMapper) -> None:
        """Proximity is not coverage — the distinction Phase 2 established."""
        assert mapper.covering_row(7.5) == 3
        assert mapper.covering_row(6.6) == -1
        assert mapper.covering_row(4.0) == -1

    def test_covering_row_is_half_open(self, mapper: ContentNeuralTimeMapper) -> None:
        assert mapper.covering_row(2.0) == 2
        assert mapper.covering_row(3.0) == -1

    def test_content_time_for_row(self, mapper: ContentNeuralTimeMapper) -> None:
        assert mapper.content_time_for_row(3) == 7.0
        assert mapper.content_time_for_row(99) == 8.0  # clamped

    def test_non_aligned_backend_shifts_the_other_way(self) -> None:
        """A model that did not compensate the lag reports acquisition time."""
        mapper = ContentNeuralTimeMapper(
            np.array([5.0, 6.0]),
            tr_seconds=1.0,
            output_is_stimulus_aligned=False,
            hemodynamic_offset_seconds=5.0,
        )
        assert mapper.content_time_for_row(0) == 0.0
        assert mapper.covering_row(0.5) == 0
        assert mapper.neural_window_for_content(0.25, 1.25) == (0, 2)

    @pytest.mark.parametrize(
        "times,tr,offset",
        [
            (np.array([1.0, 0.0]), 1.0, 0.0),
            (np.array([0.0, np.nan]), 1.0, 0.0),
            (np.array([0.0]), 0.0, 0.0),
            (np.array([0.0]), 1.0, -1.0),
        ],
    )
    def test_invalid_time_contract_is_rejected(
        self, times: np.ndarray, tr: float, offset: float
    ) -> None:
        with pytest.raises(ValueError):
            ContentNeuralTimeMapper(
                times, tr_seconds=tr, hemodynamic_offset_seconds=offset
            )

    def test_neural_window_for_content(self, mapper: ContentNeuralTimeMapper) -> None:
        assert mapper.neural_window_for_content(0.5, 2.5) == (0, 3)
        assert mapper.neural_window_for_content(3.0, 4.0) == (0, 0)

    def test_empty_neural_timeline_is_safe(self) -> None:
        mapper = ContentNeuralTimeMapper(np.zeros(0), tr_seconds=1.0)
        assert mapper.nearest_row(1.0) == -1
        assert mapper.covering_row(1.0) == -1


class TestAssociation:
    @pytest.fixture
    def timeline(self) -> ContentTimeline:
        return ContentTimeline(
            [
                event("A", 0, 5, visual_description="gym"),
                event("B", 5, 10, visual_description="montage", speech_text="Keep going"),
                event("C", 10, 15, visual_description="product"),
            ]
        )

    def test_synthetic_alignment_case(self, timeline: ContentTimeline) -> None:
        """Step 60: events A/B/C at 0-5/5-10/10-15, neural event at 7 -> B."""
        association = associate_neural_events(
            [{"event_id": "n1", "event_type": "response_peak", "timestamp_seconds": 7.0,
              "score": 1.5}],
            timeline,
            context_window_seconds=2.0,
        )[0]
        assert association.content_event_ids == ("B",)
        assert association.speech_context == ("Keep going",)
        assert (association.window_start, association.window_end) == (5.0, 9.0)

    def test_wider_window_pulls_in_neighbours(self, timeline: ContentTimeline) -> None:
        association = associate_neural_events(
            [{"event_id": "n1", "event_type": "response_peak", "timestamp_seconds": 7.0,
              "score": 1.0}],
            timeline,
            context_window_seconds=4.0,
        )[0]
        assert set(association.content_event_ids) == {"A", "B", "C"}

    def test_window_is_clamped_at_zero(self, timeline: ContentTimeline) -> None:
        association = associate_neural_events(
            [{"event_id": "n1", "event_type": "peak", "timestamp_seconds": 1.0, "score": 1.0}],
            timeline,
            context_window_seconds=5.0,
        )[0]
        assert association.window_start == 0.0

    def test_association_states_it_is_not_causal(self, timeline: ContentTimeline) -> None:
        """The guarantee this whole phase rests on."""
        association = associate_neural_events(
            [{"event_id": "n1", "event_type": "peak", "timestamp_seconds": 7.0, "score": 1.0}],
            timeline,
        )[0]
        text = association.interpretation.lower()
        assert "co-occurrence" in text
        assert "not" in text and "produced" in text

    def test_ranking_is_by_neural_score(self, timeline: ContentTimeline) -> None:
        associations = associate_neural_events(
            [
                {"event_id": "a", "event_type": "peak", "timestamp_seconds": 2.0, "score": 0.4},
                {"event_id": "b", "event_type": "peak", "timestamp_seconds": 7.0, "score": 2.9},
            ],
            timeline,
        )
        assert [item.neural_event_id for item in rank_associations(associations)] == ["b", "a"]

    def test_malformed_neural_event_is_rejected(self, timeline: ContentTimeline) -> None:
        with pytest.raises(ValueError, match="missing required"):
            associate_neural_events([{"event_id": "bad"}], timeline)


class TestCorrelation:
    def test_too_few_samples_reports_nothing(self) -> None:
        """Statistical honesty: 4 neural samples cannot support a correlation."""
        results, warnings = explore_correlations(
            np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32),
            ("motion", "audio_energy"),
            np.linspace(0, 10, 20).astype(np.float32),
            {"global_response": np.array([0.1, 0.2, 0.3, 0.4])},
            np.array([0.0, 1.0, 2.0, 3.0]),
        )
        assert results == []
        assert any("at least" in warning for warning in warnings)

    def test_recovers_a_planted_relationship(self) -> None:
        """With enough samples a real signal must be found."""
        times = np.linspace(0, 30, 120).astype(np.float32)
        motion = np.sin(times / 3.0).astype(np.float32)
        matrix = np.stack([motion, np.zeros_like(motion)], axis=1)

        neural_times = np.linspace(0, 30, 31)
        neural = np.sin(neural_times / 3.0)

        results, _ = explore_correlations(
            matrix, ("motion", "flat"), times, {"global_response": neural}, neural_times
        )
        best = next(r for r in results if r.content_feature == "motion")
        assert abs(best.coefficient) > 0.9
        assert best.sample_count == 31
        assert "not a causal effect" in best.interpretation.lower()

    def test_multiplicity_is_reported_rather_than_hidden(self) -> None:
        """A raw p-value alone would overstate a best-of-many-offsets result.

        Measured on a real 10-sample run: the strongest coefficient had raw
        p=0.0118, which reads as significant until the 360 comparisons behind it
        are accounted for. Reporting both is more useful than reporting neither.
        """
        times = np.linspace(0, 30, 120).astype(np.float32)
        motion = np.sin(times / 3.0).astype(np.float32)
        matrix = np.stack([motion, np.cos(times / 4.0).astype(np.float32)], axis=1)
        neural_times = np.linspace(0, 30, 31)
        neural = np.sin(neural_times / 3.0)

        results, _ = explore_correlations(
            matrix, ("motion", "other"), times, {"global_response": neural}, neural_times
        )
        best = results[0]

        assert best.p_value is not None
        assert best.lags_tested > 1
        assert best.comparisons >= len(results)
        assert best.p_value_bonferroni is not None
        # Correction can only make a p-value larger, and never exceeds 1.
        assert best.p_value <= best.p_value_bonferroni <= 1.0
        assert "p_value_bonferroni" in best.interpretation

    def test_warns_when_nothing_survives_correction(self) -> None:
        """The honest headline for a small run with many comparisons."""
        rng = np.random.default_rng(0)
        times = np.linspace(0, 30, 120).astype(np.float32)
        matrix = rng.normal(size=(120, 4)).astype(np.float32)
        neural_times = np.linspace(0, 30, 12)
        series = {"global_response": rng.normal(size=12)}

        results, warnings = explore_correlations(
            matrix, ("a", "b", "c", "d"), times, series, neural_times
        )
        if results and all(
            r.p_value_bonferroni is not None and r.p_value_bonferroni >= 0.05
            for r in results
        ):
            assert any("survives correction" in w for w in warnings)

    def test_lag_search_does_not_extrapolate_or_count_missing_pairs(self) -> None:
        content_times = np.arange(20, dtype=np.float64)
        content = content_times[:, None]
        neural_times = np.arange(20, dtype=np.float64)
        neural = neural_times - 5.0
        results, _ = explore_correlations(
            content,
            ("ramp",),
            content_times,
            {"global_mean": neural},
            neural_times,
            lags=(5.0,),
            min_samples=8,
        )
        assert results[0].sample_count == 15
        assert results[0].p_value is not None

    def test_nonfinite_pairs_are_excluded(self) -> None:
        times = np.arange(12, dtype=np.float64)
        content = times[:, None]
        neural = times.copy()
        neural[3] = np.nan
        results, _ = explore_correlations(
            content,
            ("ramp",),
            times,
            {"global_mean": neural},
            times,
            lags=(0.0,),
            min_samples=8,
        )
        assert results[0].sample_count == 11

    def test_constant_features_are_skipped(self) -> None:
        times = np.linspace(0, 30, 120).astype(np.float32)
        matrix = np.ones((120, 1), dtype=np.float32)
        neural_times = np.linspace(0, 30, 31)
        results, _ = explore_correlations(
            matrix, ("flat",), times, {"global_response": np.sin(neural_times)}, neural_times
        )
        assert results == []
