"""Steps 9-10: candidate genomes and decoding to the Phase 7 vocabulary.

The evidence tests matter most. A search candidate's parameter was chosen from a
declared range, not measured anywhere, and the schema it has to satisfy was
built for measured comparisons. If that distinction blurs, a sampled guess
becomes indistinguishable from a grounded finding everywhere downstream.
"""

from __future__ import annotations

import pytest

from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.optimization.schemas import (
    EvidenceKind,
    ExpectedDirection,
    InterventionType,
    OptimizationConstraints,
)
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.search.decode import DecodeError, GenomeDecoder
from blackmirror.search.genome import (
    CandidateGenome,
    GenomeOrigin,
    genome_distance,
    root_genome,
)
from blackmirror.search.space import ContentSearchSpace, categorical, continuous


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -6.0, 6.0, resolution=0.5),
            continuous("brightness", Op.BRIGHTNESS, -0.2, 0.2, resolution=0.01),
        )
    )


def _objective(direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE) -> NeuralObjective:
    # A TARGET objective is required by Phase 6 to declare the value it aims at,
    # which is exactly what makes "closer is better" well defined.
    normalization = (
        ObjectiveNormalization(target_value=0.5, target_tolerance=0.1)
        if direction is ObjectiveDirection.TARGET
        else ObjectiveNormalization()
    )
    return NeuralObjective(
        objective_id="roi_mean",
        name="ROI mean",
        metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction,
        normalization=normalization,
    )


def _genome(gain: float = 3.0, brightness: float = 0.1, **kwargs: object) -> CandidateGenome:
    fields: dict[str, object] = {
        "genome_id": "c1",
        "values": {"gain": gain, "brightness": brightness},
        "proposed_by": "random_search",
    }
    fields.update(kwargs)
    return CandidateGenome(**fields)  # type: ignore[arg-type]


def _decoder(**kwargs: object) -> GenomeDecoder:
    fields: dict[str, object] = {
        "search_id": "s1",
        "experiment_id": "exp",
        "space": _space(),
        "parent_variant_id": "root-run",
        "root_media_path": "/media/root.mp4",
        "root_duration_seconds": 10.0,
        "objective": _objective(),
    }
    fields.update(kwargs)
    return GenomeDecoder(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step 9 - genome
# ---------------------------------------------------------------------------


class TestGenomeIdentity:
    def test_same_point_has_the_same_fingerprint(self) -> None:
        space = _space()

        a = _genome(3.0, 0.1, genome_id="a")
        b = _genome(3.0, 0.1, genome_id="b", generation=4, origin=GenomeOrigin.MUTATION,
                    parent_genome_ids=("a",))

        assert a.fingerprint(space) == b.fingerprint(space)

    def test_identity_ignores_differences_below_the_resolution(self) -> None:
        """Two values the encoder cannot distinguish are one candidate.

        Without this a continuous strategy proposes an unbounded stream of
        near-identical points and spends the whole budget on them.
        """
        space = _space()

        assert _genome(3.0, 0.1000).fingerprint(space) == _genome(
            3.0, 0.10001
        ).fingerprint(space)

    def test_a_real_difference_changes_identity(self) -> None:
        space = _space()

        assert _genome(3.0).fingerprint(space) != _genome(3.5).fingerprint(space)


class TestGenomeLineage:
    def test_a_mutation_must_name_its_parent(self) -> None:
        with pytest.raises(ValueError, match="needs the parent"):
            _genome(origin=GenomeOrigin.MUTATION)

    def test_a_crossover_needs_two_parents(self) -> None:
        with pytest.raises(ValueError, match="at least two parents"):
            _genome(origin=GenomeOrigin.CROSSOVER, parent_genome_ids=("a",))

    def test_a_genome_cannot_be_its_own_parent(self) -> None:
        with pytest.raises(ValueError, match="its own parent"):
            _genome(origin=GenomeOrigin.MUTATION, parent_genome_ids=("c1",))

    def test_the_root_descends_from_nothing(self) -> None:
        with pytest.raises(ValueError, match="descends from nothing"):
            _genome(origin=GenomeOrigin.ROOT, parent_genome_ids=("a",))

    def test_a_child_inherits_lineage_and_advances_a_generation(self) -> None:
        parent = _genome(genome_id="p", generation=2)

        child = parent.with_values(
            {"gain": 4.0, "brightness": 0.1},
            genome_id="p1",
            origin=GenomeOrigin.NEIGHBOUR,
        )

        assert child.generation == 3
        assert child.parent_genome_ids == ("p",)
        assert child.origin is GenomeOrigin.NEIGHBOUR


class TestGenomeValidation:
    def test_a_value_outside_the_space_is_refused(self) -> None:
        """Caught before admission, not as an ffmpeg error an hour later."""
        with pytest.raises(ValueError, match="outside its declared domain"):
            _genome(gain=99.0).validate_against(_space())

    def test_a_missing_parameter_is_refused(self) -> None:
        genome = CandidateGenome(genome_id="c", values={"gain": 1.0})

        with pytest.raises(ValueError, match="missing values"):
            genome.validate_against(_space())

    def test_an_unknown_parameter_is_refused(self) -> None:
        genome = CandidateGenome(
            genome_id="c", values={"gain": 1.0, "brightness": 0.0, "ghost": 1.0}
        )

        with pytest.raises(ValueError, match="not in the space"):
            genome.validate_against(_space())


class TestGenomeDistance:
    def test_identical_points_are_at_zero_distance(self) -> None:
        assert genome_distance(_genome(), _genome(genome_id="b"), _space()) == 0.0

    def test_distance_is_normalized_across_differently_scaled_parameters(self) -> None:
        """A move across half of one range equals half of another's."""
        space = _space()

        gain_move = genome_distance(
            _genome(-6.0, 0.0), _genome(0.0, 0.0, genome_id="b"), space
        )
        brightness_move = genome_distance(
            _genome(0.0, -0.2), _genome(0.0, 0.0, genome_id="b"), space
        )

        assert gain_move == pytest.approx(brightness_move)

    def test_opposite_corners_are_at_maximum_distance(self) -> None:
        space = _space()

        assert genome_distance(
            _genome(-6.0, -0.2), _genome(6.0, 0.2, genome_id="b"), space
        ) == pytest.approx(1.0)

    def test_categorical_mismatch_counts_as_a_full_step(self) -> None:
        space = ContentSearchSpace(
            parameters=(categorical("speed", Op.SPEED, (0.9, 1.0, 1.1)),)
        )
        a = CandidateGenome(genome_id="a", values={"speed": 0.9})
        b = CandidateGenome(genome_id="b", values={"speed": 1.1})

        assert genome_distance(a, b, space) == pytest.approx(1.0)


class TestRootGenome:
    def test_the_root_sits_at_every_neutral_value(self) -> None:
        space = _space()

        root = root_genome(space)

        assert root.is_neutral(space) is True
        assert root.origin is GenomeOrigin.ROOT


# ---------------------------------------------------------------------------
# Step 10 - decoding
# ---------------------------------------------------------------------------


class TestDecodeToPlan:
    def test_produces_the_edit_that_builds_the_candidate(self) -> None:
        plan = _decoder().to_plan(_genome(3.0, 0.1))

        assert {step.operation for step in plan.effective_steps} == {
            Op.AUDIO_GAIN_DB,
            Op.BRIGHTNESS,
        }

    def test_refuses_a_genome_that_would_rebuild_the_root(self) -> None:
        """The root's score is already known; re-measuring it costs an hour."""
        with pytest.raises(DecodeError, match="rebuild the root unchanged"):
            _decoder().to_plan(root_genome(_space()))


class TestEvidenceHonesty:
    def test_evidence_is_marked_as_a_sample_not_a_measurement(self) -> None:
        """The distinction the whole module exists to preserve."""
        evidence = _decoder().evidence_for(_genome())

        assert {item.kind for item in evidence} == {EvidenceKind.SEARCH_SPACE_SAMPLE}
        assert {item.association for item in evidence} == {"not_measured"}

    def test_evidence_leaves_every_measurement_field_empty(self) -> None:
        """There is no measurement, so nothing may sit in a measured field."""
        for item in _decoder().evidence_for(_genome()):
            assert item.reference_value is None
            assert item.delta is None
            assert item.consistency is None
            assert item.reference_count == 0
            assert item.reference_variant_ids == ()

    def test_evidence_says_in_words_that_nothing_was_measured(self) -> None:
        """A reader who never inspects the enum must still not be misled."""
        for item in _decoder().evidence_for(_genome()):
            assert "not a" in item.description and "measurement" in item.description

    def test_only_changed_parameters_produce_evidence(self) -> None:
        evidence = _decoder().evidence_for(_genome(gain=3.0, brightness=0.0))

        assert [item.feature for item in evidence] == ["gain"]

    def test_evidence_records_the_value_that_was_applied(self) -> None:
        evidence = _decoder().evidence_for(_genome(gain=3.0, brightness=0.0))

        assert evidence[0].source_value == 3.0


class TestDecodeToSpec:
    def test_builds_a_specification_phase_8_can_consume(self) -> None:
        spec = _decoder().to_spec(_genome())

        assert spec.proposed_variant_id == "c1"
        assert spec.parent_variant_id == "root-run"
        assert len(spec.interventions) == 2
        assert spec.hypothesis_ids == ("HYP-s1-c1",)
        assert spec.edit_instructions

    def test_every_intervention_cites_its_own_sample(self) -> None:
        spec = _decoder().to_spec(_genome())

        cited = {eid for item in spec.interventions for eid in item.evidence_ids}
        assert cited == {"c1-gain", "c1-brightness"}

    def test_expected_direction_follows_the_objective(self) -> None:
        """Phase 8 rejects a hypothesis that contradicts its objective."""
        assert _decoder().expected_direction is ExpectedDirection.TEST_FOR_INCREASE
        assert (
            _decoder(objective=_objective(ObjectiveDirection.MINIMIZE)).expected_direction
            is ExpectedDirection.TEST_FOR_DECREASE
        )
        assert (
            _decoder(objective=_objective(ObjectiveDirection.TARGET)).expected_direction
            is ExpectedDirection.TEST_FOR_TARGET_PROXIMITY
        )

    def test_intervention_type_matches_the_modality_edited(self) -> None:
        spec = _decoder().to_spec(_genome())
        types = {item.type for item in spec.interventions}

        assert types == {InterventionType.AUDIO_EDIT, InterventionType.VISUAL_EDIT}

    def test_speed_is_recorded_as_a_pace_edit(self) -> None:
        space = ContentSearchSpace(
            parameters=(categorical("speed", Op.SPEED, (0.9, 1.1)),)
        )
        decoder = _decoder(space=space)
        genome = CandidateGenome(genome_id="c1", values={"speed": 1.1})

        spec = decoder.to_spec(genome)

        assert spec.interventions[0].type is InterventionType.PACE_EDIT
        assert spec.interventions[0].edit_instructions[0].operation == "adjust_pace"

    def test_the_note_states_the_values_were_selected_not_measured(self) -> None:
        assert "selected, not measured" in _decoder().to_spec(_genome()).note


class TestConstraints:
    def test_refuses_a_candidate_that_edits_a_frozen_modality(self) -> None:
        """Refused before building it, not after an hour of inference."""
        decoder = _decoder(
            constraints=OptimizationConstraints(frozen_modalities=("audio",))
        )

        with pytest.raises(DecodeError, match="modality is frozen"):
            decoder.to_spec(_genome(gain=3.0, brightness=0.1))

    def test_a_frozen_modality_left_untouched_is_fine(self) -> None:
        decoder = _decoder(
            constraints=OptimizationConstraints(frozen_modalities=("audio",))
        )

        spec = decoder.to_spec(_genome(gain=0.0, brightness=0.1))

        assert [item.modality for item in spec.interventions] == ["visual"]

    def test_duration_is_checked_against_the_produced_file(self) -> None:
        """A speed change's exact effect is only known once written."""
        decoder = _decoder(
            constraints=OptimizationConstraints(max_duration_change_seconds=0.5)
        )

        decoder.check_duration(10.4)
        with pytest.raises(DecodeError, match=r"beyond the 0\.5s"):
            decoder.check_duration(11.0)

    def test_preserve_intervals_are_carried_into_the_spec(self) -> None:
        decoder = _decoder(
            constraints=OptimizationConstraints(preserve_intervals=((0.0, 2.0),))
        )

        assert decoder.to_spec(_genome()).preserve_intervals == ((0.0, 2.0),)
