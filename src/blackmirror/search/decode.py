"""Step 10: genome to `ProposedVariantSpec`, through the Phase 7/8 contracts.

WHY NOT A SHORTCUT
    It would be less work to hand Phase 8 a media path and skip the Phase 7
    vocabulary entirely. That would also mean a search candidate carried no
    record of what was changed, no objective it was aimed at, and no expected
    direction, so nothing downstream could explain or audit it. Reusing the
    existing spec keeps a search candidate the same kind of object as a
    human-approved one, and reviewable by the same screens.

THE ONE THING THAT MUST NOT BE REUSED
    `ContentIntervention` requires at least one evidence id, and every Phase 7
    evidence kind records a measured comparison between variants. A search
    candidate has no such measurement: a strategy picked a number out of a
    declared range. Citing a measured kind would make a guess look like a
    finding, so these cite `SEARCH_SPACE_SAMPLE` items whose `association` is
    `not_measured`, and whose description says in words that nothing was
    measured.

    That distinction is the whole reason this module is more than a mapping
    table.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from blackmirror.materialization.schemas import (
    OPERATION_BOUNDS,
    EditOperation,
    MaterializationPlan,
)
from blackmirror.optimization.schemas import (
    ContentIntervention,
    EditInstruction,
    Evidence,
    EvidenceKind,
    ExpectedDirection,
    InterventionType,
    OptimizationConstraints,
    ProposedVariantSpec,
)
from blackmirror.scoring.schemas import NeuralObjective, ObjectiveDirection
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.space import ContentSearchSpace

#: Phase 7 intervention type for each mechanical edit.
_INTERVENTION_TYPE: dict[EditOperation, InterventionType] = {
    EditOperation.AUDIO_GAIN_DB: InterventionType.AUDIO_EDIT,
    EditOperation.BRIGHTNESS: InterventionType.VISUAL_EDIT,
    EditOperation.CONTRAST: InterventionType.VISUAL_EDIT,
    EditOperation.SATURATION: InterventionType.VISUAL_EDIT,
    EditOperation.SPEED: InterventionType.PACE_EDIT,
}

#: `adjust_pace` exists precisely for the one operation that retimes content.
_EDIT_OPERATION: dict[EditOperation, str] = {
    EditOperation.AUDIO_GAIN_DB: "adjust_feature",
    EditOperation.BRIGHTNESS: "adjust_feature",
    EditOperation.CONTRAST: "adjust_feature",
    EditOperation.SATURATION: "adjust_feature",
    EditOperation.SPEED: "adjust_pace",
}

#: The only claim a candidate may make, taken from the objective's direction so
#: it can never contradict it. Phase 8's request schema rejects a mismatch.
_EXPECTED_DIRECTION: dict[ObjectiveDirection, ExpectedDirection] = {
    ObjectiveDirection.MAXIMIZE: ExpectedDirection.TEST_FOR_INCREASE,
    ObjectiveDirection.MINIMIZE: ExpectedDirection.TEST_FOR_DECREASE,
    ObjectiveDirection.TARGET: ExpectedDirection.TEST_FOR_TARGET_PROXIMITY,
}


class DecodeError(ValueError):
    """The genome cannot be turned into a candidate specification."""


class GenomeDecoder(BaseModel):
    """Turns a point in the space into everything downstream needs.

    Holds the context a genome deliberately does not carry: which media it edits,
    which objective it is aimed at, and what it is not allowed to touch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    search_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    space: ContentSearchSpace
    #: The run whose media every candidate is edited from.
    parent_variant_id: str = Field(min_length=1)
    root_media_path: str = Field(min_length=1)
    root_duration_seconds: float = Field(gt=0)
    objective: NeuralObjective
    constraints: OptimizationConstraints = OptimizationConstraints()

    # --- media -----------------------------------------------------------

    def to_plan(self, genome: CandidateGenome) -> MaterializationPlan:
        """The edit that builds this candidate's file."""
        genome.validate_against(self.space)
        if genome.is_neutral(self.space):
            raise DecodeError(
                f"genome {genome.genome_id!r} leaves every parameter at its neutral "
                f"value, so it would rebuild the root unchanged; the root's score is "
                f"already known and evaluating it again would cost a full inference "
                f"to learn nothing"
            )
        return MaterializationPlan(
            source_path=Path(self.root_media_path),
            steps=self.space.to_edit_steps(genome.values),
        )

    # --- Phase 7 vocabulary ----------------------------------------------

    @property
    def expected_direction(self) -> ExpectedDirection:
        return _EXPECTED_DIRECTION[self.objective.direction]

    def hypothesis_id(self, genome: CandidateGenome) -> str:
        return f"HYP-{self.search_id}-{genome.genome_id}"

    def evidence_for(self, genome: CandidateGenome) -> tuple[Evidence, ...]:
        """One item per changed parameter, marked as not measured.

        `source_value` carries the setting that was applied, which is a fact
        about the candidate rather than an observation of anything. Every
        measurement-shaped field is left empty, because there is no measurement
        to put in it.
        """
        genome.validate_against(self.space)
        items: list[Evidence] = []
        for parameter in self.space.parameters:
            value = genome.values[parameter.name]
            neutral = OPERATION_BOUNDS[parameter.operation][2]
            if value == neutral:
                continue
            domain = (
                f"[{parameter.low:g}, {parameter.high:g}]"
                if parameter.is_ranged
                else "{" + ", ".join(f"{c:g}" for c in parameter.choices) + "}"
            )
            items.append(
                Evidence(
                    evidence_id=f"{genome.genome_id}-{parameter.name}",
                    kind=EvidenceKind.SEARCH_SPACE_SAMPLE,
                    description=(
                        f"Search set {parameter.name} to {value:g} (parent {neutral:g}), "
                        f"chosen by {genome.proposed_by or 'the search strategy'} from the "
                        f"declared range {domain}. This is a selected setting, not a "
                        f"measurement: nothing has been observed about its effect."
                    ),
                    feature=parameter.name,
                    source_value=value,
                    interval_start_seconds=0.0,
                    interval_end_seconds=self.root_duration_seconds,
                    source_variant_id=self.parent_variant_id,
                    association="not_measured",
                    note=f"origin={genome.origin.value}, generation={genome.generation}",
                )
            )
        if not items:  # pragma: no cover - to_plan rejects a neutral genome first
            raise DecodeError(f"genome {genome.genome_id!r} changes nothing")
        return tuple(items)

    def interventions(self, genome: CandidateGenome) -> tuple[ContentIntervention, ...]:
        """One intervention per changed parameter, each citing its own sample."""
        evidence = {item.feature: item for item in self.evidence_for(genome)}
        out: list[ContentIntervention] = []
        for parameter in self.space.parameters:
            item = evidence.get(parameter.name)
            if item is None:
                continue
            value = genome.values[parameter.name]
            out.append(
                ContentIntervention(
                    intervention_id=f"{genome.genome_id}-{parameter.name}",
                    type=_INTERVENTION_TYPE[parameter.operation],
                    target_interval_start=0.0,
                    target_interval_end=self.root_duration_seconds,
                    description=(
                        f"Set {parameter.operation.value} to {value:g} across the whole "
                        f"stimulus."
                    ),
                    parameters={parameter.operation.value: value},
                    rationale=(
                        "Proposed by automated search over a declared space. The value "
                        "was selected, not inferred from any measured comparison, and "
                        "no claim is made about its effect until it is evaluated."
                    ),
                    evidence_ids=(item.evidence_id,),
                    objective_id=self.objective.objective_id,
                    expected_direction=self.expected_direction,
                    edit_instructions=(
                        EditInstruction(
                            operation=_EDIT_OPERATION[parameter.operation],  # type: ignore[arg-type]
                            from_time=0.0,
                            to_time=self.root_duration_seconds,
                            feature=parameter.operation.value,
                            target_value=value,
                            note="Applied mechanically by the ffmpeg materializer.",
                        ),
                    ),
                    modality=parameter.modality,
                )
            )
        return tuple(out)

    def to_spec(self, genome: CandidateGenome) -> ProposedVariantSpec:
        """The Phase 8 input for this candidate.

        Checked against the declared constraints first: a candidate that edits a
        frozen modality is refused here rather than built, evaluated, and only
        then found to be inadmissible after an hour of inference.
        """
        genome.validate_against(self.space)
        self.check_constraints(genome)
        interventions = self.interventions(genome)
        if not interventions:
            raise DecodeError(f"genome {genome.genome_id!r} changes nothing")
        return ProposedVariantSpec(
            proposed_variant_id=genome.genome_id,
            parent_variant_id=self.parent_variant_id,
            experiment_id=self.experiment_id,
            hypothesis_ids=(self.hypothesis_id(genome),),
            interventions=interventions,
            constraints=self.constraints,
            target_objective_id=self.objective.objective_id,
            expected_direction=self.expected_direction,
            edit_instructions=tuple(
                instruction
                for intervention in interventions
                for instruction in intervention.edit_instructions
            ),
            preserve_intervals=self.constraints.preserve_intervals,
            note=(
                "A candidate proposed by automated search over a declared parameter "
                "space. Its parameter values were selected, not measured, and it "
                "carries no evidence about its effect until re-simulation measures it."
            ),
        )

    # --- constraints -----------------------------------------------------

    def check_constraints(self, genome: CandidateGenome) -> None:
        """Refuse a candidate the declared constraints forbid.

        Only the constraints a genome can violate on its own are checked here.
        Duration is checked against the produced file instead, because a speed
        change is the one edit whose effect on length is not exactly known until
        the container has been written.
        """
        genome.validate_against(self.space)
        frozen = set(self.constraints.frozen_modalities)
        for parameter in self.space.parameters:
            value = genome.values[parameter.name]
            if value == OPERATION_BOUNDS[parameter.operation][2]:
                continue
            if parameter.modality in frozen:
                raise DecodeError(
                    f"genome {genome.genome_id!r} changes {parameter.name}, which edits "
                    f"the {parameter.modality} modality, and that modality is frozen"
                )

    def check_duration(self, produced_seconds: float) -> None:
        """Check a materialized candidate's length against the constraint."""
        allowed = self.constraints.max_duration_change_seconds
        if allowed is None:
            return
        change = abs(produced_seconds - self.root_duration_seconds)
        if change > allowed + 1e-9:
            raise DecodeError(
                f"the produced candidate changes duration by {change:.3f}s, beyond the "
                f"{allowed:g}s the constraints allow"
            )


__all__ = ["DecodeError", "GenomeDecoder"]
