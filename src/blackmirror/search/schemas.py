"""Steps 1-2: the search experiment and the configuration that governs it.

WHAT A SEARCH EXPERIMENT IS
    Phase 8 gave the system an evaluation function: a variant goes in, a score
    on a declared objective comes out. Phase 9 turns that into a search problem,
    and a `SearchExperiment` is one attempt at it: a root variant, a space of
    permitted changes, an objective set, a budget, and the record of what was
    tried. It is the unit that gets checkpointed, resumed and reported on.

WHY THE MINIMUM IMPROVEMENT DEFAULT IS WHAT IT IS
    Measured on this corpus: swapping between two mirrors of nominally identical
    language-model weights shifted the whole-cortex mean by an average of
    0.0157, while the real differences between three content variants were
    0.0117 to 0.0241. Two of three comparisons were smaller than that nuisance
    shift, and one ranking reversed.

    A search that reports an improvement below that floor has found nothing
    distinguishable from an implementation detail. So `minimum_improvement`
    defaults to the measured floor rather than to zero or to some tidy small
    number, and a configuration that lowers it records a warning rather than
    silently accepting a threshold the pipeline cannot resolve.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from blackmirror.scoring.schemas import NeuralObjective
from blackmirror.search.budget import BudgetLedger, SearchBudget
from blackmirror.search.space import ContentSearchSpace

SEARCH_VERSION = "1.0"

#: Average shift in whole-cortex mean caused by swapping between two mirrors of
#: nominally identical Llama-3.2-3B weights, measured on three variants of one
#: 10 s clip. An improvement smaller than this is not distinguishable from a
#: change of implementation detail. See docs/scientific_limitations.md §6a.
MEASURED_NUISANCE_FLOOR = 0.0157


class SearchModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class SearchStrategyName(StrEnum):
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    LOCAL_SEARCH = "local_search"
    HILL_CLIMBING = "hill_climbing"
    BEAM_SEARCH = "beam_search"
    EPSILON_GREEDY = "epsilon_greedy"
    EVOLUTIONARY_SEARCH = "evolutionary_search"


class SearchStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class StoppingReason(StrEnum):
    MAX_CANDIDATES = "max_candidates"
    MAX_GENERATIONS = "max_generations"
    MAX_COST = "max_cost"
    MAX_WALL_TIME = "max_wall_time"
    MAX_TRIBE_RUNS = "max_tribe_runs"
    NO_IMPROVEMENT = "no_improvement"
    TARGET_REACHED = "target_reached"
    SPACE_EXHAUSTED = "space_exhausted"
    USER_STOPPED = "user_stopped"
    FAILED = "failed"


class SearchConfig(SearchModel):
    """How the search behaves. Versioned, because a user may change it midway."""

    strategy: SearchStrategyName = SearchStrategyName.RANDOM_SEARCH
    #: Seeded so a search is reproducible. The pipeline underneath is
    #: bit-identical across runs, so this is the only source of randomness.
    random_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    candidate_batch_size: int = Field(default=1, ge=1, le=64)
    #: Defaults to 1 deliberately: inference is memory-bound before it is
    #: compute-bound, so parallel evaluations contend for the same weights.
    max_concurrent_candidates: int = Field(default=1, ge=1, le=16)
    exploration_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    exploration_decay: float = Field(default=1.0, gt=0.0, le=1.0)
    minimum_exploration_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    minimum_improvement: float = Field(default=MEASURED_NUISANCE_FLOOR, ge=0.0)
    patience: int = Field(default=3, ge=1, le=100)
    beam_width: int = Field(default=3, ge=1, le=64)
    grid_steps: int = Field(default=5, ge=2, le=100)
    #: Refuse to enumerate a grid larger than this many points.
    max_grid_points: int = Field(default=64, ge=1, le=10_000)
    target_fitness: float | None = None
    config_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _coherent(self) -> SearchConfig:
        if self.minimum_exploration_rate > self.exploration_rate:
            raise ValueError("minimum_exploration_rate cannot exceed exploration_rate")
        if self.max_concurrent_candidates > self.candidate_batch_size:
            raise ValueError(
                "max_concurrent_candidates cannot exceed candidate_batch_size; a "
                "worker with nothing to take would sit idle"
            )
        return self

    @property
    def resolves_below_noise(self) -> bool:
        """Whether this config would accept an improvement it cannot resolve."""
        return self.minimum_improvement < MEASURED_NUISANCE_FLOOR

    def warnings(self) -> tuple[str, ...]:
        """Configuration facts a reader of the final report needs to know."""
        notes: list[str] = []
        if self.resolves_below_noise:
            notes.append(
                f"minimum_improvement is {self.minimum_improvement:g}, below the measured "
                f"nuisance floor of {MEASURED_NUISANCE_FLOOR:g}. Improvements this small "
                f"are not distinguishable from an implementation detail, so a winner "
                f"chosen at this threshold may not be a real finding."
            )
        if self.max_concurrent_candidates > 1:
            notes.append(
                f"max_concurrent_candidates is {self.max_concurrent_candidates}. Inference "
                f"is memory-bound on this machine; concurrent evaluations may run slower "
                f"in total than sequential ones."
            )
        if self.exploration_rate == 0.0:
            notes.append(
                "exploration_rate is 0, so the search only exploits and can settle in a "
                "local optimum without ever sampling elsewhere."
            )
        return tuple(notes)

    def exploration_rate_at(self, generation: int) -> float:
        """Exploration for a given generation, after decay (Step 27)."""
        if generation < 0:
            raise ValueError("generation cannot be negative")
        rate = self.exploration_rate * (self.exploration_decay**generation)
        return max(self.minimum_exploration_rate, rate)


class ConfigRevision(SearchModel):
    """A superseded configuration, kept rather than overwritten (Step 73)."""

    config: SearchConfig
    replaced_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    reason: str = ""


class SearchExperiment(SearchModel):
    """One search: where it started, what it may change, and what it may spend.

    Deliberately holds no population or trajectory. Those grow with every
    evaluation and belong in the search state that is checkpointed beside this;
    keeping the definition separate means the thing that describes the
    experiment stays comparable across runs of it.
    """

    search_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    #: The Phase 1 run the search starts from, and whose media is edited.
    root_run_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    root_media_path: str = Field(min_length=1)
    root_media_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: The stored Phase 6 score this search optimises against.
    objective_set_hash: str = Field(pattern=r"^[0-9a-f]{16,64}$")
    objective_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    objectives: tuple[NeuralObjective, ...] = Field(min_length=1)
    #: Which objective drives a scalar fitness. Required when there are several
    #: and the search is not running multi-objective.
    primary_objective_id: str | None = None
    space: ContentSearchSpace
    config: SearchConfig = SearchConfig()
    budget: SearchBudget = SearchBudget()
    ledger: BudgetLedger = BudgetLedger()
    status: SearchStatus = SearchStatus.CREATED
    generation: int = Field(default=0, ge=0)
    stopping_reason: StoppingReason | None = None
    config_history: tuple[ConfigRevision, ...] = ()
    search_version: str = SEARCH_VERSION
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    interpretation_notice: str = (
        "A search reports the best variant it observed among those it evaluated, "
        "under one explicitly declared objective on model-predicted cortical "
        "responses. It is not a global optimum, not a statement about human "
        "viewers, and not evidence that the edit caused the difference."
    )

    @model_validator(mode="after")
    def _coherent(self) -> SearchExperiment:
        objective_ids = {objective.objective_id for objective in self.objectives}
        if self.primary_objective_id is not None:
            if self.primary_objective_id not in objective_ids:
                raise ValueError("primary objective is not among the declared objectives")
        elif len(objective_ids) > 1:
            raise ValueError(
                "several objectives were declared without a primary; a scalar search "
                "needs to know which one it is maximising, and a multi-objective "
                "search should say so explicitly"
            )
        if self.status is SearchStatus.COMPLETED and self.stopping_reason is None:
            raise ValueError("a completed search must record why it stopped")
        return self

    @property
    def is_multi_objective(self) -> bool:
        return len(self.objectives) > 1

    @property
    def primary_objective(self) -> NeuralObjective:
        if self.primary_objective_id is None:
            return self.objectives[0]
        return next(
            objective
            for objective in self.objectives
            if objective.objective_id == self.primary_objective_id
        )

    def with_config(self, config: SearchConfig, *, reason: str = "") -> SearchExperiment:
        """Adopt a new configuration, keeping the old one (Step 73).

        History is appended rather than replaced so a report can say which
        settings were in force when each candidate was proposed. Silently
        rewriting the configuration would make an earlier decision look like it
        was taken under rules that did not exist yet.
        """
        revision = ConfigRevision(config=self.config, reason=reason)
        return self.model_copy(
            update={
                "config": config.model_copy(
                    update={"config_version": self.config.config_version + 1}
                ),
                "config_history": (*self.config_history, revision),
                "updated_at": dt.datetime.now(dt.UTC),
            }
        )

    def warnings(self) -> tuple[str, ...]:
        notes = list(self.config.warnings())
        if self.space.cardinality(self.config.grid_steps) < (self.budget.max_candidates or 0):
            notes.append(
                "the budget allows more candidates than the space has distinct points; "
                "the search will exhaust the space before its budget"
            )
        return tuple(notes)


__all__ = [
    "MEASURED_NUISANCE_FLOOR",
    "SEARCH_VERSION",
    "ConfigRevision",
    "SearchConfig",
    "SearchExperiment",
    "SearchStatus",
    "SearchStrategyName",
    "StoppingReason",
]
