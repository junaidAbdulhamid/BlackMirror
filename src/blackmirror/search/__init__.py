"""Phase 9: automated, budgeted search over content variants.

Phase 8 gave the system an evaluation function. Phase 9 turns it into a search
problem: find the strongest observed candidate using as few expensive
evaluations as possible, under an explicit space, budget and objective.

Built so far: the experiment definition, its configuration, the budget system
with hard admission, the typed search space, candidate genomes, decoding to the
Phase 7 candidate vocabulary, the evaluator that drives Phase 8, and random
search as the baseline every other strategy is measured against.
"""

from blackmirror.search.budget import (
    AdmissionDecision,
    BudgetLedger,
    BudgetLimit,
    SearchBudget,
    admit,
)
from blackmirror.search.cache import (
    CacheIdentity,
    CachingEvaluator,
    SearchEvaluationCache,
)
from blackmirror.search.decode import DecodeError, GenomeDecoder
from blackmirror.search.evaluator import CandidateEvaluator
from blackmirror.search.fidelity import FidelityLevel
from blackmirror.search.fitness import (
    CandidateFitness,
    ComputeCost,
    FitnessStatus,
    directional_fitness,
)
from blackmirror.search.genome import (
    CandidateGenome,
    GenomeOrigin,
    genome_distance,
    root_genome,
)
from blackmirror.search.guardrails import (
    Guardrail,
    GuardrailKind,
    GuardrailPolicy,
)
from blackmirror.search.mutation import (
    MutationMagnitude,
    MutationRegistry,
    crossover,
    neighbours,
)
from blackmirror.search.orchestrator import NeuralSearchOrchestrator
from blackmirror.search.pareto import ParetoFront, pareto_front, select_by_pareto_rank
from blackmirror.search.population import (
    SearchPopulation,
    build_population,
    diversity_aware_selection,
    successive_halving,
)
from blackmirror.search.scheduler import CandidateScheduler, HumanOverride
from blackmirror.search.schemas import (
    MEASURED_NUISANCE_FLOOR,
    SearchConfig,
    SearchExperiment,
    SearchStatus,
    SearchStrategyName,
    StoppingReason,
)
from blackmirror.search.space import (
    ContentSearchSpace,
    ParameterType,
    SearchParameter,
    SearchSpaceError,
    categorical,
    continuous,
)
from blackmirror.search.strategies import (
    BeamSearch,
    EpsilonGreedy,
    EvolutionarySearch,
    GridSearch,
    HillClimbing,
    LocalSearch,
)
from blackmirror.search.strategy import (
    RandomSearch,
    SearchStrategy,
    SearchStrategyRegistry,
    default_registry,
)
from blackmirror.search.tracking import (
    BestCandidateRecord,
    SearchEfficiencyMetrics,
    SearchEvent,
    SearchEventType,
    SearchTracker,
    StoppingPolicy,
    TrajectoryPoint,
)

__all__ = [
    "MEASURED_NUISANCE_FLOOR",
    "AdmissionDecision",
    "BeamSearch",
    "BestCandidateRecord",
    "BudgetLedger",
    "BudgetLimit",
    "CacheIdentity",
    "CachingEvaluator",
    "CandidateEvaluator",
    "CandidateFitness",
    "CandidateGenome",
    "CandidateScheduler",
    "ComputeCost",
    "ContentSearchSpace",
    "DecodeError",
    "EpsilonGreedy",
    "EvolutionarySearch",
    "FidelityLevel",
    "FitnessStatus",
    "GenomeDecoder",
    "GenomeOrigin",
    "GridSearch",
    "Guardrail",
    "GuardrailKind",
    "GuardrailPolicy",
    "HillClimbing",
    "HumanOverride",
    "LocalSearch",
    "MutationMagnitude",
    "MutationRegistry",
    "NeuralSearchOrchestrator",
    "ParameterType",
    "ParetoFront",
    "RandomSearch",
    "SearchBudget",
    "SearchConfig",
    "SearchEfficiencyMetrics",
    "SearchEvaluationCache",
    "SearchEvent",
    "SearchEventType",
    "SearchExperiment",
    "SearchParameter",
    "SearchPopulation",
    "SearchSpaceError",
    "SearchStatus",
    "SearchStrategy",
    "SearchStrategyName",
    "SearchStrategyRegistry",
    "SearchTracker",
    "StoppingPolicy",
    "StoppingReason",
    "TrajectoryPoint",
    "admit",
    "build_population",
    "categorical",
    "continuous",
    "crossover",
    "default_registry",
    "directional_fitness",
    "diversity_aware_selection",
    "genome_distance",
    "neighbours",
    "pareto_front",
    "root_genome",
    "select_by_pareto_rank",
    "successive_halving",
]
