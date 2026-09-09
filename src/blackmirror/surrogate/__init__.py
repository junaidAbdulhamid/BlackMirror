"""Phase 10: a cheap learned approximation used to prioritise expensive work.

The real objective costs tens of minutes per evaluation. A surrogate learns an
approximation from candidates already evaluated and ranks thousands of untested
ones in milliseconds, so the expensive evaluations go where they are most
likely to be worth spending.

It is a screening model. It never replaces the real evaluator, a predicted
score is never an outcome, and a surrogate that cannot demonstrate rank skill
on held-out data is not allowed to choose anything.
"""

from blackmirror.surrogate.acquisition import (
    AcquisitionName,
    AcquisitionRegistry,
    AcquisitionScore,
    expected_improvement,
    greedy_mean,
    probability_of_improvement,
    score_candidates,
    select_diverse_batch,
    upper_confidence_bound,
)
from blackmirror.surrogate.encoder import GenomeFeatureEncoder
from blackmirror.surrogate.models import (
    ForestSurrogate,
    GaussianProcessSurrogate,
    GradientBoostingSurrogate,
    Surrogate,
    SurrogateError,
    SurrogateRegistry,
    default_registry,
)
from blackmirror.surrogate.multi import (
    MultiObjectiveSurrogate,
    ScalarizationWeights,
    sample_weights,
)
from blackmirror.surrogate.orchestrator import (
    BayesianOptimizationConfig,
    BayesianOptimizationOrchestrator,
    RoundRecord,
)
from blackmirror.surrogate.pool import CandidatePoolGenerator, PoolConfig
from blackmirror.surrogate.schemas import (
    DatasetIdentity,
    SurrogateDataset,
    SurrogateModelMetadata,
    SurrogateModelType,
    SurrogatePrediction,
    SurrogateState,
    SurrogateTrainingRecord,
)
from blackmirror.surrogate.trainer import (
    SurrogateTrainer,
    TrainedSurrogate,
    TrustPolicy,
    compare_families,
)
from blackmirror.surrogate.validation import (
    SurrogateMetrics,
    UncertaintyDiagnostics,
    cross_validate,
    ood_scores,
    prefix_validate,
)

__all__ = [
    "AcquisitionName",
    "AcquisitionRegistry",
    "AcquisitionScore",
    "BayesianOptimizationConfig",
    "BayesianOptimizationOrchestrator",
    "CandidatePoolGenerator",
    "DatasetIdentity",
    "ForestSurrogate",
    "GaussianProcessSurrogate",
    "GenomeFeatureEncoder",
    "GradientBoostingSurrogate",
    "MultiObjectiveSurrogate",
    "PoolConfig",
    "RoundRecord",
    "ScalarizationWeights",
    "Surrogate",
    "SurrogateDataset",
    "SurrogateError",
    "SurrogateMetrics",
    "SurrogateModelMetadata",
    "SurrogateModelType",
    "SurrogatePrediction",
    "SurrogateRegistry",
    "SurrogateState",
    "SurrogateTrainer",
    "SurrogateTrainingRecord",
    "TrainedSurrogate",
    "TrustPolicy",
    "UncertaintyDiagnostics",
    "compare_families",
    "cross_validate",
    "default_registry",
    "expected_improvement",
    "greedy_mean",
    "ood_scores",
    "prefix_validate",
    "probability_of_improvement",
    "sample_weights",
    "score_candidates",
    "select_diverse_batch",
    "upper_confidence_bound",
]
