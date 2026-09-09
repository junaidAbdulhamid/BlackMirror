"""Steps 13-18: how well the surrogate predicts, and when not to trust it.

WHY RANK METRICS MATTER MORE THAN ERROR
    The surrogate's job is to decide which candidate gets the next expensive
    evaluation. That is a ranking decision. A model with a large constant bias
    but perfect ordering picks exactly the right candidate every time; a model
    with tiny error but shuffled ordering picks badly. So Spearman correlation
    and top-K recall are the metrics that decide whether the surrogate is
    useful, and MAE and RMSE are reported alongside because they say whether the
    numbers shown to a person are believable.

WHY ORDER-AWARE VALIDATION IS THE ONE THAT COUNTS
    Random cross-validation predicts a held-out point using observations that,
    in the real search, had not happened yet. That is leakage from the future,
    and it flatters the model relative to what it will actually do online, where
    it always predicts the next candidate from the ones before it. Both are
    computed; the prefix score is the honest one.

WHY OUT-OF-DISTRIBUTION DISTANCE IS SEPARATE FROM UNCERTAINTY
    A forest's spread can be small in a region it has never seen. Distance to
    the nearest training point cannot: it is a property of the data, not of the
    model, so it catches the failure mode model-derived uncertainty misses.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from blackmirror.surrogate.models import Surrogate


class ValidationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class SurrogateMetrics(ValidationModel):
    """Step 15. What was measured, and on how many points."""

    n: int = Field(ge=0)
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    spearman: float | None = None
    kendall: float | None = None
    top_k_recall: float | None = None
    top_k: int = 0
    #: How the folds were built, because the two answers differ and it matters.
    scheme: str = "unknown"

    @property
    def is_informative(self) -> bool:
        """Whether these numbers can support any judgement at all."""
        return self.n >= 3 and self.spearman is not None


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """`None` when the actual values are constant, where R² is undefined."""
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total == 0:
        return None
    residual = float(np.sum((actual - predicted) ** 2))
    return 1.0 - residual / total


def spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Rank correlation. `None` when either side has no variation to rank."""
    if actual.size < 2:
        return None
    from scipy.stats import spearmanr

    if np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
        return None
    value = spearmanr(actual, predicted).statistic
    return None if np.isnan(value) else float(value)


def kendall(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if actual.size < 2:
        return None
    from scipy.stats import kendalltau

    if np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
        return None
    value = kendalltau(actual, predicted).statistic
    return None if np.isnan(value) else float(value)


def top_k_recall(actual: np.ndarray, predicted: np.ndarray, k: int) -> float | None:
    """Step 16. Of the true top k, how many the surrogate also ranked top k.

    The metric that most directly reflects the surrogate's job: it does not have
    to reproduce every score, only to put the good candidates near the front.
    """
    if k <= 0 or actual.size < k:
        return None
    true_top = set(np.argsort(actual)[::-1][:k].tolist())
    predicted_top = set(np.argsort(predicted)[::-1][:k].tolist())
    return len(true_top & predicted_top) / k


def evaluate_predictions(
    actual: np.ndarray, predicted: np.ndarray, *, k: int = 3, scheme: str = "holdout"
) -> SurrogateMetrics:
    if actual.size == 0:
        return SurrogateMetrics(n=0, scheme=scheme)
    return SurrogateMetrics(
        n=int(actual.size),
        mae=mean_absolute_error(actual, predicted),
        rmse=root_mean_squared_error(actual, predicted),
        r2=r_squared(actual, predicted),
        spearman=spearman(actual, predicted),
        kendall=kendall(actual, predicted),
        top_k_recall=top_k_recall(actual, predicted, min(k, actual.size)),
        top_k=min(k, int(actual.size)),
        scheme=scheme,
    )


def cross_validate(
    build: object,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    folds: int = 5,
    k: int = 3,
) -> SurrogateMetrics:
    """Step 13. K-fold, falling back to leave-one-out when data is scarce.

    Leave-one-out is chosen automatically below `2 * folds` samples because a
    five-fold split of eight points leaves folds too small to say anything, and
    an arbitrary split would look like a measurement without being one.
    """
    n = features.shape[0]
    if n < 3:
        return SurrogateMetrics(n=0, scheme="insufficient_data")
    effective = min(folds, n) if n >= 2 * folds else n
    scheme = "leave_one_out" if effective == n else f"{effective}_fold"

    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=effective, shuffle=True, random_state=0)
    predictions = np.zeros(n, dtype=float)
    for train_index, test_index in splitter.split(features):
        if train_index.size < 2:
            return SurrogateMetrics(n=0, scheme="insufficient_data")
        model: Surrogate = build()  # type: ignore[operator]
        model.fit(features[train_index], targets[train_index])
        predictions[test_index] = model.predict(features[test_index])
    return evaluate_predictions(targets, predictions, k=k, scheme=scheme)


def prefix_validate(
    build: object,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_train: int = 3,
    k: int = 3,
) -> SurrogateMetrics:
    """Step 14. Predict each observation from the ones that preceded it.

    This is what the surrogate actually does online, so it is the number to
    believe when deciding whether to let it choose candidates. Random folds
    predict the past from the future and read higher than reality.
    """
    n = features.shape[0]
    if n <= minimum_train:
        return SurrogateMetrics(n=0, scheme="insufficient_data")
    actual: list[float] = []
    predicted: list[float] = []
    for index in range(minimum_train, n):
        model: Surrogate = build()  # type: ignore[operator]
        model.fit(features[:index], targets[:index])
        predicted.append(float(model.predict(features[index : index + 1])[0]))
        actual.append(float(targets[index]))
    return evaluate_predictions(
        np.asarray(actual), np.asarray(predicted), k=k, scheme="prefix"
    )


class UncertaintyDiagnostics(ValidationModel):
    """Step 17. Whether the model's uncertainty tracks its actual error."""

    n: int = Field(ge=0)
    #: Correlation between predicted uncertainty and absolute error. A useful
    #: uncertainty is positively correlated with the error it should anticipate.
    error_correlation: float | None = None
    mean_uncertainty: float | None = None
    mean_absolute_error: float | None = None
    #: Fraction of observations inside the one-standard-deviation band. For a
    #: calibrated Gaussian this is about 0.68; far from it means the scale is
    #: wrong even if the ordering is useful.
    coverage_68: float | None = None
    kind: str = "none"

    @property
    def is_useful(self) -> bool:
        """Whether uncertainty carries signal about error at all."""
        return self.error_correlation is not None and self.error_correlation > 0.0


def diagnose_uncertainty(
    actual: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray | None,
    *,
    kind: str = "none",
) -> UncertaintyDiagnostics:
    if uncertainty is None or actual.size == 0:
        return UncertaintyDiagnostics(n=int(actual.size), kind=kind)
    errors = np.abs(actual - predicted)
    correlation: float | None = None
    if actual.size >= 3:
        # `spearman` guards both sides. Constant *errors* are as undefined as a
        # constant uncertainty and happen just as readily: a model that fits its
        # training points exactly has an error of zero everywhere, and asking how
        # uncertainty correlates with that is a question with no answer, not an
        # answer of zero.
        correlation = spearman(uncertainty, errors)
    covered = float(np.mean(errors <= uncertainty)) if uncertainty.size else None
    return UncertaintyDiagnostics(
        n=int(actual.size),
        error_correlation=correlation,
        mean_uncertainty=float(uncertainty.mean()),
        mean_absolute_error=float(errors.mean()),
        coverage_68=covered,
        kind=kind,
    )


def ood_scores(candidates: np.ndarray, training: np.ndarray) -> np.ndarray:
    """Step 18. Distance to the nearest training point, per candidate.

    Euclidean in the encoded space, where every parameter has been mapped onto
    a comparable range. A large value means the model is extrapolating, which is
    worth knowing independently of what its own uncertainty claims.
    """
    if training.size == 0 or candidates.size == 0:
        return np.full(candidates.shape[0], np.inf)
    differences = candidates[:, None, :] - training[None, :, :]
    return np.sqrt((differences**2).sum(axis=2)).min(axis=1)


__all__ = [
    "SurrogateMetrics",
    "UncertaintyDiagnostics",
    "cross_validate",
    "diagnose_uncertainty",
    "evaluate_predictions",
    "kendall",
    "mean_absolute_error",
    "ood_scores",
    "prefix_validate",
    "r_squared",
    "root_mean_squared_error",
    "spearman",
    "top_k_recall",
]
