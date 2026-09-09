"""Steps 20-25, 28, 45, 47: deciding which candidate is worth a real evaluation.

THE PROBLEM AN ACQUISITION FUNCTION SOLVES
    The surrogate can score ten thousand candidates in under a second. Only a
    handful can be truly evaluated. Picking the highest predicted score is the
    obvious rule and it is a trap: it sends every evaluation into the region the
    model already understands, so the model never learns anything new and the
    search converges on whatever its early data happened to suggest.

    An acquisition function values a candidate by what evaluating it would be
    worth, which combines how good it might be with how unsure the model is.
    High predicted score is exploitation; high uncertainty is exploration.

THE FORMULAE, AND WHAT THEY ASSUME
    Let mu(x) be the predicted score, sigma(x) its standard deviation, and f*
    the best real observation so far.

        UCB(x)  = mu(x) + kappa * sigma(x)
        PI(x)   = Phi(z),                      z = (mu - f* - xi) / sigma
        EI(x)   = (mu - f* - xi) * Phi(z) + sigma * phi(z)

    where Phi and phi are the standard normal CDF and PDF. EI and PI are exact
    only if the predictive distribution really is Gaussian, which is true for a
    Gaussian Process and is *not* true for the spread of a forest's trees. UCB
    makes no distributional claim, only that sigma orders uncertainty sensibly,
    so it degrades gracefully where the others mislead.

    That is why the ensemble path warns rather than silently computing EI on a
    quantity that does not have the required semantics.
"""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict

from blackmirror.surrogate.schemas import SurrogatePrediction


class AcquisitionName(StrEnum):
    GREEDY_MEAN = "greedy_mean"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"


class AcquisitionError(ValueError):
    """The acquisition cannot be computed as asked."""


class AcquisitionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class AcquisitionScore(AcquisitionModel):
    """Step 34. One candidate's value, and everything behind that number."""

    candidate_id: str
    value: float
    predicted_mean: float
    predicted_uncertainty: float | None = None
    incumbent: float | None = None
    ood_score: float | None = None
    feasibility_probability: float | None = None
    acquisition: AcquisitionName
    #: Plain-language reason, so a selection can be explained rather than trusted.
    reason: str = ""


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


def greedy_mean(
    mean: np.ndarray, uncertainty: np.ndarray | None = None, **_: object
) -> np.ndarray:
    """Step 24. The baseline: value a candidate at its predicted score.

    Takes `uncertainty` and ignores it, so every acquisition shares one calling
    convention. Included precisely so the uncertainty-aware ones have something
    to beat: this is pure exploitation and will happily re-sample one region
    forever.
    """
    return np.asarray(mean, dtype=float)


def upper_confidence_bound(
    mean: np.ndarray, uncertainty: np.ndarray | None = None, *, kappa: float = 2.0, **_: object
) -> np.ndarray:
    """Step 22. mu + kappa * sigma. Higher kappa explores more."""
    if uncertainty is None:
        raise AcquisitionError(
            "upper confidence bound needs a predictive uncertainty; this surrogate "
            "does not supply one, so use greedy_mean or a model that does"
        )
    return np.asarray(mean, dtype=float) + kappa * np.asarray(uncertainty, dtype=float)


def expected_improvement(
    mean: np.ndarray,
    uncertainty: np.ndarray | None = None,
    *,
    incumbent: float | None = None,
    xi: float = 0.0,
    **_: object,
) -> np.ndarray:
    """Step 21. Expected size of the improvement over the incumbent.

    Zero uncertainty collapses this to the improvement itself when positive and
    to zero otherwise, which is the correct limit: with no doubt about the
    prediction there is nothing to gain from a candidate that cannot beat the
    incumbent.
    """
    if uncertainty is None:
        raise AcquisitionError(
            "expected improvement needs a predictive uncertainty; this surrogate "
            "does not supply one"
        )
    if incumbent is None:
        raise AcquisitionError(
            "expected improvement is defined against the best real observation; "
            "with no observations there is no incumbent to improve on"
        )
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(uncertainty, dtype=float)
    gain = mu - incumbent - xi
    out = np.maximum(gain, 0.0)
    positive = sigma > 1e-12
    if np.any(positive):
        z = gain[positive] / sigma[positive]
        out[positive] = gain[positive] * _normal_cdf(z) + sigma[positive] * _normal_pdf(z)
    return out


def probability_of_improvement(
    mean: np.ndarray,
    uncertainty: np.ndarray | None = None,
    *,
    incumbent: float | None = None,
    xi: float = 0.0,
    **_: object,
) -> np.ndarray:
    """Step 23. Chance of beating the incumbent by at least xi.

    Reports probability, so it is only meaningful when the predictive
    distribution genuinely is Gaussian. Use it with a Gaussian Process; with an
    ensemble spread the number looks like a probability and is not one.
    """
    if uncertainty is None:
        raise AcquisitionError("probability of improvement needs a predictive uncertainty")
    if incumbent is None:
        raise AcquisitionError("probability of improvement needs an incumbent")
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(uncertainty, dtype=float)
    out = (mu > incumbent + xi).astype(float)
    positive = sigma > 1e-12
    if np.any(positive):
        z = (mu[positive] - incumbent - xi) / sigma[positive]
        out[positive] = _normal_cdf(z)
    return out


class AcquisitionRegistry:
    """Step 25. Named lookup for the acquisitions that exist."""

    def __init__(self) -> None:
        self._functions = {
            AcquisitionName.GREEDY_MEAN: greedy_mean,
            AcquisitionName.UPPER_CONFIDENCE_BOUND: upper_confidence_bound,
            AcquisitionName.EXPECTED_IMPROVEMENT: expected_improvement,
            AcquisitionName.PROBABILITY_OF_IMPROVEMENT: probability_of_improvement,
        }

    def get(self, name: AcquisitionName) -> object:
        function = self._functions.get(name)
        if function is None:  # pragma: no cover - the enum is exhaustive
            raise AcquisitionError(f"acquisition {name.value!r} is not implemented")
        return function

    def available(self) -> tuple[AcquisitionName, ...]:
        return tuple(sorted(self._functions, key=lambda item: item.value))

    def requires_uncertainty(self, name: AcquisitionName) -> bool:
        return name is not AcquisitionName.GREEDY_MEAN

    def assumes_gaussian(self, name: AcquisitionName) -> bool:
        """Which acquisitions read sigma as a real predictive distribution."""
        return name in (
            AcquisitionName.EXPECTED_IMPROVEMENT,
            AcquisitionName.PROBABILITY_OF_IMPROVEMENT,
        )


def score_candidates(
    predictions: list[SurrogatePrediction],
    *,
    acquisition: AcquisitionName,
    incumbent: float | None,
    kappa: float = 2.0,
    xi: float = 0.0,
    feasibility: np.ndarray | None = None,
) -> list[AcquisitionScore]:
    """Value every prediction, recording why each got the number it did.

    `feasibility` implements Step 45: multiplying by an estimated probability of
    satisfying the guardrails downranks candidates likely to be disqualified. It
    is only applied when the caller supplies it, because an unreliable
    feasibility model would suppress good candidates for no reason.
    """
    if not predictions:
        return []
    registry = AcquisitionRegistry()
    function = registry.get(acquisition)
    mean = np.asarray([item.predicted_score for item in predictions], dtype=float)
    has_uncertainty = all(item.uncertainty is not None for item in predictions)
    sigma = (
        np.asarray([item.uncertainty for item in predictions], dtype=float)
        if has_uncertainty
        else None
    )
    values = function(mean, sigma, incumbent=incumbent, kappa=kappa, xi=xi)  # type: ignore[operator]
    if feasibility is not None:
        values = np.asarray(values, dtype=float) * np.asarray(feasibility, dtype=float)

    out: list[AcquisitionScore] = []
    for index, prediction in enumerate(predictions):
        out.append(
            AcquisitionScore(
                candidate_id=prediction.candidate_id,
                value=float(values[index]),
                predicted_mean=prediction.predicted_score,
                predicted_uncertainty=prediction.uncertainty,
                incumbent=incumbent,
                ood_score=prediction.ood_score,
                feasibility_probability=(
                    None if feasibility is None else float(feasibility[index])
                ),
                acquisition=acquisition,
                reason=_explain(acquisition, prediction, incumbent, kappa, xi),
            )
        )
    return out


def _explain(
    acquisition: AcquisitionName,
    prediction: SurrogatePrediction,
    incumbent: float | None,
    kappa: float,
    xi: float,
) -> str:
    band = "" if prediction.uncertainty is None else f", uncertainty {prediction.uncertainty:.4f}"
    if acquisition is AcquisitionName.GREEDY_MEAN:
        return f"predicted {prediction.predicted_score:.4f}{band}; chosen on prediction alone"
    if acquisition is AcquisitionName.UPPER_CONFIDENCE_BOUND:
        return (
            f"predicted {prediction.predicted_score:.4f}{band}; optimistic bound at "
            f"kappa={kappa:g}"
        )
    reference = "no incumbent" if incumbent is None else f"best so far {incumbent:.4f}"
    return (
        f"predicted {prediction.predicted_score:.4f}{band}; weighed against {reference} "
        f"with xi={xi:g}"
    )


def select_diverse_batch(
    scores: list[AcquisitionScore],
    genomes: dict[str, dict[str, float]],
    distance: object,
    *,
    count: int,
    minimum_distance: float = 0.05,
) -> list[AcquisitionScore]:
    """Step 28. Take the best, then the best that is far enough from it.

    Ranking by acquisition alone returns near-identical points, because a smooth
    surrogate gives neighbouring candidates near-identical values. Spending
    several hour-long evaluations on one point is the most expensive way to
    learn nothing, so each pick must be at least `minimum_distance` from the
    ones already chosen. If nothing qualifies, the constraint is relaxed rather
    than returning a short batch, and that relaxation is recorded.
    """
    if count <= 0 or not scores:
        return []
    ranked = sorted(scores, key=lambda item: -item.value)
    chosen: list[AcquisitionScore] = [ranked[0]]
    for candidate in ranked[1:]:
        if len(chosen) >= count:
            break
        far_enough = all(
            distance(  # type: ignore[operator]
                genomes[candidate.candidate_id], genomes[picked.candidate_id]
            )
            >= minimum_distance
            for picked in chosen
        )
        if far_enough:
            chosen.append(candidate)
    if len(chosen) < count:
        for candidate in ranked[1:]:
            if len(chosen) >= count:
                break
            if candidate.candidate_id not in {item.candidate_id for item in chosen}:
                chosen.append(
                    candidate.model_copy(
                        update={
                            "reason": (
                                f"{candidate.reason}; diversity constraint of "
                                f"{minimum_distance:g} relaxed to fill the batch"
                            )
                        }
                    )
                )
    return chosen[:count]


__all__ = [
    "AcquisitionError",
    "AcquisitionName",
    "AcquisitionRegistry",
    "AcquisitionScore",
    "expected_improvement",
    "greedy_mean",
    "probability_of_improvement",
    "score_candidates",
    "select_diverse_batch",
    "upper_confidence_bound",
]
