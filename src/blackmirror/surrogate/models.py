"""Steps 7-12: the surrogate interface and the families behind it.

THE TWO KINDS OF UNCERTAINTY, AND WHY THE DIFFERENCE MATTERS
    A Gaussian Process returns a posterior standard deviation. It is small near
    observed points and grows with distance from them, and it is derived from an
    explicit model of the function, so Expected Improvement and Probability of
    Improvement have their intended meaning under it.

    A forest returns the spread of its trees. That is disagreement among
    correlated learners fitted to the same data, not a posterior. It behaves
    roughly like uncertainty in the interior of the data and badly outside it:
    extrapolate past the training range and every tree predicts the same edge
    value, so the spread *collapses* exactly where confidence should be lowest.

    Both are exposed, and `uncertainty_kind` records which one a prediction
    carries so a reader cannot mistake the second for the first.

WHY THE DEFAULT IS A GAUSSIAN PROCESS
    The search space is two to five dimensions, observations are deterministic
    and exactly repeatable, and the sample count is tiny. That is the regime GPs
    were designed for and the regime tree ensembles are worst in. The forest
    stays as a baseline, which is what Step 8 asks for.

WHY OBSERVATION NOISE IS NEAR ZERO
    Measured, not assumed: two independent runs of the same stimulus under the
    same configuration produce bit-identical predictions. Re-evaluating a genome
    returns exactly the same number, so the GP's noise term reflects numerical
    tolerance rather than measurement scatter. The 0.0157 nuisance floor is a
    different quantity: sensitivity to a configuration change, not noise in a
    label, and inflating the noise term with it would smooth away real structure.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from blackmirror.surrogate.schemas import SurrogateModelType
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Deterministic pipeline, so this reflects numerical tolerance only.
DETERMINISTIC_NOISE = 1e-8


class SurrogateError(ValueError):
    """The surrogate cannot be fitted or used as asked."""


def _ensure_finite(values: np.ndarray, *, what: str) -> np.ndarray:
    """Refuse to hand back a prediction that is not a number.

    Checked explicitly rather than inferred from numpy's floating-point
    warnings, which cannot be trusted to mean anything on every platform: numpy
    2.2 built against Apple Accelerate raises `divide by zero`, `overflow` and
    `invalid value` for any sufficiently large matmul regardless of its inputs,
    including one whose operands and result are plainly finite. A warning there
    is not evidence of a problem, and its absence elsewhere is not evidence of
    health, so the values themselves are what gets inspected.

    Raising matters because the failure is otherwise silent and expensive. A NaN
    mean flows into the acquisition, `argmax` over NaN returns index 0 without
    complaint, and the loop spends a real evaluation on an arbitrary candidate
    while recording it as chosen.
    """
    if not np.all(np.isfinite(values)):
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise SurrogateError(
            f"{what} contained {bad} non-finite value(s) out of {values.size}; "
            "a surrogate that cannot produce a number must not choose a candidate"
        )
    return values


@runtime_checkable
class Surrogate(Protocol):
    """Step 7. What every surrogate family must provide."""

    model_type: SurrogateModelType
    supports_uncertainty: bool
    uncertainty_kind: str

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None: ...
    def predict(self, features: np.ndarray) -> np.ndarray: ...
    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]: ...
    def hyperparameters(self) -> dict[str, Any]: ...


class _Base:
    """Shared fitting guards and state."""

    model_type: SurrogateModelType
    supports_uncertainty = False
    uncertainty_kind = "none"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self._model: Any = None
        self._trained_on = 0
        self._feature_width = 0

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _check(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.ndim != 2:
            raise SurrogateError("features must be a two-dimensional matrix")
        if features.shape[0] != targets.shape[0]:
            raise SurrogateError(
                f"{features.shape[0]} feature rows against {targets.shape[0]} targets"
            )
        if features.shape[0] < 2:
            raise SurrogateError(
                "a surrogate needs at least two observations; with fewer there is "
                "nothing to interpolate between"
            )
        if not np.all(np.isfinite(targets)):
            raise SurrogateError("targets contain non-finite values")

    def _check_predict(self, features: np.ndarray) -> None:
        if not self.is_fitted:
            raise SurrogateError("surrogate used before fit()")
        if features.ndim != 2 or features.shape[1] != self._feature_width:
            raise SurrogateError(
                f"expected {self._feature_width} features per row, got "
                f"{features.shape[-1] if features.ndim == 2 else 'a non-matrix'}; "
                f"the encoder and the model disagree, which means one of them is stale"
            )

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return self.predict(features), None

    def hyperparameters(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def predict(self, features: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class ForestSurrogate(_Base):
    """Step 8/10. Tree ensemble baseline, with spread as an uncertainty proxy.

    Deliberately the baseline rather than the default. It cannot extrapolate:
    beyond the training range every tree returns the same boundary value, so
    both the prediction flattens and the spread collapses, which is the opposite
    of what an acquisition function needs at the edge of the explored region.
    """

    supports_uncertainty = True
    uncertainty_kind = "ensemble_spread"

    def __init__(
        self,
        *,
        model_type: SurrogateModelType = SurrogateModelType.RANDOM_FOREST,
        n_estimators: int = 200,
        seed: int = 0,
        min_samples_leaf: int = 1,
    ) -> None:
        super().__init__(seed=seed)
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._check(features, targets)
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

        factory = (
            ExtraTreesRegressor
            if self.model_type is SurrogateModelType.EXTRA_TREES
            else RandomForestRegressor
        )
        # bootstrap=False for ExtraTrees keeps every tree on the full sample,
        # which at these sizes matters more than the extra decorrelation.
        self._model = factory(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            min_samples_leaf=self.min_samples_leaf,
        )
        self._model.fit(features, targets)
        self._trained_on = features.shape[0]
        self._feature_width = features.shape[1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check_predict(features)
        return _ensure_finite(
            np.asarray(self._model.predict(features), dtype=float),
            what="ensemble prediction",
        )

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._check_predict(features)
        per_tree = np.stack(
            [tree.predict(features) for tree in self._model.estimators_], axis=0
        )
        return (
            _ensure_finite(per_tree.mean(axis=0), what="ensemble prediction"),
            _ensure_finite(per_tree.std(axis=0), what="ensemble spread"),
        )

    def feature_importance(self) -> np.ndarray:
        """Step 63. Association learned by the model, not a cause of anything."""
        if not self.is_fitted:
            raise SurrogateError("surrogate used before fit()")
        return np.asarray(self._model.feature_importances_, dtype=float)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
        }


class GradientBoostingSurrogate(_Base):
    """Step 10. Available, and not the default: it overfits at this sample size."""

    model_type = SurrogateModelType.GRADIENT_BOOSTING
    supports_uncertainty = False
    uncertainty_kind = "none"

    def __init__(
        self, *, n_estimators: int = 100, learning_rate: float = 0.1, seed: int = 0
    ) -> None:
        super().__init__(seed=seed)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._check(features, targets)
        from sklearn.ensemble import GradientBoostingRegressor

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        )
        self._model.fit(features, targets)
        self._trained_on = features.shape[0]
        self._feature_width = features.shape[1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check_predict(features)
        return _ensure_finite(
            np.asarray(self._model.predict(features), dtype=float),
            what="boosted prediction",
        )

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
        }


class GaussianProcessSurrogate(_Base):
    """Step 9. The default: low dimensions, tiny n, deterministic observations.

    A Matérn kernel with nu=2.5 rather than an RBF. RBF assumes the function is
    infinitely differentiable, which is a strong claim about a response surface
    nobody has characterised; Matérn 2.5 is twice differentiable and is the
    standard choice in Bayesian optimization for exactly that reason.

    Targets are centred before fitting because the GP's prior mean is zero, and
    objective values here sit near -0.023. Without centring the posterior is
    pulled toward zero everywhere and the model spends its capacity representing
    an offset.
    """

    model_type = SurrogateModelType.GAUSSIAN_PROCESS
    supports_uncertainty = True
    uncertainty_kind = "posterior_standard_deviation"

    def __init__(
        self,
        *,
        length_scale: float = 0.5,
        nu: float = 2.5,
        noise: float = DETERMINISTIC_NOISE,
        normalize_targets: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.length_scale = length_scale
        self.nu = nu
        self.noise = noise
        self.normalize_targets = normalize_targets
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._at_bounds = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._check(features, targets)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        centred = targets.astype(float)
        if self.normalize_targets:
            self._target_mean = float(centred.mean())
            spread = float(centred.std())
            # A constant target has no scale to divide by; leaving it at one
            # keeps the model well defined and its predictions constant, which
            # is the honest answer for a constant function.
            self._target_scale = spread if spread > 0 else 1.0
            centred = (centred - self._target_mean) / self._target_scale
        else:
            self._target_mean, self._target_scale = 0.0, 1.0

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=self.length_scale, length_scale_bounds=(1e-2, 2.0), nu=self.nu
        ) + WhiteKernel(noise_level=self.noise, noise_level_bounds=(1e-12, 1e-2))
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            random_state=self.seed,
            n_restarts_optimizer=3,
        )
        # On a smooth surface the amplitude and the length scale are only weakly
        # identified together: a long length scale needs a large amplitude to
        # produce the same variance, so the optimizer walks to a bound and
        # sklearn reports non-convergence. Measured across sample sizes and
        # bound choices, rank correlation stayed at 1.0 throughout, so this is a
        # degeneracy in the parameterisation rather than a failure to fit.
        #
        # The warning is silenced and the fact recorded instead: `at_bounds`
        # surfaces in the metadata, so a genuinely stuck fit is still visible.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._model.fit(features, centred)
        self._trained_on = features.shape[0]
        self._feature_width = features.shape[1]
        self._at_bounds = self._hyperparameters_at_bounds()

    def predict(self, features: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(features)
        return mean

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._check_predict(features)
        mean, std = self._model.predict(features, return_std=True)
        # Undo the centring so predictions live in the objective's own units,
        # which is what every acquisition and every comparison expects.
        return (
            _ensure_finite(
                np.asarray(mean, dtype=float) * self._target_scale + self._target_mean,
                what="posterior mean",
            ),
            _ensure_finite(
                np.asarray(std, dtype=float) * self._target_scale,
                what="posterior standard deviation",
            ),
        )

    def _hyperparameters_at_bounds(self) -> bool:
        """Whether any fitted hyperparameter sits on its bound.

        Recorded rather than raised. At a bound the value is not identified by
        the data, so the kernel's numbers should not be read as estimates even
        though the predictions may be excellent.
        """
        kernel = self._model.kernel_
        theta = kernel.theta
        bounds = kernel.bounds
        if theta.size == 0:  # pragma: no cover - every kernel here has parameters
            return False
        tolerance = 1e-6
        return bool(
            np.any(np.abs(theta - bounds[:, 0]) < tolerance)
            or np.any(np.abs(theta - bounds[:, 1]) < tolerance)
        )

    @property
    def hyperparameters_at_bounds(self) -> bool:
        return self._at_bounds

    def learned_kernel(self) -> str:
        """Step 65. The fitted kernel, for diagnostics."""
        if not self.is_fitted:
            raise SurrogateError("surrogate used before fit()")
        return str(self._model.kernel_)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "kernel": "constant * matern + white",
            "nu": self.nu,
            "length_scale": self.length_scale,
            "noise": self.noise,
            "learned_kernel": self.learned_kernel() if self.is_fitted else None,
            "hyperparameters_at_bounds": self._at_bounds,
        }


class SurrogateRegistry:
    """Step 12. Only families that are implemented and tested."""

    def __init__(self, factories: dict[SurrogateModelType, Any] | None = None) -> None:
        self._factories: dict[SurrogateModelType, Any] = dict(
            factories
            or {
                SurrogateModelType.GAUSSIAN_PROCESS: GaussianProcessSurrogate,
                SurrogateModelType.RANDOM_FOREST: lambda **kw: ForestSurrogate(
                    model_type=SurrogateModelType.RANDOM_FOREST, **kw
                ),
                SurrogateModelType.EXTRA_TREES: lambda **kw: ForestSurrogate(
                    model_type=SurrogateModelType.EXTRA_TREES, **kw
                ),
                SurrogateModelType.GRADIENT_BOOSTING: GradientBoostingSurrogate,
            }
        )

    def create(self, model_type: SurrogateModelType, **kwargs: Any) -> Surrogate:
        factory = self._factories.get(model_type)
        if factory is None:
            available = ", ".join(sorted(item.value for item in self._factories))
            raise SurrogateError(
                f"surrogate {model_type.value!r} is not implemented; available: {available}"
            )
        return factory(**kwargs)

    def available(self) -> tuple[SurrogateModelType, ...]:
        return tuple(sorted(self._factories, key=lambda item: item.value))

    def __contains__(self, model_type: object) -> bool:
        return model_type in self._factories


def default_registry() -> SurrogateRegistry:
    return SurrogateRegistry()


__all__ = [
    "DETERMINISTIC_NOISE",
    "ForestSurrogate",
    "GaussianProcessSurrogate",
    "GradientBoostingSurrogate",
    "Surrogate",
    "SurrogateError",
    "SurrogateRegistry",
    "default_registry",
]
