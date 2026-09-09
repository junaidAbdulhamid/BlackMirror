"""Steps 30-31, 36-38, 50-53: fitting the surrogate and deciding to trust it.

THE TRUST GATE IS THE POINT OF THIS MODULE
    A surrogate that cannot rank candidates is worse than no surrogate. It will
    confidently steer every expensive evaluation into a region it has
    misunderstood, and because each evaluation costs tens of minutes, a bad
    model wastes more budget than the search had to begin with.

    So the surrogate does not get to decide anything until it has demonstrated
    rank skill on held-out data, and it loses that right the moment it stops
    demonstrating it. Falling back to a Phase 9 strategy is not a failure mode;
    it is the correct behaviour when the evidence does not support handing over
    control.

WHY THE GATE USES ORDER-AWARE VALIDATION
    Random folds let the model predict an early candidate using later ones,
    which is information it will never have online. That reads higher than
    reality, and a gate tuned against it would open too easily.

WHY THERE IS A MINIMUM SAMPLE COUNT AT ALL
    Below a handful of observations, cross-validation produces a number but not
    a measurement: one fold's luck moves it entirely. The bootstrap phase exists
    so that early evaluations are spent covering the space rather than being
    steered by a model fitted to three points.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from blackmirror.surrogate.encoder import GenomeFeatureEncoder
from blackmirror.surrogate.models import (
    Surrogate,
    SurrogateError,
    default_registry,
)
from blackmirror.surrogate.schemas import (
    SurrogateDataset,
    SurrogateModelMetadata,
    SurrogateModelType,
    SurrogatePrediction,
    SurrogateState,
)
from blackmirror.surrogate.validation import (
    SurrogateMetrics,
    UncertaintyDiagnostics,
    cross_validate,
    diagnose_uncertainty,
    ood_scores,
    prefix_validate,
)
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class TrustPolicy(BaseModel):
    """Step 50. The bar a surrogate must clear to be allowed to choose."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    #: Step 31. Below this, the surrogate does not run at all.
    min_samples: int = Field(default=8, ge=2)
    #: Rank correlation on order-aware validation. The primary gate, because
    #: ranking is the decision the surrogate is actually making.
    min_spearman: float = Field(default=0.2, ge=-1.0, le=1.0)
    #: Of the true best k, how many the surrogate also placed in its top k.
    min_top_k_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1)
    #: Retrain cadence. One is right for small datasets, where every
    #: observation is a large fraction of what is known.
    retrain_every: int = Field(default=1, ge=1)

    def verdict(self, metrics: SurrogateMetrics, samples: int) -> tuple[bool, str]:
        """Whether the surrogate may drive selection, and why not if it may not."""
        if samples < self.min_samples:
            return False, (
                f"{samples} real observations, below the minimum of {self.min_samples}; "
                f"still bootstrapping"
            )
        if metrics.spearman is None:
            return False, (
                "validation produced no rank correlation, usually because the "
                "observed objective values do not vary enough to rank"
            )
        if metrics.spearman < self.min_spearman:
            return False, (
                f"rank correlation {metrics.spearman:.3f} on {metrics.scheme} validation "
                f"is below the required {self.min_spearman:.2f}; the surrogate cannot "
                f"order candidates well enough to be worth an expensive evaluation"
            )
        if self.min_top_k_recall is not None and (
            metrics.top_k_recall is None or metrics.top_k_recall < self.min_top_k_recall
        ):
            found = "none" if metrics.top_k_recall is None else f"{metrics.top_k_recall:.2f}"
            return False, (
                f"top-{metrics.top_k} recall {found} is below the required "
                f"{self.min_top_k_recall:.2f}"
            )
        return True, (
            f"rank correlation {metrics.spearman:.3f} on {metrics.scheme} validation "
            f"over {samples} real observations"
        )


class TrainedSurrogate(BaseModel):
    """A fitted model plus everything needed to judge and reproduce it."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    metadata: SurrogateModelMetadata
    cross_validation: SurrogateMetrics
    prefix_validation: SurrogateMetrics
    uncertainty: UncertaintyDiagnostics
    state: SurrogateState
    trusted: bool
    trust_reason: str
    #: The label spread, so a caller can see when the targets are too flat to
    #: learn from whatever the correlation happens to say.
    target_spread: float = 0.0

    @property
    def usable(self) -> bool:
        return self.trusted and self.state is SurrogateState.READY


class SurrogateTrainer:
    """Fits, validates, gates and predicts. Owns no search logic."""

    def __init__(
        self,
        encoder: GenomeFeatureEncoder,
        *,
        model_type: SurrogateModelType = SurrogateModelType.GAUSSIAN_PROCESS,
        policy: TrustPolicy | None = None,
        seed: int = 0,
    ) -> None:
        self.encoder = encoder
        self.model_type = model_type
        self.policy = policy or TrustPolicy()
        self.seed = seed
        self.registry = default_registry()
        self._model: Surrogate | None = None
        self._version = 0
        self._training_features: np.ndarray | None = None
        self._dataset_hash = ""
        self._dataset_size = 0

    # --- fitting ----------------------------------------------------------

    def _build(self) -> Surrogate:
        return self.registry.create(self.model_type, seed=self.seed)

    def fit(self, dataset: SurrogateDataset) -> TrainedSurrogate:
        """Train, validate both ways, and decide whether to trust the result."""
        if not dataset.consistent_genomes():
            raise SurrogateError(
                "the dataset mixes genomes with different parameter names; the "
                "search space changed and these labels cannot share a model"
            )
        genomes = [item.genome for item in dataset.records]
        targets = dataset.targets()
        features = self.encoder.encode_dataset(genomes)

        if dataset.size < 2:
            return self._untrained(
                dataset,
                SurrogateState.BOOTSTRAPPING,
                f"{dataset.size} observation(s); at least two are needed to fit anything",
            )

        model = self._build()
        model.fit(features, targets)
        self._model = model
        self._version += 1
        self._training_features = features
        self._dataset_hash = dataset.dataset_hash()
        self._dataset_size = dataset.size

        cv = cross_validate(self._build, features, targets, k=self.policy.top_k)
        prefix = prefix_validate(
            self._build,
            features,
            targets,
            minimum_train=max(2, min(3, dataset.size - 1)),
            k=self.policy.top_k,
        )
        # The gate reads the order-aware number when there is one, because that
        # is what the model will actually be doing.
        gating = prefix if prefix.is_informative else cv
        trusted, reason = self.policy.verdict(gating, dataset.size)

        fitted_mean, fitted_sigma = model.predict_with_uncertainty(features)
        diagnostics = diagnose_uncertainty(
            targets, fitted_mean, fitted_sigma, kind=model.uncertainty_kind
        )

        state = SurrogateState.READY if trusted else SurrogateState.DEGRADED
        if dataset.size < self.policy.min_samples:
            state = SurrogateState.BOOTSTRAPPING

        return TrainedSurrogate(
            metadata=SurrogateModelMetadata(
                model_type=self.model_type,
                model_version=self._version,
                training_dataset_hash=self._dataset_hash,
                training_dataset_size=dataset.size,
                dataset_identity_fingerprint=dataset.identity.fingerprint(),
                feature_names=self.encoder.feature_names,
                hyperparameters={
                    key: value
                    for key, value in model.hyperparameters().items()
                    if isinstance(value, float | int | str | bool | type(None))
                },
                training_seed=self.seed,
                supports_uncertainty=model.supports_uncertainty,
                uncertainty_kind=model.uncertainty_kind,
                trained_at=dt.datetime.now(dt.UTC),
            ),
            cross_validation=cv,
            prefix_validation=prefix,
            uncertainty=diagnostics,
            state=state,
            trusted=trusted,
            trust_reason=reason,
            target_spread=dataset.target_spread(),
        )

    def _untrained(
        self, dataset: SurrogateDataset, state: SurrogateState, reason: str
    ) -> TrainedSurrogate:
        return TrainedSurrogate(
            metadata=SurrogateModelMetadata(
                model_type=self.model_type,
                model_version=max(1, self._version),
                training_dataset_hash=dataset.dataset_hash(),
                training_dataset_size=dataset.size,
                dataset_identity_fingerprint=dataset.identity.fingerprint(),
                feature_names=self.encoder.feature_names,
                training_seed=self.seed,
            ),
            cross_validation=SurrogateMetrics(n=0, scheme="insufficient_data"),
            prefix_validation=SurrogateMetrics(n=0, scheme="insufficient_data"),
            uncertainty=UncertaintyDiagnostics(n=0),
            state=state,
            trusted=False,
            trust_reason=reason,
            target_spread=dataset.target_spread(),
        )

    # --- predicting -------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def predict(
        self, candidates: dict[str, dict[str, float]]
    ) -> list[SurrogatePrediction]:
        """Cheap estimates for many candidates, each labelled as an estimate."""
        if self._model is None:
            raise SurrogateError("predict() called before fit()")
        if not candidates:
            return []
        ids = list(candidates)
        features = self.encoder.encode_dataset([candidates[key] for key in ids])
        mean, sigma = self._model.predict_with_uncertainty(features)
        distances = (
            ood_scores(features, self._training_features)
            if self._training_features is not None
            else np.full(len(ids), np.inf)
        )
        return [
            SurrogatePrediction(
                candidate_id=candidate_id,
                genome=candidates[candidate_id],
                predicted_score=float(mean[index]),
                uncertainty=None if sigma is None else float(sigma[index]),
                ood_score=float(distances[index]) if np.isfinite(distances[index]) else None,
                model_type=self.model_type,
                model_version=self._version,
                training_dataset_size=self._dataset_size,
                training_dataset_hash=self._dataset_hash,
            )
            for index, candidate_id in enumerate(ids)
        ]

    def should_retrain(self, evaluations_since: int) -> bool:
        return evaluations_since >= self.policy.retrain_every


def compare_families(
    encoder: GenomeFeatureEncoder,
    dataset: SurrogateDataset,
    *,
    families: tuple[SurrogateModelType, ...] | None = None,
    seed: int = 0,
) -> dict[SurrogateModelType, SurrogateMetrics]:
    """Step 38. Score each family on the same data, using order-aware folds.

    For choosing a default with evidence rather than by preference. Deliberately
    returns the numbers instead of switching anything: at these sample sizes a
    model that wins one round can lose the next, and a search whose surrogate
    changes family every evaluation is not reproducible.
    """
    registry = default_registry()
    selected = families or registry.available()
    genomes = [item.genome for item in dataset.records]
    features = encoder.encode_dataset(genomes)
    targets = dataset.targets()

    out: dict[SurrogateModelType, SurrogateMetrics] = {}
    for family in selected:
        def build(family: SurrogateModelType = family) -> Surrogate:
            return registry.create(family, seed=seed)

        try:
            metrics = prefix_validate(
                build, features, targets, minimum_train=max(2, min(3, dataset.size - 1))
            )
            if not metrics.is_informative:
                metrics = cross_validate(build, features, targets)
        except SurrogateError as exc:  # pragma: no cover - reported, not raised
            logger.warning("family %s could not be validated: %s", family.value, exc)
            continue
        out[family] = metrics
    return out


__all__ = ["SurrogateTrainer", "TrainedSurrogate", "TrustPolicy", "compare_families"]
