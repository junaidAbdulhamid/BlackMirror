"""Read/create Phase 7 optimizations over persisted artifacts.

Optimization is deterministic and fast (7 ms on a real two-variant experiment),
so the create path recomputes rather than queueing. It never runs TRIBE, never
edits media, and never turns a recommendation into a candidate on its own -- a
candidate requires a recorded human decision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from blackmirror.optimization.engine import (
    OptimizationOrchestrator,
    content_series_from_arrays,
)
from blackmirror.optimization.evidence import ContentSeries
from blackmirror.optimization.schemas import (
    OptimizationRequest,
    OptimizationResult,
    ProposedVariantSpec,
    RecommendationReview,
)
from blackmirror.optimization.storage import OptimizationStore, build_spec
from blackmirror.scoring.storage import ScoringStore
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationLoadError(ValueError):
    """The optimization cannot be assembled from what is stored."""


def request_key(request: OptimizationRequest) -> str:
    """Stable identity for a request, so an identical one lands in one place."""
    payload = request.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


class OptimizationLoader:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store = OptimizationStore(self.artifact_root)
        self.scores = ScoringStore(self.artifact_root)
        self.orchestrator = OptimizationOrchestrator()

    def create(
        self, request: OptimizationRequest, *, variant_runs: dict[str, str] | None = None
    ) -> OptimizationResult:
        try:
            score = self.scores.read(request.experiment_id, request.objective_set_hash)
        except (OSError, ValueError) as exc:
            raise OptimizationLoadError(
                f"no stored score for experiment {request.experiment_id!r} and objective set "
                f"{request.objective_set_hash[:16]!r}; score the experiment first"
            ) from exc

        mapping = variant_runs or {
            variant.variant_id: variant.variant_id for variant in score.variant_scores
        }
        content, events, warnings = self._content(mapping)
        result = self.orchestrator.optimize(
            request, score, content=content, events=events
        )
        if warnings:
            result = result.model_copy(
                update={"warnings": (*result.warnings, *warnings)}
            )
        self.store.write(result, request_key(request))
        return result

    def read(self, experiment_id: str, key: str) -> OptimizationResult:
        return self.store.read(experiment_id, key)

    def keys(self, experiment_id: str) -> list[str]:
        return self.store.list_keys(experiment_id)

    def reviews(self, experiment_id: str, key: str) -> list[RecommendationReview]:
        return self.store.reviews(experiment_id, key)

    def review(
        self, experiment_id: str, key: str, review: RecommendationReview
    ) -> list[RecommendationReview]:
        result = self.store.read(experiment_id, key)
        known = {item.recommendation_id for item in result.recommendations}
        if review.recommendation_id not in known:
            raise OptimizationLoadError(
                f"recommendation {review.recommendation_id!r} is not part of this optimization"
            )
        return self.store.record_review(experiment_id, key, review)

    def build_candidate(
        self, experiment_id: str, key: str, proposed_variant_id: str
    ) -> ProposedVariantSpec:
        result = self.store.read(experiment_id, key)
        reviews = self.store.reviews(experiment_id, key)
        spec = build_spec(result, reviews, proposed_variant_id=proposed_variant_id)
        self.store.write_spec(experiment_id, key, spec)
        return spec

    def specs(self, experiment_id: str, key: str) -> list[ProposedVariantSpec]:
        return self.store.specs(experiment_id, key)

    def _content(
        self, mapping: dict[str, str]
    ) -> tuple[dict[str, ContentSeries], dict[str, list[dict[str, object]]], list[str]]:
        """Load each variant's Phase 4 features and events.

        The array a run's metadata *references* is loaded, not whichever feature
        file happens to be on disk: a run directory accumulates one per
        analysis, and pairing a current name list with a stale array is a
        version mismatch waiting to happen.
        """
        content: dict[str, ContentSeries] = {}
        events: dict[str, list[dict[str, object]]] = {}
        warnings: list[str] = []
        for variant_id, run_id in mapping.items():
            directory = self.artifact_root / "runs" / run_id / "content_analysis"
            manifest = directory / "metadata.json"
            if not manifest.is_file():
                warnings.append(
                    f"variant {variant_id} has no content analysis, so no content evidence "
                    f"could be gathered for it"
                )
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                # Structural markers (hook, product reveal) live outside
                # `events` because `events` must stay a contiguous partition.
                # Scope resolution needs both, and does not need a partition.
                events[variant_id] = [
                    *payload.get("events", []),
                    *payload.get("structural_markers", []),
                ]
                arrays = payload.get("arrays") or {}
                names = tuple(arrays.get("feature_names") or ())
                referenced = str(arrays.get("path") or "").split("/")[-1]
                path = directory / referenced
                if not names or not referenced or not path.is_file():
                    warnings.append(
                        f"variant {variant_id} has no referenced feature array; content "
                        f"feature evidence is unavailable for it"
                    )
                    continue
                with np.load(path, allow_pickle=False) as archive:
                    content[variant_id] = content_series_from_arrays(
                        variant_id, archive["feature_times"], archive["features"], names
                    )
            except (OSError, ValueError, KeyError) as exc:
                warnings.append(f"variant {variant_id} content could not be read: {exc}")
        return content, events, warnings


__all__ = ["OptimizationLoadError", "OptimizationLoader", "request_key"]
