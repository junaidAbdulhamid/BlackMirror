"""Persistence for optimization runs, reviews and candidate specifications.

WHY REVIEWS ARE STORED SEPARATELY FROM RESULTS
    An optimization result is derived data: rerun the pipeline on the same
    inputs and you get it back byte for byte. A human decision is not. Approvals
    and rejections are recorded in their own file so that re-optimizing never
    destroys them, and so that a rejection ("violates brand") survives to inform
    later work.

WHY A PROPOSED VARIANT IS BUILT FROM A REVIEW, NOT A RECOMMENDATION
    Recommendations do not become candidates on their own. A spec is produced
    only from an approved or modified review, which is the human-in-the-loop
    gate the whole phase depends on.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

from blackmirror.optimization.schemas import (
    ApprovalState,
    ContentIntervention,
    OptimizationRecommendation,
    OptimizationResult,
    ProposedVariantSpec,
    RecommendationReview,
)

OPTIMIZATION_DIRNAME = "optimization"


class OptimizationStore:
    def __init__(self, artifact_root: Path) -> None:
        self.root = Path(artifact_root) / "experiments"

    def directory(self, experiment_id: str, request_key: str) -> Path:
        _validate_id(experiment_id, "experiment")
        _validate_key(request_key)
        return self.root / experiment_id / OPTIMIZATION_DIRNAME / request_key

    def write(
        self, result: OptimizationResult, request_key: str, *, overwrite: bool = True
    ) -> Path:
        destination = self.directory(result.request.experiment_id, request_key)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"optimization already exists at {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".optimization.", dir=destination.parent))
        try:
            _write(temporary / "request.json", result.request.model_dump_json(indent=2))
            _write(temporary / "metadata.json", result.model_dump_json(indent=2))
            _write(
                temporary / "weak_intervals.json",
                _list_json([i.model_dump(mode="json") for i in result.weak_intervals]),
            )
            _write(
                temporary / "evidence.json",
                _list_json([e.model_dump(mode="json") for e in result.evidence]),
            )
            _write(
                temporary / "recommendations.json",
                _list_json([r.model_dump(mode="json") for r in result.recommendations]),
            )
            _write(
                temporary / "hypotheses.json",
                _list_json([h.model_dump(mode="json") for h in result.hypotheses]),
            )
            _write(
                temporary / "traces.json",
                _list_json([t.model_dump(mode="json") for t in result.traces]),
            )
            # Human decisions are not derived data. Re-optimizing regenerates
            # everything else, but carrying these across the atomic swap is
            # what stops an approval or a rejection being recomputed away.
            _carry_over(destination, temporary)
            _fsync_directory(temporary)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def read(self, experiment_id: str, request_key: str) -> OptimizationResult:
        path = self.directory(experiment_id, request_key) / "metadata.json"
        return OptimizationResult.model_validate_json(path.read_text(encoding="utf-8"))

    def list_keys(self, experiment_id: str) -> list[str]:
        _validate_id(experiment_id, "experiment")
        base = self.root / experiment_id / OPTIMIZATION_DIRNAME
        if not base.is_dir():
            return []
        return sorted(
            entry.name
            for entry in base.iterdir()
            if entry.is_dir() and (entry / "metadata.json").is_file()
        )

    # --- human decisions -------------------------------------------------

    def reviews(self, experiment_id: str, request_key: str) -> list[RecommendationReview]:
        path = self.directory(experiment_id, request_key) / "reviews.json"
        if not path.is_file():
            return []
        import json

        return [
            RecommendationReview.model_validate(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        ]

    def record_review(
        self, experiment_id: str, request_key: str, review: RecommendationReview
    ) -> list[RecommendationReview]:
        """Append or replace one decision. Reviews survive re-optimization."""
        existing = [
            item
            for item in self.reviews(experiment_id, request_key)
            if item.recommendation_id != review.recommendation_id
        ]
        existing.append(review)
        directory = self.directory(experiment_id, request_key)
        directory.mkdir(parents=True, exist_ok=True)
        _write(
            directory / "reviews.json",
            _list_json([item.model_dump(mode="json") for item in existing]),
        )
        return existing

    def write_spec(
        self, experiment_id: str, request_key: str, spec: ProposedVariantSpec
    ) -> Path:
        directory = self.directory(experiment_id, request_key) / "proposed_variants"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{spec.proposed_variant_id}.json"
        _write(path, spec.model_dump_json(indent=2))
        return path

    def specs(self, experiment_id: str, request_key: str) -> list[ProposedVariantSpec]:
        directory = self.directory(experiment_id, request_key) / "proposed_variants"
        if not directory.is_dir():
            return []
        output: list[ProposedVariantSpec] = []
        for path in sorted(directory.glob("*.json")):
            try:
                output.append(
                    ProposedVariantSpec.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (ValueError, OSError):
                continue
        return output


def build_spec(
    result: OptimizationResult,
    reviews: list[RecommendationReview],
    *,
    proposed_variant_id: str,
) -> ProposedVariantSpec:
    """Turn approved reviews into the Phase 8 input.

    Only APPROVED and MODIFIED reviews contribute. A modified review supplies
    its own interventions, and the original recommendation stays in the stored
    result, so the two are never confused.
    """
    accepted = [r for r in reviews if r.state in (ApprovalState.APPROVED, ApprovalState.MODIFIED)]
    if not accepted:
        raise ValueError(
            "a candidate specification needs at least one approved or modified "
            "recommendation; recommendations never become candidates on their own"
        )
    by_id = {r.recommendation_id: r for r in result.recommendations}
    interventions: list[ContentIntervention] = []
    hypothesis_ids: list[str] = []
    for review in accepted:
        recommendation: OptimizationRecommendation | None = by_id.get(review.recommendation_id)
        if recommendation is None:
            continue
        interventions.extend(
            review.modified_interventions
            if review.state is ApprovalState.MODIFIED and review.modified_interventions
            else recommendation.interventions
        )
        if recommendation.hypothesis_id:
            hypothesis_ids.append(recommendation.hypothesis_id)

    if not interventions:
        raise ValueError("no approved recommendation resolved to an intervention")

    first = next(
        r
        for r in result.recommendations
        if r.recommendation_id == accepted[0].recommendation_id
    )
    return ProposedVariantSpec(
        proposed_variant_id=proposed_variant_id,
        parent_variant_id=result.request.source_variant_id,
        experiment_id=result.request.experiment_id,
        hypothesis_ids=tuple(hypothesis_ids) or (f"HYP-{proposed_variant_id}",),
        interventions=tuple(interventions),
        constraints=result.request.constraints,
        target_objective_id=first.objective_id,
        expected_direction=first.expected_direction,
        edit_instructions=tuple(
            instruction
            for intervention in interventions
            for instruction in intervention.edit_instructions
        ),
        preserve_intervals=tuple(
            (interval.start_seconds, interval.end_seconds)
            for interval in result.strong_intervals
        ),
    )


#: Files and directories that record human decisions rather than derived data.
_HUMAN_ARTIFACTS = ("reviews.json", "proposed_variants")


def _carry_over(existing: Path, temporary: Path) -> None:
    """Copy human-authored artifacts from an existing run into the new one."""
    if not existing.is_dir():
        return
    for name in _HUMAN_ARTIFACTS:
        source = existing / name
        if source.is_dir():
            shutil.copytree(source, temporary / name, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, temporary / name)


def _list_json(items: list[dict]) -> str:
    import json

    return json.dumps(items, indent=2, default=str)


def _write(path: Path, payload: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(descriptor)


def _validate_id(value: str, label: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError(f"invalid {label} id")


def _validate_key(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{8,64}", value) is None:
        raise ValueError("invalid optimization request key")
