"""Step 88: compare two strategies on the real NeuroSplit pipeline.

Every evaluation here materializes a real variant, runs TRIBE, analytics,
content analysis, scoring and comparison. There is no stand-in. That is what
makes this the only benchmark whose numbers describe the actual system, and
also why the budget is single digits: each candidate costs tens of minutes.

The two strategies get identical root media, objective, search space, budget and
seed, which is the only way the comparison means anything. What it can show is
narrow: whether one strategy found a better candidate than the other within this
budget, on this surface, once. It is not evidence that either is better in
general, and with a budget this small the difference between them may well sit
inside the pipeline's own nuisance floor. The report says so.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from blackmirror.api.scoring_loader import ScoringLoader
from blackmirror.inference.service import InferenceService
from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.resimulation.backend import ExistingPipelineBackend
from blackmirror.resimulation.orchestrator import ResimulationOrchestrator
from blackmirror.search.budget import SearchBudget
from blackmirror.search.cache import CacheIdentity, CachingEvaluator, SearchEvaluationCache
from blackmirror.search.decode import GenomeDecoder
from blackmirror.search.evaluator import CandidateEvaluator
from blackmirror.search.orchestrator import NeuralSearchOrchestrator
from blackmirror.search.schemas import (
    MEASURED_NUISANCE_FLOOR,
    SearchConfig,
    SearchExperiment,
    SearchStrategyName,
)
from blackmirror.search.space import ContentSearchSpace, continuous
from blackmirror.search.strategy import default_registry

ARTIFACTS = Path("artifacts")
EXPERIMENT = "search-bench"


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -6.0, 6.0, resolution=0.5),
            continuous("brightness", Op.BRIGHTNESS, -0.15, 0.15, resolution=0.01),
        )
    )


def run_one(
    strategy_name: SearchStrategyName,
    root_run: str,
    *,
    budget: int,
    seed: int,
    batch: int,
) -> dict[str, object]:
    manifest = json.loads((ARTIFACTS / "runs" / root_run / "manifest.json").read_text())
    media = Path(manifest["stimulus"]["path"])
    duration = float(manifest["stimulus"]["duration_seconds"])
    fingerprint = manifest["provenance"]["model_fingerprint"]

    scores = ScoringLoader(ARTIFACTS)
    stored = [
        item for item in scores.store.list_for_experiment(EXPERIMENT)
        if root_run in {variant.variant_id for variant in item.variant_scores}
    ]
    if not stored:
        raise SystemExit(
            f"no stored score for {EXPERIMENT} containing {root_run}; create one first"
        )
    score = stored[0]
    objective = score.objectives[0]
    root_value = score.variant_scores[0].objective_scores[0].raw_value

    space = _space()
    search_id = f"bench-{strategy_name.value}-{seed}"
    decoder = GenomeDecoder(
        search_id=search_id,
        experiment_id=EXPERIMENT,
        space=space,
        parent_variant_id=root_run,
        root_media_path=str(media),
        root_duration_seconds=duration,
        objective=objective,
    )
    inference = InferenceService()
    orchestrator_backend = ExistingPipelineBackend(ARTIFACTS, inference)
    evaluator = CandidateEvaluator(
        ARTIFACTS,
        decoder,
        ResimulationOrchestrator(ARTIFACTS, orchestrator_backend),
        objectives=score.objectives,
        objective_set_hash=score.metadata.objective_set_hash,
        objective_definition_hash=score.metadata.objective_definition_hash,
        candidate_dir=ARTIFACTS / "search" / search_id / "candidates",
    )
    # The cache is shared across strategies deliberately: if both propose the
    # same point, paying twice would inflate one strategy's cost for no reason.
    cached = CachingEvaluator(
        evaluator,
        SearchEvaluationCache(
            ARTIFACTS,
            CacheIdentity(
                root_media_sha256=manifest["stimulus"]["sha256"],
                objective_definition_hash=score.metadata.objective_definition_hash,
                space_version=space.search_space_version,
                materialization_version="1.0",
                scoring_version=score.metadata.scoring_version,
                model_fingerprint=fingerprint,
            ),
        ),
        space,
    )

    experiment = SearchExperiment(
        search_id=search_id,
        experiment_id=EXPERIMENT,
        root_run_id=root_run,
        root_media_path=str(media),
        root_media_sha256=manifest["stimulus"]["sha256"],
        objective_set_hash=score.metadata.objective_set_hash,
        objective_definition_hash=score.metadata.objective_definition_hash,
        objectives=score.objectives,
        space=space,
        config=SearchConfig(
            strategy=strategy_name,
            random_seed=seed,
            candidate_batch_size=batch,
            patience=100,
        ),
        budget=SearchBudget(max_candidates=budget, max_generations=budget),
    )
    strategy = default_registry().create(strategy_name)
    search = NeuralSearchOrchestrator(
        experiment, strategy, cached, artifact_root=ARTIFACTS, root_fitness=root_value
    )

    started = time.perf_counter()
    state = search.run(
        on_progress=lambda s: print(
            f"    [{strategy_name.value}] {s.tracker.results[-1].candidate_id} "
            f"= {s.tracker.results[-1].scalar_fitness} "
            f"({s.ledger.wall_seconds / 60:.1f} min spent)",
            flush=True,
        )
    )
    elapsed = time.perf_counter() - started
    report = search.report()
    metrics = state.metrics(experiment.config.minimum_improvement)
    return {
        "strategy": strategy_name.value,
        "search_id": search_id,
        "root_fitness": root_value,
        "best_fitness": metrics.best_fitness,
        "absolute_improvement": metrics.absolute_improvement,
        "improvement_is_resolvable": metrics.improvement_is_resolvable,
        "evaluations": metrics.evaluations,
        "evaluations_to_best": metrics.evaluations_to_best,
        "tribe_runs": metrics.tribe_runs,
        "wall_seconds": elapsed,
        "wall_seconds_to_best": metrics.wall_seconds_to_best,
        "cache_hit_rate": metrics.cache_hit_rate,
        "failure_rate": metrics.failure_rate,
        "stopping_reason": report["stopping_reason"],
        "best_genome": (
            report["best_observed"]["genome"] if report["best_observed"] else None
        ),
        "tied_with_best": list(metrics.tied_candidate_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="20260906T225022Z-630b9b86")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/benchmarks/search_pipeline.json")
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for strategy in (SearchStrategyName.RANDOM_SEARCH, SearchStrategyName.LOCAL_SEARCH):
        print(f"\n=== {strategy.value} (budget {args.budget}, seed {args.seed}) ===", flush=True)
        rows.append(
            run_one(
                strategy, args.root, budget=args.budget, seed=args.seed, batch=args.batch
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'strategy':16} {'best':>10} {'vs root':>10} {'evals':>6} {'to best':>8} {'wall':>8}")
    for row in rows:
        print(
            f"{row['strategy']:16} {row['best_fitness']:>10.5f} "
            f"{row['absolute_improvement']:>+10.5f} {row['evaluations']:>6} "
            f"{row['evaluations_to_best']:>8} "
            f"{row['wall_seconds'] / 60:>7.1f}m"
        )
    print(
        f"\nNuisance floor {MEASURED_NUISANCE_FLOOR:g}. An improvement or a difference "
        f"between strategies smaller than that is not distinguishable from a change "
        f"of implementation detail."
    )
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
