"""Steps 74-76: how many expensive evaluations the surrogate actually saves.

WHAT THIS MEASURES
    Not "is the surrogate accurate". The question that decides whether Phase 10
    was worth building is narrower: reaching the same best-observed score, how
    many real evaluations did surrogate-assisted search need compared with
    random sampling and with a Phase 9 strategy?

    That is a sample-efficiency question, and on a synthetic surface it is
    answerable exactly, because the optimum is known and evaluation is free.

WHY SYNTHETIC
    On the real pipeline the optimum is unknown, one evaluation costs tens of
    minutes, and the corpus's label spread sits below the pipeline's own
    nuisance floor. A real comparison at this budget could not distinguish the
    methods from each other or from noise. Here the answer is checkable, and
    what transfers is the mechanism, not the number.

    Every method gets the same surface, the same budget, the same seeds and the
    same search space, which is what makes the comparison mean anything at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.search.budget import SearchBudget
from blackmirror.search.fitness import CandidateFitness, ComputeCost, FitnessStatus
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.schemas import SearchStrategyName
from blackmirror.search.space import ContentSearchSpace, continuous
from blackmirror.search.strategy import default_registry as strategy_registry
from blackmirror.surrogate.acquisition import AcquisitionName
from blackmirror.surrogate.orchestrator import (
    BayesianOptimizationConfig,
    BayesianOptimizationOrchestrator,
)
from blackmirror.surrogate.pool import PoolConfig
from blackmirror.surrogate.schemas import DatasetIdentity
from blackmirror.surrogate.trainer import TrustPolicy

Surface = Callable[[dict[str, float]], float]


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),
            continuous("brightness", Op.BRIGHTNESS, -1.0, 1.0, resolution=0.01),
        )
    )


def one_peak(values: dict[str, float]) -> float:
    """A single smooth maximum at (3, 0.4). Optimum 0."""
    return -((values["gain"] - 3.0) ** 2) - 10.0 * (values["brightness"] - 0.4) ** 2


def two_peaks(values: dict[str, float]) -> float:
    """A broad low peak and a tall narrow one, to punish pure exploitation."""
    x, y = values["gain"], values["brightness"]
    low = 2.0 - 0.05 * ((x + 6.0) ** 2 + 10.0 * y**2)
    high = 6.0 - 0.6 * ((x - 6.0) ** 2 + 10.0 * (y - 0.5) ** 2)
    return max(low, high)


SURFACES: dict[str, tuple[Surface, float]] = {
    "one_peak": (one_peak, 0.0),
    "two_peaks": (two_peaks, 6.0),
}


def _identity() -> DatasetIdentity:
    return DatasetIdentity(
        root_media_sha256="a" * 64,
        objective_definition_hash="b" * 64,
        search_space_version="1.0",
        materialization_version="1.0",
        scoring_version="1.0",
        model_fingerprint="synthetic",
    )


def _objective() -> NeuralObjective:
    return NeuralObjective(
        objective_id="o1", name="o1", metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=ObjectiveDirection.MAXIMIZE,
    )


def _fitness(genome: CandidateGenome, value: float) -> CandidateFitness:
    return CandidateFitness(
        candidate_id=genome.genome_id, genome=genome,
        status=FitnessStatus.EVALUATED, scalar_fitness=value,
        primary_objective_id="o1", raw_objectives={"o1": value},
        compute=ComputeCost(wall_seconds=0.0, tribe_runs=1),
    )


def run_phase9(name: SearchStrategyName, surface: Surface, seed: int, budget: int) -> list[float]:
    """Best-so-far after each evaluation, for a Phase 9 strategy."""
    space = _space()
    strategy = strategy_registry().create(name)
    strategy.initialize(space, seed)
    curve: list[float] = []
    best = float("-inf")
    generation = 0
    while len(curve) < budget:
        stop, _ = strategy.should_stop()
        if stop:
            break
        proposed = strategy.propose(min(4, budget - len(curve)), generation)
        if not proposed:
            break
        results = []
        for genome in proposed:
            value = surface(genome.values)
            best = max(best, value)
            curve.append(best)
            results.append(_fitness(genome, value))
        strategy.observe(results)
        generation += 1
    while len(curve) < budget:
        curve.append(best)
    return curve


def run_surrogate(
    surface: Surface, seed: int, budget: int, *, bootstrap: int, acquisition: AcquisitionName
) -> tuple[list[float], dict[str, object]]:
    """Best-so-far after each evaluation, for surrogate-assisted search."""
    def evaluate(genome: CandidateGenome) -> CandidateFitness:
        return _fitness(genome, surface(genome.values))

    orchestrator = BayesianOptimizationOrchestrator(
        _space(), _objective(), _identity(), evaluate,
        config=BayesianOptimizationConfig(
            bootstrap_evaluations=bootstrap,
            acquisition=acquisition,
            pool=PoolConfig(size=1500),
            random_seed=seed,
        ),
        trust=TrustPolicy(min_samples=bootstrap, min_spearman=0.2),
        budget=SearchBudget(max_candidates=budget, max_generations=budget),
    )
    orchestrator.run(budget)
    curve: list[float] = []
    best = float("-inf")
    for item in orchestrator.results:
        best = max(best, item.scalar_fitness or float("-inf"))
        curve.append(best)
    while len(curve) < budget:
        curve.append(best)
    return curve, orchestrator.diagnostics()


def evaluations_to(curve: list[float], threshold: float) -> int | None:
    for index, value in enumerate(curve, start=1):
        if value >= threshold:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=6)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/benchmarks/surrogate.json")
    )
    args = parser.parse_args()

    report: dict[str, object] = {}
    for surface_name, (surface, optimum) in SURFACES.items():
        methods: dict[str, list[list[float]]] = {}
        for seed in range(args.seeds):
            methods.setdefault("random", []).append(
                run_phase9(SearchStrategyName.RANDOM_SEARCH, surface, seed, args.budget)
            )
            methods.setdefault("beam", []).append(
                run_phase9(SearchStrategyName.BEAM_SEARCH, surface, seed, args.budget)
            )
            curve, _ = run_surrogate(
                surface, seed, args.budget,
                bootstrap=args.bootstrap,
                acquisition=AcquisitionName.EXPECTED_IMPROVEMENT,
            )
            methods.setdefault("bayesian", []).append(curve)

        # The threshold is what random search reaches with the full budget.
        # Asking how quickly each method gets there is the sample-efficiency
        # question; an arbitrary threshold would flatter whichever method
        # happens to suit it.
        threshold = statistics.median(curve[-1] for curve in methods["random"])
        rows: dict[str, object] = {"threshold": threshold, "optimum": optimum}
        for method, curves in methods.items():
            reached = [evaluations_to(curve, threshold) for curve in curves]
            hit = [value for value in reached if value is not None]
            # Censoring matters here. Taking the median only over runs that
            # reached the threshold rewards a method that succeeds rarely but
            # occasionally early, because its failures simply vanish from the
            # statistic. Runs that never reached it are counted at budget + 1,
            # which is the standard convention and the fair comparison.
            censored = [
                value if value is not None else args.budget + 1 for value in reached
            ]
            rows[method] = {
                "median_best": statistics.median(curve[-1] for curve in curves),
                "best_at_5": statistics.median(curve[4] for curve in curves),
                "best_at_10": statistics.median(curve[9] for curve in curves),
                "best_at_20": statistics.median(
                    curve[min(19, len(curve) - 1)] for curve in curves
                ),
                "median_evaluations_to_threshold_censored": statistics.median(censored),
                "median_evaluations_among_reachers": (
                    statistics.median(hit) if hit else None
                ),
                "runs_reaching_threshold": f"{len(hit)}/{len(curves)}",
            }
        report[surface_name] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for surface_name, rows in report.items():
        block = rows if isinstance(rows, dict) else {}
        print(f"\n{surface_name}  (optimum {block['optimum']}, "
              f"threshold {block['threshold']:.3f}, budget {args.budget}, "
              f"{args.seeds} seeds)")
        print(f"  {'method':12} {'best@5':>9} {'best@10':>9} {'best@20':>9} "
              f"{'evals→thr':>10} {'reached':>9}")
        for method in ("random", "beam", "bayesian"):
            stats = block[method]
            assert isinstance(stats, dict)
            censored = stats["median_evaluations_to_threshold_censored"]
            print(
                f"  {method:12} {stats['best_at_5']:>9.3f} {stats['best_at_10']:>9.3f} "
                f"{stats['best_at_20']:>9.3f} "
                f"{censored:>10.0f} "
                f"{stats['runs_reaching_threshold']:>9}"
            )
    print(
        "\nThreshold: what random search reaches on the full budget. "
        "'evals to threshold' counts runs that never reached it at budget + 1, "
        "so a method that succeeds rarely but sometimes early is not flattered."
    )
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
