"""Step 87: compare search strategies on synthetic objectives with known optima.

WHY SYNTHETIC
    On the real pipeline the optimum is unknown, so "this strategy found a good
    point" cannot be falsified. Here the answer is known in advance, evaluation
    is free, and every strategy can be run on identical budgets from identical
    seeds. What this measures is sample efficiency: how close each gets, and
    after how many evaluations.

WHAT IT DOES NOT SHOW
    Nothing here says which strategy is best in general. These are two small,
    smooth, low-dimensional surfaces and a real objective is none of those
    things. What the numbers support is narrower and still useful: whether a
    strategy beats random sampling on a surface where the answer is known.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.search.fitness import CandidateFitness, FitnessStatus
from blackmirror.search.space import ContentSearchSpace, continuous
from blackmirror.search.strategies import (
    BeamSearch,
    EpsilonGreedy,
    EvolutionarySearch,
    GridSearch,
    HillClimbing,
    LocalSearch,
)
from blackmirror.search.strategy import RandomSearch

Surface = Callable[[dict[str, float]], float]


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),
            continuous("brightness", Op.BRIGHTNESS, -1.0, 1.0, resolution=0.01),
        )
    )


def parabola(values: dict[str, float]) -> float:
    """One smooth peak at (3, 0.4). Optimum 0.0."""
    return -((values["gain"] - 3.0) ** 2) - 10.0 * (values["brightness"] - 0.4) ** 2


def two_peaks(values: dict[str, float]) -> float:
    """A broad low peak and a tall narrow one, to expose local optima."""
    x, y = values["gain"], values["brightness"]
    low = 2.0 - 0.05 * ((x + 6.0) ** 2 + 10.0 * y**2)
    high = 6.0 - 0.6 * ((x - 6.0) ** 2 + 10.0 * (y - 0.5) ** 2)
    return max(low, high)


SURFACES: dict[str, tuple[Surface, float]] = {
    "one_peak": (parabola, 0.0),
    "two_peaks": (two_peaks, 6.0),
}

STRATEGIES: dict[str, Callable[[], object]] = {
    "random": RandomSearch,
    "grid": lambda: GridSearch(steps=6, max_points=64),
    "local": LocalSearch,
    "hill_climbing": HillClimbing,
    "beam": lambda: BeamSearch(beam_width=3),
    "epsilon_greedy": lambda: EpsilonGreedy(exploration_rate=0.3, decay=0.85),
    "evolutionary": lambda: EvolutionarySearch(population_size=6),
}


def run(strategy_name: str, surface: Surface, seed: int, budget: int) -> dict[str, float]:
    """One run: returns best found and how many evaluations it took."""
    space = _space()
    strategy = STRATEGIES[strategy_name]()
    strategy.initialize(space, seed)  # type: ignore[attr-defined]

    best = float("-inf")
    evaluations = 0
    evaluations_to_best = 0
    generation = 0
    while evaluations < budget:
        stop, _ = strategy.should_stop()  # type: ignore[attr-defined]
        if stop:
            break
        proposed = strategy.propose(min(4, budget - evaluations), generation)  # type: ignore[attr-defined]
        if not proposed:
            break
        results = []
        for genome in proposed:
            value = surface(genome.values)
            evaluations += 1
            if value > best:
                best = value
                evaluations_to_best = evaluations
            results.append(
                CandidateFitness(
                    candidate_id=genome.genome_id,
                    genome=genome,
                    status=FitnessStatus.EVALUATED,
                    scalar_fitness=value,
                    primary_objective_id="o1",
                )
            )
        strategy.observe(results)  # type: ignore[attr-defined]
        generation += 1
    return {
        "best": best,
        "evaluations": float(evaluations),
        "evaluations_to_best": float(evaluations_to_best),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report: dict[str, dict[str, dict[str, float]]] = {}
    for surface_name, (surface, optimum) in SURFACES.items():
        report[surface_name] = {}
        for strategy_name in STRATEGIES:
            runs = [
                run(strategy_name, surface, seed, args.budget)
                for seed in range(args.seeds)
            ]
            bests = [item["best"] for item in runs]
            report[surface_name][strategy_name] = {
                "median_best": statistics.median(bests),
                "worst_best": min(bests),
                "median_gap_to_optimum": optimum - statistics.median(bests),
                "median_evaluations": statistics.median(
                    item["evaluations"] for item in runs
                ),
                "median_evaluations_to_best": statistics.median(
                    item["evaluations_to_best"] for item in runs
                ),
            }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    for surface_name, rows in report.items():
        optimum = SURFACES[surface_name][1]
        print(f"\n{surface_name}  (optimum {optimum:g}, budget {args.budget}, "
              f"{args.seeds} seeds)")
        print(f"  {'strategy':16} {'median best':>12} {'gap':>9} {'evals':>7} {'to best':>8}")
        for name, stats in sorted(rows.items(), key=lambda kv: -kv[1]["median_best"]):
            print(
                f"  {name:16} {stats['median_best']:>12.3f} "
                f"{stats['median_gap_to_optimum']:>9.3f} "
                f"{stats['median_evaluations']:>7.0f} "
                f"{stats['median_evaluations_to_best']:>8.0f}"
            )
    print(
        "\nThese are small smooth synthetic surfaces. They show whether a strategy "
        "beats random sampling where the answer is known, and nothing about which "
        "is best on a real objective."
    )


if __name__ == "__main__":
    main()
