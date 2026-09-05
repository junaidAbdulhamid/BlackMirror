"""Reproducible Phase 6 benchmark using the production scoring engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
import tracemalloc
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from blackmirror.scoring.engine import GoalConditionedScoringEngine
from blackmirror.scoring.evaluator import ScoringVariant
from blackmirror.scoring.schemas import (
    NeuralObjective,
    NormalizationStrategy,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)

DEFAULT_SEED = 46


def _objectives(count: int, regions: int) -> tuple[NeuralObjective, ...]:
    definitions = (
        ("roi_mean", "MEAN_RESPONSE", ObjectiveTarget(type=TargetType.ROI, region_id=0)),
        (
            "cortex_change",
            "TEMPORAL_CHANGE_MAGNITUDE",
            ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        ),
        (
            "roi_integral",
            "INTEGRATED_RESPONSE",
            ObjectiveTarget(type=TargetType.ROI, region_id=min(1, regions - 1)),
        ),
    )
    if not 1 <= count <= len(definitions):
        raise ValueError(f"objective count must be in [1, {len(definitions)}]")
    return tuple(
        NeuralObjective(
            objective_id=objective_id,
            name=objective_id.replace("_", " "),
            metric=metric,
            target=target,
            temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
            direction=ObjectiveDirection.MAXIMIZE,
            normalization=ObjectiveNormalization(
                strategy=NormalizationStrategy.MIN_MAX_WITHIN_EXPERIMENT
            ),
            weight=1.0 / count,
        )
        for objective_id, metric, target in definitions[:count]
    )


def _fixture(
    count: int, *, time_points: int, vertices: int, regions: int, seed: int
) -> tuple[tuple[ScoringVariant, ...], str]:
    rng = np.random.default_rng(seed)
    base_responses = rng.normal(size=(time_points, vertices)).astype(np.float32)
    base_roi = rng.normal(size=(time_points, regions)).astype(np.float64)
    times = np.arange(time_points, dtype=np.float64)
    wall = np.zeros(vertices, dtype=bool)
    wall[: min(1769, max(0, vertices // 10))] = True
    digest = hashlib.sha256()
    digest.update(count.to_bytes(8, byteorder="little", signed=False))
    for array in (base_responses, base_roi, times, wall):
        digest.update(array.tobytes(order="C"))
    variants = tuple(
        ScoringVariant(
            variant_id=f"variant-{index:03d}",
            responses=base_responses + np.float32(index * 0.001),
            times=times,
            roi_timeseries=base_roi + index * 0.001,
            region_names=tuple(f"region-{region}" for region in range(regions)),
            medial_wall_mask=wall,
            duration_seconds=float(time_points),
            hemisphere_ranges={"left": (0, vertices // 2), "right": (vertices // 2, vertices)},
            atlas_name="synthetic-destrieux",
            atlas_version="benchmark-v1",
            analytics_version="benchmark-v1",
            model_fingerprint="deterministic-synthetic-fixture",
        )
        for index in range(count)
    )
    return variants, digest.hexdigest()


def generate_report(
    *, variant_counts: tuple[int, ...], objective_counts: tuple[int, ...],
    time_points: int, vertices: int, regions: int, seed: int
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    fixture_hashes: dict[int, str] = {}
    for variant_count in variant_counts:
        variants, fixture_hash = _fixture(
            variant_count, time_points=time_points, vertices=vertices, regions=regions, seed=seed
        )
        fixture_hashes[variant_count] = fixture_hash
        for objective_count in objective_counts:
            objectives = _objectives(objective_count, regions)
            tracemalloc.start()
            started = time.perf_counter_ns()
            result = GoalConditionedScoringEngine().score_experiment(
                "phase6-benchmark", variants, objectives
            )
            elapsed_ns = time.perf_counter_ns() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            encoded = result.model_dump_json().encode("utf-8")
            cases.append(
                {
                    "variants": variant_count,
                    "objectives": objective_count,
                    "runtime_seconds": elapsed_ns / 1_000_000_000,
                    "python_peak_memory_bytes": peak,
                    "result_json_bytes": len(encoded),
                    "valid_variant_scores": sum(score.valid for score in result.variant_scores),
                    "top_variant_id": result.variant_scores[0].variant_id,
                }
            )
    try:
        package_version = version("blackmirror")
    except PackageNotFoundError:
        package_version = "source-tree"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    process_peak_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    return {
        "benchmark": "phase6_goal_conditioned_scoring",
        "schema_version": "1.0",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blackmirror": package_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "fixture": {
            "kind": "numpy.default_rng synthetic predicted average-subject outputs",
            "seed": seed,
            "sha256_by_variant_count": {str(k): v for k, v in fixture_hashes.items()},
        },
        "dimensions": {
            "variant_counts": list(variant_counts),
            "objective_counts": list(objective_counts),
            "time_points": time_points,
            "vertices": vertices,
            "regions": regions,
        },
        "scoring_config": {
            "metrics_in_order": [
                "MEAN_RESPONSE",
                "TEMPORAL_CHANGE_MAGNITUDE",
                "INTEGRATED_RESPONSE",
            ],
            "target_types_in_order": ["roi", "whole_cortex", "roi"],
            "temporal_scope": "full_stimulus",
            "normalization": "min_max_within_experiment",
            "direction": "maximize",
            "weights": "equal",
        },
        "memory": {"process_peak_rss_bytes_after_cases": process_peak_bytes},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="2,10,50")
    parser.add_argument("--objective-counts", default="1,3")
    parser.add_argument("--time-points", type=int, default=30)
    parser.add_argument("--vertices", type=int, default=20_484)
    parser.add_argument("--regions", type=int, default=148)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/phase6_scoring.json")
    )
    args = parser.parse_args()
    report = generate_report(
        variant_counts=tuple(int(value) for value in args.variants.split(",")),
        objective_counts=tuple(int(value) for value in args.objective_counts.split(",")),
        time_points=args.time_points, vertices=args.vertices, regions=args.regions, seed=args.seed,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
