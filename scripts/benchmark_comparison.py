"""Benchmark Phase 5 vectorized A/B difference calculations at fsaverage5 width."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from blackmirror.comparison.alignment import align_observed_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-points", type=int, default=300)
    parser.add_argument("--vertices", type=int, default=20484)
    parser.add_argument("--regions", type=int, default=148)
    parser.add_argument("--alignment-points", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(45)
    reference = rng.normal(size=(args.time_points, args.vertices)).astype(np.float32)
    candidate = rng.normal(size=(args.time_points, args.vertices)).astype(np.float32)
    reference_roi = rng.normal(size=(args.time_points, args.regions)).astype(np.float64)
    candidate_roi = rng.normal(size=(args.time_points, args.regions)).astype(np.float64)

    tracemalloc.start()
    started = time.perf_counter()
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    l2 = np.linalg.norm(delta, axis=1)
    roi_delta = candidate_roi - reference_roi
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timeline = np.arange(args.alignment_points, dtype=np.float64)
    tracemalloc.start()
    alignment_started = time.perf_counter()
    aligned = align_observed_times(timeline, timeline, tolerance_seconds=0)
    alignment_elapsed = time.perf_counter() - alignment_started
    _, alignment_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {
        "input_shape_each": [args.time_points, args.vertices],
        "roi_shape_each": [args.time_points, args.regions],
        "difference_seconds": elapsed,
        "python_peak_memory_bytes": peak,
        "alignment": {
            "input_points_each": args.alignment_points,
            "aligned_points": len(aligned.times),
            "seconds": alignment_elapsed,
            "python_peak_memory_bytes": alignment_peak,
        },
        "outputs": {
            "cortical_delta_shape": list(delta.shape),
            "roi_delta_shape": list(roi_delta.shape),
            "l2_shape": list(l2.shape),
        },
        "seed": 45,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
