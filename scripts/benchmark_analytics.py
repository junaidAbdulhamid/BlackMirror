"""Benchmark the vectorized Phase 3 pipeline at TRIBE's fsaverage5 width."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from blackmirror.analytics.metrics import aggregate_regions, change_magnitude, global_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-points", type=int, default=300)
    parser.add_argument("--vertices", type=int, default=20484)
    parser.add_argument("--regions", type=int, default=148)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    responses = rng.normal(size=(args.time_points, args.vertices)).astype(np.float32)
    mapping = (np.arange(args.vertices) % args.regions).astype(np.int32)

    tracemalloc.start()
    started = time.perf_counter()
    roi = aggregate_regions(responses, mapping, args.regions)
    roi_seconds = time.perf_counter() - started
    started = time.perf_counter()
    global_metrics(responses)
    change_magnitude(responses)
    metrics_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {
        "input_shape": [args.time_points, args.vertices],
        "roi_shape": list(roi.shape),
        "aggregation_seconds": roi_seconds,
        "global_and_change_seconds": metrics_seconds,
        "total_seconds": roi_seconds + metrics_seconds,
        "python_peak_memory_bytes": peak_bytes,
        "seed": 42,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
