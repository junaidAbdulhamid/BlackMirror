from __future__ import annotations

import importlib.util
from pathlib import Path


def test_phase6_benchmark_report_is_reproducible_and_complete() -> None:
    script = Path(__file__).parents[2] / "scripts" / "benchmark_scoring.py"
    spec = importlib.util.spec_from_file_location("benchmark_scoring", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kwargs = dict(
        variant_counts=(2,), objective_counts=(1,), time_points=4,
        vertices=20, regions=3, seed=46,
    )
    first = module.generate_report(**kwargs)
    second = module.generate_report(**kwargs)

    assert first["fixture"] == second["fixture"]
    assert first["dimensions"] == second["dimensions"]
    assert first["scoring_config"] == second["scoring_config"]
    case = first["cases"][0]
    assert case["valid_variant_scores"] == 2
    assert case["top_variant_id"] == "variant-001"
    assert case["runtime_seconds"] >= 0
    assert case["python_peak_memory_bytes"] > 0
    assert case["result_json_bytes"] > 0
    assert first["memory"]["process_peak_rss_bytes_after_cases"] > 0
