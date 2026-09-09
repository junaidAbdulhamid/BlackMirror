"""Measure Phase 8 atomic checkpoint overhead independently of model pipelines."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from blackmirror.resimulation.storage import ResimulationStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path, help="A valid Phase 8 state.json fixture.")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    from blackmirror.resimulation.schemas import ResimulationResult

    result = ResimulationResult.model_validate_json(args.state.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        store = ResimulationStore(Path(temporary))
        started = time.perf_counter()
        for _ in range(args.iterations):
            store.checkpoint(result)
            store.read(result.request.resimulation_id)
        elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "iterations": args.iterations,
                "total_seconds": elapsed,
                "checkpoint_and_read_ms": elapsed * 1000 / args.iterations,
                "state_bytes": args.state.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
