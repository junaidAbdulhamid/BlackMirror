#!/usr/bin/env python
"""Run one end-to-end inference and print a full summary.

    python scripts/run_sample_inference.py examples/media/sample.mp4
    python scripts/run_sample_inference.py --backend mock examples/media/sample.mp4

Verifies the environment first, then runs the same InferenceService the CLI and
a future API use.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackmirror.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "predict", *sys.argv[1:]]
    app()
