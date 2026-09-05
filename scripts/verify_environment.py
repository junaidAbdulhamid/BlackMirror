#!/usr/bin/env python
"""Diagnose whether this environment can run BlackMirror inference.

    python scripts/verify_environment.py [--backend mock] [--json]

Thin wrapper over `blackmirror verify-env` so the check is runnable before the
console script is on PATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackmirror.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "verify-env", *sys.argv[1:]]
    app()
