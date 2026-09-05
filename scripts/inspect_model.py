#!/usr/bin/env python
"""Load the configured model and print its provenance.

    python scripts/inspect_model.py [--backend mock] [--device cpu]

Prints identity, license, device, dtype, parameter count and the frozen feature
extractors — not the full architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackmirror.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "inspect-model", *sys.argv[1:]]
    app()
