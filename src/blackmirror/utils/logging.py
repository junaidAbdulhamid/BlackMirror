"""Structured console logging.

Rules enforced by convention throughout the codebase:
  * never log tensors or arrays, only their shape/dtype/summary statistics
  * never log secrets (HF tokens are redacted by :func:`redact`)
  * one line per pipeline stage, not per batch
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False
_LOGGER_NAME = "blackmirror"


def configure_logging(level: str = "INFO") -> None:
    """Install a single stream handler on the ``blackmirror`` logger.

    Idempotent: repeated calls only update the level, so importing the CLI and
    the service in the same process does not duplicate log lines.
    """
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        # Our handler is the only one; do not also propagate to root.
        logger.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``blackmirror`` namespace."""
    suffix = name.removeprefix("blackmirror.")
    return logging.getLogger(f"{_LOGGER_NAME}.{suffix}" if suffix else _LOGGER_NAME)


def redact(value: str | None, keep: int = 4) -> str:
    """Redact a secret for logging, keeping a short prefix for identification."""
    if not value:
        return "<unset>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * 8}"


def describe_array(array: Any) -> str:
    """One-line description of an array. Used instead of logging the array itself."""
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    return f"shape={tuple(shape) if shape is not None else '?'} dtype={dtype}"
