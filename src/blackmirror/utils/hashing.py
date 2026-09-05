"""Content hashing.

Stimulus SHA-256 is the backbone of reproducibility: it lets us detect duplicate
uploads, key an inference cache, reproduce an experiment months later, and prove
exactly which bytes produced a given prediction matrix. Without it, "Variant A
and Variant B were evaluated under identical conditions" is an unverifiable claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024  # 1 MiB; media files are large, never read them whole.


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    """Hash an arbitrary JSON-serializable payload deterministically.

    Keys are sorted so that dict ordering never changes the digest.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(encoded)


def build_cache_key(
    *,
    stimulus_sha256: str,
    model_fingerprint: str,
    preprocessing_fingerprint: dict[str, Any],
) -> str:
    """Compute the run cache key.

    Two runs share a key only if the same bytes went through the same model
    under the same preprocessing configuration. This is the seam where a real
    cache (Redis, object store) can be introduced later without touching the
    inference service.
    """
    return stable_hash(
        {
            "stimulus": stimulus_sha256,
            "model": model_fingerprint,
            "preprocessing": preprocessing_fingerprint,
        }
    )


def short_hash(value: str, length: int = 12) -> str:
    """Truncate a hex digest for human-readable logging."""
    return value[:length]
