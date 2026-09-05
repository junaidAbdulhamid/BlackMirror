"""Hashing underpins reproducibility and the run cache key."""

from __future__ import annotations

from pathlib import Path

from blackmirror.utils.hashing import (
    build_cache_key,
    sha256_file,
    sha256_text,
    short_hash,
    stable_hash,
)


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello world")
    # Independently verifiable: echo -n "hello world" | shasum -a 256
    assert sha256_file(path) == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_sha256_file_streams_large_files(tmp_path: Path) -> None:
    """Files larger than the 1 MiB read chunk must hash identically."""
    payload = b"x" * (1024 * 1024 * 2 + 17)
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_text(payload.decode())


def test_stable_hash_ignores_key_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_stable_hash_distinguishes_values() -> None:
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_cache_key_changes_with_each_component() -> None:
    base = {
        "stimulus_sha256": "a" * 64,
        "model_fingerprint": "tribe_v2|facebook/tribev2|best.ckpt|0.1.0|float32",
        "preprocessing_fingerprint": {"remove_empty_segments": True},
    }
    reference = build_cache_key(**base)

    assert build_cache_key(**{**base, "stimulus_sha256": "b" * 64}) != reference
    assert build_cache_key(**{**base, "model_fingerprint": "other"}) != reference
    assert (
        build_cache_key(**{**base, "preprocessing_fingerprint": {"remove_empty_segments": False}})
        != reference
    )
    # Identical inputs must be stable across calls, or caching is unsound.
    assert build_cache_key(**base) == reference


def test_short_hash_truncates() -> None:
    assert short_hash("a" * 64, 12) == "a" * 12
