"""Artifact persistence and the run cache index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blackmirror.errors import ArtifactReadError, ArtifactWriteError
from blackmirror.storage.artifact_store import ArtifactStore, generate_run_id


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def test_run_ids_are_unique_and_sort_chronologically() -> None:
    ids = [generate_run_id() for _ in range(50)]
    assert len(set(ids)) == 50

    # A fixed-width UTC timestamp prefix (YYYYmmddTHHMMSSZ) is what makes lexical
    # sorting equal chronological sorting, which `list_runs` relies on.
    stamps = [run_id.split("-")[0] for run_id in ids]
    assert all(len(stamp) == 16 for stamp in stamps)
    assert stamps == sorted(stamps)

    import datetime as dt

    parsed = dt.datetime.strptime(stamps[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC)
    assert abs((dt.datetime.now(dt.UTC) - parsed).total_seconds()) < 120


def test_predictions_round_trip_byte_for_byte(store: ArtifactStore) -> None:
    run_id = generate_run_id()
    store.create_run_dir(run_id)
    array = np.random.default_rng(0).normal(size=(7, 32)).astype(np.float32)

    store.write_predictions(run_id, array)
    loaded = store.load_predictions(run_id)

    np.testing.assert_array_equal(loaded, array)
    assert loaded.dtype == array.dtype


def test_nan_and_inf_survive_persistence(store: ArtifactStore) -> None:
    """Raw scientific data must round-trip unmodified, anomalies included."""
    run_id = generate_run_id()
    store.create_run_dir(run_id)
    array = np.array([[np.nan, np.inf], [-np.inf, 1.5]], dtype=np.float32)

    store.write_predictions(run_id, array)
    loaded = store.load_predictions(run_id)

    assert np.isnan(loaded[0, 0])
    assert np.isposinf(loaded[0, 1])
    assert np.isneginf(loaded[1, 0])


def test_arrays_round_trip(store: ArtifactStore) -> None:
    run_id = generate_run_id()
    store.create_run_dir(run_id)
    arrays = {
        "segment_start_seconds": np.array([0.0, 1.0, 7.0]),
        "segment_n_events": np.array([1, 2, 3]),
    }
    path = store.write_arrays(run_id, "temporal.npz", arrays)

    with np.load(path) as loaded:
        np.testing.assert_array_equal(
            loaded["segment_start_seconds"], arrays["segment_start_seconds"]
        )
        np.testing.assert_array_equal(loaded["segment_n_events"], arrays["segment_n_events"])


def test_duplicate_run_dir_is_refused(store: ArtifactStore) -> None:
    run_id = generate_run_id()
    store.create_run_dir(run_id)
    with pytest.raises(ArtifactWriteError, match="already exists"):
        store.create_run_dir(run_id)


def test_reading_a_missing_manifest_raises(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactReadError, match="No manifest"):
        store.read_manifest("does-not-exist")


def test_cache_key_lookup_returns_the_latest_match(store: ArtifactStore) -> None:
    for run_id, key in (("run-a", "key1"), ("run-b", "key2"), ("run-c", "key1")):
        store.create_run_dir(run_id)
        (store.run_dir(run_id) / "manifest.json").write_text("{}", encoding="utf-8")
        store.append_index({"run_id": run_id, "cache_key": key, "status": "completed"})

    assert store.find_by_cache_key("key1") == "run-c"
    assert store.find_by_cache_key("key2") == "run-b"
    assert store.find_by_cache_key("absent") is None


def test_cache_lookup_ignores_failed_runs(store: ArtifactStore) -> None:
    store.create_run_dir("run-x")
    (store.run_dir("run-x") / "manifest.json").write_text("{}", encoding="utf-8")
    store.append_index({"run_id": "run-x", "cache_key": "k", "status": "failed"})
    assert store.find_by_cache_key("k") is None


def test_cache_lookup_ignores_runs_whose_files_are_gone(store: ArtifactStore) -> None:
    """A pruned artifacts directory must not produce a phantom cache hit."""
    store.append_index({"run_id": "vanished", "cache_key": "k", "status": "completed"})
    assert store.find_by_cache_key("k") is None


def test_corrupt_index_lines_are_skipped(store: ArtifactStore) -> None:
    store.create_run_dir("run-ok")
    (store.run_dir("run-ok") / "manifest.json").write_text("{}", encoding="utf-8")
    store.index_path.parent.mkdir(parents=True, exist_ok=True)
    store.index_path.write_text("not json\n\n", encoding="utf-8")
    store.append_index({"run_id": "run-ok", "cache_key": "k", "status": "completed"})
    assert store.find_by_cache_key("k") == "run-ok"


def test_list_runs_requires_a_manifest(store: ArtifactStore) -> None:
    store.create_run_dir("with-manifest")
    (store.run_dir("with-manifest") / "manifest.json").write_text("{}", encoding="utf-8")
    store.create_run_dir("without-manifest")
    assert store.list_runs() == ["with-manifest"]
