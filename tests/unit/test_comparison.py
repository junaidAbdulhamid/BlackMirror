from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blackmirror.analytics.schemas import CorticalRegion, Hemisphere
from blackmirror.comparison.alignment import (
    align_linear_candidate_to_reference,
    align_observed_times,
    interpolate_rows,
)
from blackmirror.comparison.content import ContentVariant, compare_content_features
from blackmirror.comparison.engine import NeuralComparisonEngine, VariantData
from blackmirror.comparison.metrics import (
    UnsupportedComparisonMetricError,
    require_safe_metrics,
)
from blackmirror.comparison.network import load_yeo7_mapping
from blackmirror.comparison.pipeline import _with_reference_content_failure
from blackmirror.comparison.storage import ComparisonStore
from blackmirror.comparison.validation import validate_comparability


def _prediction(run_id: str, stimulus_hash: str, *, fingerprint: str = "same") -> object:
    return SimpleNamespace(
        run_id=run_id,
        stimulus=SimpleNamespace(sha256=stimulus_hash),
        model=SimpleNamespace(is_synthetic=False),
        provenance=SimpleNamespace(
            model_fingerprint=fingerprint,
            preprocessing_config={"segments": "kept"},
            applied_compat_patches=(),
        ),
        cortical=SimpleNamespace(
            surface_space="fsaverage5",
            vertex_count=4,
            hemisphere_order=("left", "right"),
            hemisphere_index_ranges={"left": (0, 2), "right": (2, 4)},
        ),
        prediction=SimpleNamespace(semantics="predicted BOLD", units=None),
    )


def _analytics(run_id: str) -> object:
    regions = (
        CorticalRegion(
            region_id=0,
            atlas_label=1,
            name="A",
            hemisphere=Hemisphere.LEFT,
            atlas="test",
            atlas_version="1",
            vertex_count=2,
        ),
        CorticalRegion(
            region_id=1,
            atlas_label=1,
            name="B",
            hemisphere=Hemisphere.RIGHT,
            atlas="test",
            atlas_version="1",
            vertex_count=2,
        ),
    )
    return SimpleNamespace(
        run_id=run_id,
        schema_version="1.0",
        metadata=SimpleNamespace(analytics_version="1.0", aggregation="mean"),
        atlas=SimpleNamespace(
            name="test", version="1", surface_space="fsaverage5", regions=regions
        ),
    )


def _variant(
    run_id: str,
    stimulus_hash: str,
    responses: np.ndarray,
    times: np.ndarray,
    *,
    fingerprint: str = "same",
) -> VariantData:
    roi = np.column_stack(
        (np.nanmean(responses[:, :2], axis=1), np.nanmean(responses[:, 2:], axis=1))
    )
    return VariantData(
        prediction=_prediction(run_id, stimulus_hash, fingerprint=fingerprint),  # type: ignore[arg-type]
        analytics=_analytics(run_id),  # type: ignore[arg-type]
        responses=responses,
        times=times,
        roi_timeseries=roi,
        vertex_to_region=np.array([0, 0, 1, 1], dtype=np.int32),
        medial_wall_mask=np.zeros(4, dtype=bool),
        network_timeseries=np.column_stack(
            (np.nanmean(responses[:, :2], axis=1), np.nanmean(responses[:, 2:], axis=1))
        ),
        network_names=("network_1", "network_2"),
        network_mapping_sha256="same-network-map",
    )


def test_exact_alignment_handles_gaps_and_unequal_durations() -> None:
    aligned = align_observed_times(
        np.array([0.0, 1.0, 7.0, 9.0]), np.array([1.0, 4.0, 7.0]), tolerance_seconds=0
    )
    np.testing.assert_array_equal(aligned.times, [1.0, 7.0])
    np.testing.assert_array_equal(aligned.reference_indices, [1, 2])
    np.testing.assert_array_equal(aligned.candidate_indices, [0, 2])


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ([0.0], [2.0], "no observed samples"),
        ([0.0, 0.0], [0.0], "strictly increasing"),
        ([0.0], [-0.1, 0.1], "ambiguous"),
        ([-0.1, 0.1], [0.0], "ambiguous"),
        ([0.0, np.nan], [0.0], "nonfinite"),
    ],
)
def test_invalid_timeline_alignment_fails(
    first: list[float], second: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        align_observed_times(np.array(first), np.array(second), tolerance_seconds=0.2)


def test_alignment_rejects_nonfinite_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        align_observed_times(np.array([0.0]), np.array([0.0]), tolerance_seconds=np.inf)


def test_comparability_rejects_model_mismatch() -> None:
    first = _prediction("a", "a" * 64)
    second = _prediction("b", "b" * 64, fingerprint="different")
    report = validate_comparability(first, second, _analytics("a"), _analytics("b"))  # type: ignore[arg-type]
    assert not report.comparable
    assert report.checks["model_fingerprint"] is False


def test_engine_rejects_different_atlas_vertex_order() -> None:
    values = np.ones((1, 4))
    reference = _variant("a", "a", values, np.array([0.0]))
    candidate = _variant("b", "b", values, np.array([0.0]))
    candidate.vertex_to_region[:] = [1, 1, 0, 0]
    with pytest.raises(ValueError, match="vertex-to-region ordering"):
        NeuralComparisonEngine().compare("bad-map", reference, (candidate,))


def test_signed_delta_and_order_reversal_are_exact() -> None:
    baseline = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=float)
    candidate = baseline + np.array([[1, -1, 2, -2], [2, -2, 4, -4]])
    times = np.array([0.0, 7.0])
    engine = NeuralComparisonEngine(max_events=2)
    ab, ab_arrays = engine.compare(
        "ab", _variant("a", "a", baseline, times), (_variant("b", "b", candidate, times),)
    )
    ba, ba_arrays = engine.compare(
        "ba", _variant("b", "b", candidate, times), (_variant("a", "a", baseline, times),)
    )
    delta_ab = ab_arrays["b"]["cortical_signed_delta"]
    delta_ba = ba_arrays["a"]["cortical_signed_delta"]
    np.testing.assert_array_equal(delta_ab, -delta_ba)
    np.testing.assert_array_equal(
        ab_arrays["b"]["cortical_absolute_delta"], ba_arrays["a"]["cortical_absolute_delta"]
    )
    assert ab.pairs[0].regions[0].mean_signed_delta == -ba.pairs[0].regions[0].mean_signed_delta
    assert ab.pairs[0].alignment.interpolation_applied is False
    assert ab.pairs[0].windows[0].observed_samples >= 1
    assert ab.pairs[0].networks[0].mean_signed_delta == 0.0


def test_nonfinite_values_are_pairwise_omitted_but_preserved_in_delta() -> None:
    first = np.array([[1, np.nan, 2, 3]], dtype=float)
    second = np.array([[3, 100, 4, 5]], dtype=float)
    result, arrays = NeuralComparisonEngine().compare(
        "finite",
        _variant("a", "a", first, np.array([0.0])),
        (_variant("b", "b", second, np.array([0.0])),),
    )
    assert np.isnan(arrays["b"]["cortical_signed_delta"][0, 1])
    assert arrays["b"]["global_mean_signed_delta"][0] == 2.0
    assert result.pairs[0].alignment.aligned_samples == 1


def test_medial_wall_values_do_not_affect_cortical_metrics() -> None:
    first = np.array([[1.0, 2.0, 3.0, 4.0]])
    second = np.array([[2.0, 1_000_000.0, 5.0, -1_000_000.0]])
    reference = _variant("a", "a", first, np.array([0.0]))
    candidate = _variant("b", "b", second, np.array([0.0]))
    reference.medial_wall_mask[:] = [False, True, False, True]
    candidate.medial_wall_mask[:] = [False, True, False, True]
    reference.vertex_to_region[:] = [0, -1, 1, -1]
    candidate.vertex_to_region[:] = [0, -1, 1, -1]
    _, arrays = NeuralComparisonEngine().compare("wall", reference, (candidate,))
    assert arrays["b"]["cortical_l2_difference"][0] == pytest.approx(np.sqrt(5))
    assert arrays["b"]["global_mean_signed_delta"][0] == 1.5
    assert np.isnan(arrays["b"]["cortical_signed_delta"][0, 1::2]).all()


def test_identical_variants_have_no_divergence_events_or_windows() -> None:
    values = np.ones((3, 4))
    times = np.arange(3, dtype=float)
    result, _ = NeuralComparisonEngine().compare(
        "same", _variant("a", "a", values, times), (_variant("b", "b", values, times),)
    )
    assert result.pairs[0].events == ()
    assert result.pairs[0].windows == ()


def test_engine_rejects_response_vertex_count_mismatch() -> None:
    reference = _variant("a", "a", np.ones((1, 4)), np.array([0.0]))
    candidate = _variant("b", "b", np.ones((1, 4)), np.array([0.0]))
    candidate = replace(candidate, responses=np.ones((1, 3)))
    with pytest.raises(ValueError, match="cortical response shape"):
        NeuralComparisonEngine().compare("shape", reference, (candidate,))


def test_atomic_comparison_persistence(tmp_path: Path) -> None:
    first = np.ones((1, 4))
    second = first * 2
    result, arrays = NeuralComparisonEngine().compare(
        "cmp-1",
        _variant("a", "a", first, np.array([0.0])),
        (_variant("b", "b", second, np.array([0.0])),),
    )
    store = ComparisonStore(tmp_path)
    path = store.write(result, arrays)
    assert store.read("cmp-1") == result
    with np.load(path / "pairs" / "b.npz") as saved:
        np.testing.assert_array_equal(saved["cortical_signed_delta"], np.ones((1, 4)))
    with pytest.raises(FileExistsError):
        store.write(result, arrays)


def test_comparison_store_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid comparison id"):
        ComparisonStore(tmp_path).read("../escape")
    with pytest.raises(ValueError, match="invalid comparison id"):
        ComparisonStore(tmp_path).read("..")


def test_packaged_yeo7_mapping_is_exact_and_excludes_wall() -> None:
    root = Path(__file__).parents[2] / "artifacts"
    mapping = load_yeo7_mapping(
        root / "atlases" / "yeo2011_fsaverage5", root / "mesh" / "fsaverage5"
    )
    assert mapping.vertex_to_network.shape == (20_484,)
    assert int(mapping.medial_wall_mask.sum()) == 1_769
    assert set(np.unique(mapping.vertex_to_network)) == {-1, 0, 1, 2, 3, 4, 5, 6}
    values = np.ones((1, 20_484), dtype=np.float64)
    values[:, mapping.medial_wall_mask] = 1e9
    np.testing.assert_array_equal(mapping.aggregate(values), np.ones((1, 7)))


def test_linear_alignment_is_opt_in_and_never_extrapolates() -> None:
    aligned = align_linear_candidate_to_reference(
        np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 2.0]),
        max_gap_seconds=2.0,
    )
    np.testing.assert_array_equal(aligned.times, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(aligned.candidate_observed_mask, [True, False, True])
    result = interpolate_rows(np.array([[0.0], [4.0]]), aligned)
    np.testing.assert_array_equal(result[:, 0], [0.0, 2.0, 4.0])


def test_engine_persists_interpolation_brackets_and_weights() -> None:
    reference = _variant("a", "a", np.zeros((2, 4)), np.array([0.5, 1.5]))
    candidate = _variant("b", "b", np.ones((2, 4)), np.array([0.0, 2.0]))
    result, arrays = NeuralComparisonEngine(
        alignment_method="linear_candidate_to_reference", max_interpolation_gap_seconds=2.0
    ).compare("linear", reference, (candidate,))
    assert result.pairs[0].alignment.interpolated_candidate_samples == 2
    assert result.pairs[0].alignment.candidate_coverage_fraction == 1.0
    np.testing.assert_array_equal(arrays["b"]["candidate_left_row_indices"], [0, 0])
    np.testing.assert_array_equal(arrays["b"]["candidate_right_weights"], [0.25, 0.75])


def test_content_comparison_masks_unavailable_modalities() -> None:
    names = ("motion", "visual_available", "speech_present", "content_event_available")
    left = ContentVariant(
        "1",
        "4",
        {"visual": "m"},
        {"rate": 4},
        names,
        np.array([0.0]),
        np.array([[9, 0, 1, 0]], dtype=float),
    )
    right = ContentVariant(
        "1",
        "4",
        {"visual": "m"},
        {"rate": 4},
        names,
        np.array([0.0]),
        np.array([[1, 0, 0, 0]], dtype=float),
    )
    difference = compare_content_features(left, right)
    assert np.isnan(difference.signed_delta[0, 0])
    assert np.isnan(difference.signed_delta[0, 2])
    with pytest.raises(ValueError, match="analysis_version"):
        compare_content_features(left, replace(right, analysis_version="different"))


def test_unsupported_psychological_metric_is_explicitly_rejected() -> None:
    assert require_safe_metrics(("signed_delta",))[0].value == "signed_delta"
    with pytest.raises(UnsupportedComparisonMetricError, match="psychological"):
        require_safe_metrics(("engagement",))


def test_unreadable_reference_content_downgrades_all_pairs_and_neural_result_persists(
    tmp_path: Path,
) -> None:
    values = np.ones((1, 4))
    result, arrays = NeuralComparisonEngine().compare(
        "content-corrupt",
        _variant("reference", "a", values, np.array([0.0])),
        (
            _variant("candidate-1", "b", values * 2, np.array([0.0])),
            _variant("candidate-2", "c", values * 3, np.array([0.0])),
        ),
    )
    downgraded = _with_reference_content_failure(result, ValueError("corrupt metadata"))
    for pair in downgraded.pairs:
        assert pair.content_status == "unavailable_reference_artifact"
        assert pair.content_issue is not None
        assert pair.content_issue.code == "reference_content_unreadable"
        assert pair.content_issue.reference_run_id == "reference"
        assert pair.content_issue.candidate_run_id == pair.candidate_run_id
    store = ComparisonStore(tmp_path)
    store.write(downgraded, arrays)
    assert store.read("content-corrupt") == downgraded
