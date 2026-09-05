"""Prediction-to-cortex alignment.

The single most important correctness property of Phase 2: prediction column `j`
must land on cortical vertex `j`. A wrong mapping paints real predicted values
onto the wrong anatomy and still looks completely plausible, so it has to be
caught by a test rather than by eye.

No single check proves this. Three complementary ones are used, and it is worth
being precise about what each does and does not establish.

1. **Byte-identity of the exported mesh** (`test_export_matches_nilearn_exactly`).
   Our surface comes from `nilearn.datasets.load_fsaverage("fsaverage5")` — the
   *same call* TRIBE's own plotting makes (`tribev2/plotting/cortical.py:182`).
   Asserting our export is byte-identical to a fresh load proves vertex ORDERING
   is preserved end to end. This is proof by construction, not statistics.
   It does not, on its own, prove the model's output is indexed the same way.

2. **Spatial smoothness** (`test_predictions_are_spatially_smooth_on_the_mesh`).
   BOLD varies smoothly across cortex, so edge-connected vertices should hold
   more similar values than random pairs. This detects shuffles, rotations and
   reversal of the index order. Measured on real data:

       correct            2.99x        rotated by 1       1.83x
       reversed           1.15x        rotated by 101     1.21x
       shuffled           1.00x        rotated by 5000    1.15x

   **It cannot detect a hemisphere swap** — swapping LH and RH scored 3.20x,
   *higher* than correct — because each hemisphere is internally smooth either
   way. That is precisely why check 3 exists.

3. **Medial-wall variance** (`test_medial_wall_detects_hemisphere_assignment`).
   The mask comes from an anatomical atlas, not the model, and the LH and RH
   wall index sets overlap by only ~3%. Masked vertices should show markedly
   lower variance than cortex; under a hemisphere swap that signal inverts:

       correct  0.51x / 0.40x        swapped  1.25x / 0.80x

Together these pin ordering, orientation and hemisphere assignment. What remains
unproven by testing alone is the model's internal convention, which is taken
from upstream source and cited in docs/tribe_integration.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

MESH_DIR = Path("artifacts/mesh/fsaverage5")
FSAVERAGE5_PER_HEMISPHERE = 10242
FSAVERAGE5_TOTAL = 20484

pytestmark = pytest.mark.skipif(
    not (MESH_DIR / "mapping.json").exists(),
    reason="run `blackmirror export-mesh` first",
)


def _real_run_predictions() -> np.ndarray | None:
    """The newest non-synthetic run, if one exists."""
    runs = Path("artifacts/runs")
    if not runs.exists():
        return None
    for run_dir in sorted(runs.iterdir(), reverse=True):
        manifest = run_dir / "manifest.json"
        predictions = run_dir / "predictions.npy"
        if not (manifest.exists() and predictions.exists()):
            continue
        if json.loads(manifest.read_text())["model"]["is_synthetic"]:
            continue
        return np.load(predictions)
    return None


def _edges(hemisphere: str) -> np.ndarray:
    faces = np.load(MESH_DIR / f"{hemisphere}_faces.npy")
    return np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])


def _wall_variance_ratio(frame: np.ndarray, mask: np.ndarray) -> float:
    """Variance of medial-wall vertices relative to cortex."""
    return float(frame[mask].std() / frame[~mask].std())


def _smoothness_ratio(values: np.ndarray, edges: np.ndarray, seed: int = 0) -> float:
    """How much more similar edge-connected vertices are than random pairs.

    ~1.0 means no spatial structure (a broken mapping); >1 means the values vary
    smoothly across the surface, as a cortical signal must.
    """
    rng = np.random.default_rng(seed)
    neighbour = np.abs(values[edges[:, 0]] - values[edges[:, 1]]).mean()
    a = rng.integers(0, values.size, edges.shape[0])
    b = rng.integers(0, values.size, edges.shape[0])
    random_pair = np.abs(values[a] - values[b]).mean()
    return float(random_pair / neighbour)


# --- Static mesh invariants ----------------------------------------------


def test_mesh_manifest_declares_the_expected_geometry() -> None:
    manifest = json.loads((MESH_DIR / "mapping.json").read_text())
    assert manifest["vertices_per_hemisphere"] == FSAVERAGE5_PER_HEMISPHERE
    assert manifest["total_vertices"] == FSAVERAGE5_TOTAL
    assert manifest["hemisphere_index_ranges"] == {
        "left": [0, FSAVERAGE5_PER_HEMISPHERE],
        "right": [FSAVERAGE5_PER_HEMISPHERE, FSAVERAGE5_TOTAL],
    }
    assert manifest["faces_are_hemisphere_local"] is True


@pytest.mark.parametrize("hemisphere", ["left", "right"])
@pytest.mark.parametrize("surface", ["inflated", "pial"])
def test_vertex_arrays_have_the_right_shape(hemisphere: str, surface: str) -> None:
    vertices = np.load(MESH_DIR / f"{hemisphere}_{surface}_vertices.npy")
    assert vertices.shape == (FSAVERAGE5_PER_HEMISPHERE, 3)
    assert np.isfinite(vertices).all()


@pytest.mark.parametrize("hemisphere", ["left", "right"])
def test_faces_index_within_their_own_hemisphere(hemisphere: str) -> None:
    """Hemisphere-local indices. Global indices would stitch the hemispheres."""
    faces = np.load(MESH_DIR / f"{hemisphere}_faces.npy")
    assert faces.shape[1] == 3
    assert faces.min() >= 0
    assert faces.max() < FSAVERAGE5_PER_HEMISPHERE


@pytest.mark.parametrize("hemisphere", ["left", "right"])
def test_every_vertex_belongs_to_a_triangle(hemisphere: str) -> None:
    """An orphaned vertex would render as an invisible hole in the surface."""
    faces = np.load(MESH_DIR / f"{hemisphere}_faces.npy")
    assert len(np.unique(faces)) == FSAVERAGE5_PER_HEMISPHERE


def test_inflated_hemispheres_overlap_and_need_a_render_offset() -> None:
    """Documents a real trap: inflated surfaces are each centred on x=0.

    Rendered as-is they superimpose. The renderer must apply a lateral offset.
    Pial coordinates are already anatomically lateralised.
    """
    left = np.load(MESH_DIR / "left_inflated_vertices.npy")
    right = np.load(MESH_DIR / "right_inflated_vertices.npy")
    assert abs(left[:, 0].mean()) < 5.0
    assert abs(right[:, 0].mean()) < 5.0

    left_pial = np.load(MESH_DIR / "left_pial_vertices.npy")
    right_pial = np.load(MESH_DIR / "right_pial_vertices.npy")
    assert left_pial[:, 0].mean() < -10.0
    assert right_pial[:, 0].mean() > 10.0


def test_medial_wall_mask_covers_the_full_vertex_axis() -> None:
    mask = np.load(MESH_DIR / "medial_wall_mask.npy")
    assert mask.shape == (FSAVERAGE5_TOTAL,)
    assert mask.dtype == bool
    assert 0.02 < mask.mean() < 0.25


# --- Alignment against real predictions ----------------------------------


def test_predictions_are_spatially_smooth_on_the_mesh() -> None:
    """The core alignment proof.

    If prediction index i did not correspond to mesh vertex i, edge-connected
    vertices would be no more similar than random ones.
    """
    predictions = _real_run_predictions()
    if predictions is None:
        pytest.skip("no non-synthetic run available")

    frame = predictions[0]
    assert frame.size == FSAVERAGE5_TOTAL

    for hemisphere, start in (("left", 0), ("right", FSAVERAGE5_PER_HEMISPHERE)):
        values = frame[start : start + FSAVERAGE5_PER_HEMISPHERE]
        ratio = _smoothness_ratio(values, _edges(hemisphere))
        assert ratio > 2.0, (
            f"{hemisphere} hemisphere smoothness ratio {ratio:.2f} is too low; the "
            f"prediction-to-vertex mapping is probably wrong."
        )


def test_shuffling_the_mapping_destroys_smoothness() -> None:
    """Control for the test above: proves the signal is the mapping, not the data."""
    predictions = _real_run_predictions()
    if predictions is None:
        pytest.skip("no non-synthetic run available")

    rng = np.random.default_rng(0)
    shuffled = predictions[0].copy()
    rng.shuffle(shuffled)

    ratio = _smoothness_ratio(shuffled[:FSAVERAGE5_PER_HEMISPHERE], _edges("left"))
    assert ratio < 1.2, f"shuffled data should show no spatial structure, got {ratio:.2f}"


def test_medial_wall_carries_less_variance_than_cortex() -> None:
    """Independent evidence for alignment.

    The mask comes from an anatomical atlas, not from the model. If the mapping
    were wrong, masked vertices would be statistically indistinguishable from
    cortex — instead they show markedly lower variance, as a region without
    meaningful fMRI signal should.
    """
    predictions = _real_run_predictions()
    if predictions is None:
        pytest.skip("no non-synthetic run available")

    mask = np.load(MESH_DIR / "medial_wall_mask.npy")
    ratio = _wall_variance_ratio(predictions[0], mask)
    assert ratio < 0.8, (
        f"medial wall variance is {ratio:.2f}x cortex; expected clearly lower. "
        f"The mapping or the mask may be misaligned."
    )


def test_medial_wall_detects_hemisphere_assignment() -> None:
    """The check that catches what smoothness cannot: swapped hemispheres.

    LH and RH medial-wall index sets barely overlap, so applying one
    hemisphere's data against the other's mask destroys the low-variance signal.
    """
    predictions = _real_run_predictions()
    if predictions is None:
        pytest.skip("no non-synthetic run available")

    mask = np.load(MESH_DIR / "medial_wall_mask.npy")
    frame = predictions[0]
    swapped = np.concatenate(
        [frame[FSAVERAGE5_PER_HEMISPHERE:], frame[:FSAVERAGE5_PER_HEMISPHERE]]
    )

    correct_ratio = _wall_variance_ratio(frame, mask)
    swapped_ratio = _wall_variance_ratio(swapped, mask)

    assert swapped_ratio > correct_ratio * 1.4, (
        f"Swapping hemispheres barely changed the medial-wall signal "
        f"({correct_ratio:.2f} -> {swapped_ratio:.2f}), so this test cannot "
        f"actually detect a swap on this run."
    )


def test_left_and_right_medial_walls_are_distinct() -> None:
    """Why the swap test works: the two masks are nearly independent."""
    mask = np.load(MESH_DIR / "medial_wall_mask.npy")
    left = mask[:FSAVERAGE5_PER_HEMISPHERE]
    right = mask[FSAVERAGE5_PER_HEMISPHERE:]

    assert not np.array_equal(left, right)
    overlap = np.logical_and(left, right).sum() / np.logical_or(left, right).sum()
    assert overlap < 0.2, f"LH/RH wall masks overlap {overlap:.2f}; swap test would be weak"


def test_export_matches_nilearn_exactly() -> None:
    """Proof by construction that vertex ORDERING survived the export.

    TRIBE's own plotting loads the surface with
    `nilearn.datasets.load_fsaverage(mesh=...)` (tribev2/plotting/cortical.py).
    We export from the identical call, so byte-identity here means our vertex
    order is the order the model's own visualisation assumes.
    """
    nilearn_datasets = pytest.importorskip("nilearn.datasets")
    fsaverage = nilearn_datasets.load_fsaverage(mesh="fsaverage5")

    for hemisphere in ("left", "right"):
        for surface in ("pial", "inflated"):
            ours = np.load(MESH_DIR / f"{hemisphere}_{surface}_vertices.npy")
            reference = np.asarray(
                fsaverage[surface].parts[hemisphere].coordinates, dtype=np.float32
            )
            assert np.array_equal(ours, reference), (
                f"{hemisphere} {surface} vertices differ from a fresh nilearn load; "
                f"the export reordered or altered the surface."
            )

        ours_faces = np.load(MESH_DIR / f"{hemisphere}_faces.npy")
        reference_faces = np.asarray(
            fsaverage["pial"].parts[hemisphere].faces, dtype=np.int32
        )
        assert np.array_equal(ours_faces, reference_faces)


@pytest.mark.parametrize("shift", [1, 7, 101, 5000])
def test_rotating_the_mapping_destroys_smoothness(shift: int) -> None:
    """Rotation is a subtler corruption than a shuffle, and must still be caught."""
    predictions = _real_run_predictions()
    if predictions is None:
        pytest.skip("no non-synthetic run available")

    values = predictions[0][:FSAVERAGE5_PER_HEMISPHERE]
    edges = _edges("left")
    correct = _smoothness_ratio(values, edges)
    rotated = _smoothness_ratio(np.roll(values, shift), edges)

    assert rotated < correct * 0.75, (
        f"rotating by {shift} kept smoothness at {rotated:.2f} vs {correct:.2f}; "
        f"the test cannot distinguish a shifted mapping."
    )


def test_reversing_the_mapping_destroys_smoothness() -> None:
    predictions = _real_run_predictions()
    if predictions is None:
        pytest.skip("no non-synthetic run available")

    values = predictions[0][:FSAVERAGE5_PER_HEMISPHERE]
    edges = _edges("left")
    assert _smoothness_ratio(values[::-1].copy(), edges) < _smoothness_ratio(
        values, edges
    ) * 0.75


def test_smoothness_alone_cannot_detect_a_hemisphere_swap() -> None:
    """Documents the known blind spot, so nobody over-trusts the smoothness check.

    If this ever starts failing, smoothness has become swap-sensitive and the
    docs describing its limits need revisiting.
    """
    predictions = _real_run_predictions()
    if predictions is None:
        pytest.skip("no non-synthetic run available")

    frame = predictions[0]
    left_ratio = _smoothness_ratio(frame[:FSAVERAGE5_PER_HEMISPHERE], _edges("left"))
    swapped_ratio = _smoothness_ratio(frame[FSAVERAGE5_PER_HEMISPHERE:], _edges("left"))

    # Both remain high: each hemisphere is internally smooth either way.
    assert left_ratio > 2.0
    assert swapped_ratio > 2.0
