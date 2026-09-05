"""Cortical mesh export — the Phase 2 prerequisite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blackmirror.cortical.mesh import FSAVERAGE_SIZES, export_surface_mesh
from blackmirror.errors import ArtifactWriteError

nilearn = pytest.importorskip("nilearn", reason="mesh export needs the [viz] extra")


def test_known_resolutions_match_freesurfer() -> None:
    assert FSAVERAGE_SIZES["fsaverage5"] == 10242
    assert FSAVERAGE_SIZES["fsaverage7"] == 163842


def test_unknown_space_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactWriteError, match="Unknown surface space"):
        export_surface_mesh("fsaverage99", tmp_path)


@pytest.mark.skipif(
    not (Path("artifacts/mesh/fsaverage5/mapping.json").exists()),
    reason="run `blackmirror export-mesh` first (downloads the template surface)",
)
def test_exported_mesh_is_self_consistent() -> None:
    """Guards the exact invariants a visualizer depends on."""
    directory = Path("artifacts/mesh/fsaverage5")
    manifest = json.loads((directory / "mapping.json").read_text())

    per_hemisphere = manifest["vertices_per_hemisphere"]
    assert per_hemisphere == 10242
    assert manifest["total_vertices"] == 20484
    assert manifest["hemisphere_index_ranges"] == {
        "left": [0, 10242],
        "right": [10242, 20484],
    }

    for hemisphere in ("left", "right"):
        vertices = np.load(directory / f"{hemisphere}_inflated_vertices.npy")
        faces = np.load(directory / f"{hemisphere}_faces.npy")

        assert vertices.shape == (per_hemisphere, 3)
        assert faces.shape[1] == 3
        # Faces must index within their own hemisphere, or a visualizer would
        # stitch the two hemispheres together.
        assert faces.min() >= 0
        assert faces.max() < per_hemisphere

        pial = np.load(directory / f"{hemisphere}_pial_vertices.npy")
        assert pial.shape == vertices.shape

    mask = np.load(directory / "medial_wall_mask.npy")
    assert mask.shape == (20484,)
    assert mask.dtype == bool
    # The medial wall is a real but modest fraction of the surface, and roughly
    # symmetric across hemispheres.
    assert 0.02 < mask.mean() < 0.25
    left, right = int(mask[:10242].sum()), int(mask[10242:].sum())
    assert abs(left - right) / max(left, right) < 0.25
