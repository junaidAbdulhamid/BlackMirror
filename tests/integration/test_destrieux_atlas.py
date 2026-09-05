from pathlib import Path

import pytest

from blackmirror.analytics.atlas import DestrieuxAtlas, validate_atlas_mapping

MESH = Path("artifacts/mesh/fsaverage5")


@pytest.mark.skipif(not MESH.exists(), reason="exported fsaverage5 mesh unavailable")
def test_destrieux_mapping_matches_exported_mesh_order() -> None:
    pytest.importorskip("nilearn")
    mapping = DestrieuxAtlas().load(MESH)
    report = validate_atlas_mapping(mapping, prediction_vertices=20484, mesh_dir=MESH)
    assert report.valid, report.errors
    # 75 labels per hemisphere are present in the annotation, but one is the
    # non-cortical Medial_wall label and is deliberately excluded: 74 * 2.
    assert mapping.metadata.region_count == 148
    assert report.atlas_vertices == 20484
    assert report.medial_wall_vertices == 1769
    assert report.unknown_vertices == 0
