"""Verified cortical atlas loading and dense vertex-to-region mappings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from blackmirror.analytics.schemas import (
    AtlasMetadata,
    AtlasValidationReport,
    CorticalRegion,
    Hemisphere,
)

UNMAPPED = -1


@dataclass(frozen=True)
class AtlasMapping:
    """Dense mapping whose indices exactly match prediction columns."""

    metadata: AtlasMetadata
    vertex_to_region: NDArray[np.int32]
    medial_wall_mask: NDArray[np.bool_]


class CorticalAtlas(Protocol):
    name: str
    version: str
    surface_space: str

    def load(self, mesh_dir: Path) -> AtlasMapping: ...


class DestrieuxAtlas:
    """Destrieux 2009 sulco-gyral atlas distributed with FreeSurfer.

    Nilearn exposes the left/right annotations in the same fsaverage5 vertex
    order as ``load_fsaverage('fsaverage5')``. Loading refuses the atlas unless
    those source coordinates match the exported Phase 2 mesh element-for-element.
    """

    name = "Destrieux 2009"
    version = "aparc.a2009s"
    surface_space = "fsaverage5"
    source = "nilearn.datasets.fetch_atlas_surf_destrieux; Destrieux et al. (2010)"
    license = "Unknown upstream; see nilearn dataset documentation"

    def __init__(self, *, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir

    def load(self, mesh_dir: Path) -> AtlasMapping:
        try:
            from nilearn.datasets import fetch_atlas_surf_destrieux, load_fsaverage
        except ImportError as exc:
            raise RuntimeError("Destrieux atlas loading requires the 'viz' dependencies") from exc

        atlas = fetch_atlas_surf_destrieux(data_dir=self.data_dir)
        fsaverage = load_fsaverage("fsaverage5", data_dir=self.data_dir)
        labels = tuple(_label_text(value) for value in atlas["labels"])
        wall_path = mesh_dir / "medial_wall_mask.npy"
        wall = np.load(wall_path, allow_pickle=False).astype(bool)
        expected_vertices = 2 * int(np.asarray(atlas["map_left"]).size)
        if wall.shape != (expected_vertices,):
            raise ValueError(
                f"medial-wall mask has shape {wall.shape}, expected {(expected_vertices,)}"
            )

        maps: list[NDArray[np.int32]] = []
        regions: list[CorticalRegion] = []
        next_region = 0
        offset = 0
        for hemi_name, hemisphere in (
            ("left", Hemisphere.LEFT),
            ("right", Hemisphere.RIGHT),
        ):
            atlas_labels = np.asarray(atlas[f"map_{hemi_name}"], dtype=np.int32)
            exported = np.load(mesh_dir / f"{hemi_name}_pial_vertices.npy", allow_pickle=False)
            source = np.asarray(fsaverage["pial"].parts[hemi_name].coordinates, dtype=np.float32)
            if not np.array_equal(exported, source):
                raise ValueError(
                    f"{hemi_name} atlas source coordinates do not exactly match exported mesh; "
                    "vertex order is unverified"
                )
            if atlas_labels.shape != (exported.shape[0],):
                raise ValueError(f"{hemi_name} atlas label count does not match mesh")

            dense = np.full(atlas_labels.shape, UNMAPPED, dtype=np.int32)
            local_wall = wall[offset : offset + atlas_labels.size]
            for atlas_label in np.unique(atlas_labels[~local_wall]):
                if atlas_label < 0 or atlas_label >= len(labels):
                    raise ValueError(f"Invalid Destrieux label {atlas_label} in {hemi_name}")
                selected = (atlas_labels == atlas_label) & ~local_wall
                count = int(selected.sum())
                if count == 0:
                    continue
                dense[selected] = next_region
                regions.append(
                    CorticalRegion(
                        region_id=next_region,
                        atlas_label=int(atlas_label),
                        name=labels[int(atlas_label)],
                        hemisphere=hemisphere,
                        atlas=self.name,
                        atlas_version=self.version,
                        vertex_count=count,
                    )
                )
                next_region += 1
            maps.append(dense)
            offset += atlas_labels.size

        metadata = AtlasMetadata(
            name=self.name,
            version=self.version,
            surface_space=self.surface_space,
            source=self.source,
            license=self.license,
            region_count=len(regions),
            regions=tuple(regions),
        )
        return AtlasMapping(metadata, np.concatenate(maps), wall)


def validate_atlas_mapping(
    mapping: AtlasMapping,
    *,
    prediction_vertices: int,
    mesh_dir: Path,
) -> AtlasValidationReport:
    """Validate counts, range, region occupancy, hemispheres, and medial wall."""
    vertex_map = np.asarray(mapping.vertex_to_region)
    wall = np.asarray(mapping.medial_wall_mask, dtype=bool)
    manifest = json.loads((mesh_dir / "mapping.json").read_text(encoding="utf-8"))
    mesh_vertices = int(manifest["total_vertices"])
    errors: list[str] = []
    warnings: list[str] = []
    if vertex_map.ndim != 1:
        errors.append("vertex_to_region must be one-dimensional")
    if manifest.get("surface_space") not in (None, mapping.metadata.surface_space):
        errors.append("atlas and mesh surface spaces differ")
    region_ids = [region.region_id for region in mapping.metadata.regions]
    if mapping.metadata.region_count != len(mapping.metadata.regions):
        errors.append("declared region count does not match region metadata")
    if region_ids != list(range(mapping.metadata.region_count)):
        errors.append("region ids must be unique and contiguous from zero")
    atlas_vertices = int(vertex_map.size)
    if prediction_vertices != mesh_vertices:
        errors.append("prediction and mesh vertex counts differ")
    if atlas_vertices != mesh_vertices:
        errors.append("atlas and mesh vertex counts differ")
    if wall.shape != vertex_map.shape:
        errors.append("medial-wall mask and atlas mapping shapes differ")
        wall = np.zeros(vertex_map.shape, dtype=bool)
    invalid = (vertex_map < UNMAPPED) | (vertex_map >= mapping.metadata.region_count)
    invalid_count = int(invalid.sum())
    if invalid_count:
        errors.append("atlas mapping contains out-of-range region ids")
    if np.any(vertex_map[wall] != UNMAPPED):
        errors.append("medial-wall vertices must be unmapped")
    mapped = vertex_map >= 0
    counts = np.bincount(vertex_map[mapped], minlength=mapping.metadata.region_count)
    if counts.size != mapping.metadata.region_count or np.any(counts == 0):
        errors.append("one or more declared regions has no mapped vertices")
    ranges = manifest["hemisphere_index_ranges"]
    hemisphere_counts = {
        hemi: int(mapped[int(bounds[0]) : int(bounds[1])].sum())
        for hemi, bounds in ranges.items()
    }
    unknown = (~mapped) & ~wall
    unknown_count = int(unknown.sum())
    if unknown_count:
        warnings.append(f"{unknown_count} non-medial-wall vertices are unmapped")
    # Coverage is relative to prediction columns, not all atlas entries. Keep the
    # report constructible even when a mismatched atlas is longer than predictions.
    mapped_count = int(mapped[:prediction_vertices].sum())
    for region in mapping.metadata.regions:
        actual = int((vertex_map == region.region_id).sum())
        if actual != region.vertex_count:
            errors.append(f"region {region.region_id} vertex count does not match mapping")
        bounds = ranges.get(region.hemisphere.value)
        if bounds is None:
            errors.append(f"mesh has no {region.hemisphere.value} hemisphere range")
        elif np.any(
            vertex_map[: int(bounds[0])] == region.region_id
        ) or np.any(vertex_map[int(bounds[1]) :] == region.region_id):
            errors.append(f"region {region.region_id} crosses its declared hemisphere")
    denominator = max(1, prediction_vertices)
    return AtlasValidationReport(
        valid=not errors,
        prediction_vertices=prediction_vertices,
        mesh_vertices=mesh_vertices,
        atlas_vertices=atlas_vertices,
        mapped_vertices=mapped_count,
        medial_wall_vertices=int(wall.sum()),
        unknown_vertices=unknown_count,
        invalid_vertices=invalid_count,
        mapped_fraction=mapped_count / denominator,
        hemisphere_counts=hemisphere_counts,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _label_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
