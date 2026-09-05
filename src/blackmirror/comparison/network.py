"""Verified native-fsaverage5 Yeo 2011 functional-network mapping."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from blackmirror.analytics.metrics import aggregate_regions

MANIFEST_SHA256 = "d6e88be4e581986cfdf6ba1872dad8540d63a96ee2122aa1eae8d446444edff7"
NETWORK_NAMES = tuple(f"7Networks_{index}" for index in range(1, 8))


@dataclass(frozen=True)
class FunctionalNetworkMapping:
    name: str
    version: str
    network_names: tuple[str, ...]
    vertex_to_network: NDArray[np.int32]
    medial_wall_mask: NDArray[np.bool_]
    mapping_sha256: str

    def aggregate(self, responses: NDArray[np.floating]) -> NDArray[np.float64]:
        return aggregate_regions(responses, self.vertex_to_network, len(self.network_names))


def load_yeo7_mapping(atlas_dir: Path, mesh_dir: Path) -> FunctionalNetworkMapping:
    """Load only after checksums, coordinates, labels, and wall match exactly."""
    try:
        from nibabel.freesurfer.io import read_annot, read_geometry
    except ImportError as exc:
        raise RuntimeError("Yeo atlas loading requires nibabel") from exc
    manifest_path = atlas_dir / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Yeo trusted manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = atlas_dir / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Yeo atlas checksum mismatch: {relative}")
    wall = np.load(mesh_dir / "medial_wall_mask.npy", allow_pickle=False).astype(bool)
    mappings: list[NDArray[np.int32]] = []
    offset = 0
    if tuple(manifest["network_names"]) != NETWORK_NAMES:
        raise ValueError("Yeo trusted network names differ from manifest")
    expected_names = ("FreeSurfer_Defined_Medial_Wall", *NETWORK_NAMES)
    for short, full in (("lh", "left"), ("rh", "right")):
        coordinates, _ = read_geometry(atlas_dir / "surf" / f"{short}.pial")
        mesh = np.load(mesh_dir / f"{full}_pial_vertices.npy", allow_pickle=False)
        if not np.array_equal(coordinates.astype(np.float32), mesh):
            raise ValueError(f"Yeo {full} source surface does not exactly match project mesh")
        labels, _, names_raw = read_annot(
            atlas_dir / "label" / f"{short}.Yeo2011_7Networks_N1000.annot"
        )
        names = tuple(value.decode() for value in names_raw)
        if names != expected_names or labels.shape != (mesh.shape[0],):
            raise ValueError(f"Yeo {full} annotation label table is unexpected")
        local_wall = wall[offset : offset + len(labels)]
        if not np.array_equal(labels == 0, local_wall):
            raise ValueError(f"Yeo {full} medial wall differs from project mask")
        dense = labels.astype(np.int32) - 1
        dense[local_wall] = -1
        if set(np.unique(dense)) != {-1, 0, 1, 2, 3, 4, 5, 6}:
            raise ValueError(f"Yeo {full} network ids are incomplete or invalid")
        mappings.append(dense)
        offset += len(labels)
    combined = np.concatenate(mappings)
    mapping_hash = hashlib.sha256(combined.astype("<i4", copy=False).tobytes()).hexdigest()
    return FunctionalNetworkMapping(
        name=manifest["name"],
        version=manifest["version"],
        network_names=NETWORK_NAMES,
        vertex_to_network=combined,
        medial_wall_mask=wall,
        mapping_sha256=mapping_hash,
    )
