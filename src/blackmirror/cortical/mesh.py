"""Cortical surface mesh export.

Answers the question Phase 2 cannot start without: *given column j of the
prediction matrix, where on the cortex do I paint it?*

TRIBE v2 predicts onto **fsaverage5**, a standard FreeSurfer template surface
with 10,242 vertices per hemisphere, concatenated left-then-right into a 20,484
long vector (``tribev2/utils_fmri.py``: ``FSAVERAGE_5 = ("fsaverage5", (10242,))``;
``tribev2/plotting/cortical.py`` slices ``[:N]`` / ``[N:]``).

The mesh itself is not TRIBE-specific — it is the public FreeSurfer template,
obtained here through nilearn. That is deliberate: exporting it does not require
the model, so Phase 2 visualization work is unblocked independently of the
checkpoint, the gated Llama access or GPU availability.

Exported once into ``artifacts/mesh/<space>/`` and shared by every run; the
geometry never changes between runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blackmirror.errors import ArtifactWriteError, BackendUnavailableError
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: Vertices per hemisphere for each fsaverage resolution.
FSAVERAGE_SIZES: dict[str, int] = {
    "fsaverage3": 642,
    "fsaverage4": 2562,
    "fsaverage5": 10242,
    "fsaverage6": 40962,
    "fsaverage7": 163842,
}

HEMISPHERE_ORDER = ("left", "right")

#: Surfaces worth exporting. `pial` is anatomically faithful; `inflated` is what
#: you actually want to look at, because sulcal activity is hidden on the pial
#: surface. Both share vertex indexing, so one prediction vector colours either.
_SURFACES = ("pial", "inflated")

MANIFEST_FILE = "mapping.json"


def export_surface_mesh(
    space: str,
    mesh_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Export ``space``'s vertices, faces and index mapping to ``mesh_root/<space>``.

    Raises:
        BackendUnavailableError: nilearn is not installed.
        ArtifactWriteError: the export could not be written.
    """
    if space not in FSAVERAGE_SIZES:
        raise ArtifactWriteError(
            f"Unknown surface space '{space}'. Known: {sorted(FSAVERAGE_SIZES)}"
        )

    output_dir = mesh_root / space
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists() and not force:
        logger.info("Mesh already exported at %s (use --force to re-export)", output_dir)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    try:
        from nilearn.datasets import load_fsaverage
    except ImportError as exc:
        raise BackendUnavailableError(
            "nilearn is required to export the cortical mesh. "
            "Install it with `uv pip install -e '.[viz]'`."
        ) from exc

    expected_per_hemisphere = FSAVERAGE_SIZES[space]
    logger.info("Fetching %s surfaces via nilearn (downloads on first use)", space)

    try:
        fsaverage = load_fsaverage(mesh=space)
    except Exception as exc:
        raise ArtifactWriteError(
            f"Could not fetch the {space} surface via nilearn: {type(exc).__name__}: {exc}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, dict[str, Any]] = {}

    for surface in _SURFACES:
        polymesh = fsaverage[surface]
        for hemisphere in HEMISPHERE_ORDER:
            part = polymesh.parts[hemisphere]
            coordinates = np.asarray(part.coordinates, dtype=np.float32)
            faces = np.asarray(part.faces, dtype=np.int32)

            if coordinates.shape[0] != expected_per_hemisphere:
                raise ArtifactWriteError(
                    f"{space} {hemisphere} {surface} has {coordinates.shape[0]} vertices, "
                    f"expected {expected_per_hemisphere}. The prediction-to-vertex mapping "
                    f"would be wrong, so the export is refused."
                )

            vertex_name = f"{hemisphere}_{surface}_vertices.npy"
            np.save(output_dir / vertex_name, coordinates, allow_pickle=False)
            arrays[vertex_name] = {
                "shape": list(coordinates.shape),
                "dtype": str(coordinates.dtype),
                "description": f"{hemisphere} hemisphere {surface} vertex coordinates (mm, x/y/z)",
            }

            # Faces are identical across surfaces of the same hemisphere
            # (only coordinates move), so write them once.
            face_name = f"{hemisphere}_faces.npy"
            if face_name not in arrays:
                np.save(output_dir / face_name, faces, allow_pickle=False)
                arrays[face_name] = {
                    "shape": list(faces.shape),
                    "dtype": str(faces.dtype),
                    "description": (
                        f"{hemisphere} hemisphere triangles; each row holds 3 vertex indices "
                        f"into that hemisphere's coordinate array (0-based, hemisphere-local)"
                    ),
                }

    medial_wall = _export_medial_wall(space, output_dir, expected_per_hemisphere, arrays)

    total = expected_per_hemisphere * len(HEMISPHERE_ORDER)
    manifest: dict[str, Any] = {
        "surface_space": space,
        "source": "FreeSurfer fsaverage template, fetched via nilearn.datasets.load_fsaverage",
        "vertices_per_hemisphere": expected_per_hemisphere,
        "total_vertices": total,
        "hemisphere_order": list(HEMISPHERE_ORDER),
        "hemisphere_index_ranges": {
            "left": [0, expected_per_hemisphere],
            "right": [expected_per_hemisphere, total],
        },
        "prediction_mapping": (
            f"A prediction row is a length-{total} vector. Columns "
            f"[0, {expected_per_hemisphere}) are left-hemisphere vertices in mesh order; "
            f"columns [{expected_per_hemisphere}, {total}) are right-hemisphere vertices in "
            "mesh order. To colour a hemisphere, slice the row by that hemisphere's range "
            "and index it directly against that hemisphere's vertex array."
        ),
        "surfaces": list(_SURFACES),
        "faces_are_hemisphere_local": True,
        "medial_wall": medial_wall,
        "caveats": [
            "Vertices carry no functional meaning on their own; they are geometry.",
            "The model emits a value for every vertex, including the medial wall, where "
            "fMRI signal is not meaningful. Mask it before drawing conclusions.",
            "Left/right ordering is asserted by the upstream model's own plotting code, "
            "not inferred by BlackMirror.",
        ],
        "arrays": arrays,
    }

    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError as exc:
        raise ArtifactWriteError(f"Could not write {manifest_path}: {exc}") from exc

    logger.info("Exported %s mesh (%d vertices) to %s", space, total, output_dir)
    return manifest


def _export_medial_wall(
    space: str,
    output_dir: Path,
    per_hemisphere: int,
    arrays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Best-effort medial-wall mask from the Destrieux surface atlas.

    The Destrieux atlas ships in fsaverage5 space and labels the medial wall as
    ``Unknown``. For other resolutions we skip rather than resample, because a
    silently interpolated anatomical mask is worse than no mask.
    """
    if space != "fsaverage5":
        return {
            "available": False,
            "reason": (
                f"The Destrieux atlas ships in fsaverage5 space; no mask was produced for "
                f"{space} rather than resampling an anatomical label map."
            ),
        }

    try:
        from nilearn.datasets import fetch_atlas_surf_destrieux
    except ImportError:
        return {"available": False, "reason": "nilearn atlas fetcher unavailable"}

    try:
        atlas = fetch_atlas_surf_destrieux()
        labels = [
            label.decode() if isinstance(label, bytes) else str(label) for label in atlas["labels"]
        ]
        unknown_indices = {
            i for i, name in enumerate(labels) if name.strip().lower() in {"unknown", "medial_wall"}
        }
        if not unknown_indices:
            return {"available": False, "reason": "no 'Unknown'/'Medial_wall' label in the atlas"}

        left = np.isin(np.asarray(atlas["map_left"]), list(unknown_indices))
        right = np.isin(np.asarray(atlas["map_right"]), list(unknown_indices))
        if left.shape[0] != per_hemisphere or right.shape[0] != per_hemisphere:
            return {
                "available": False,
                "reason": (
                    f"atlas hemispheres are {left.shape[0]}/{right.shape[0]} vertices, "
                    f"expected {per_hemisphere}"
                ),
            }

        mask = np.concatenate([left, right]).astype(bool)
    except Exception as exc:
        logger.warning("Could not derive the medial-wall mask: %s: %s", type(exc).__name__, exc)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    name = "medial_wall_mask.npy"
    np.save(output_dir / name, mask, allow_pickle=False)
    arrays[name] = {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "description": (
            "Boolean mask over the full concatenated vertex axis. True marks medial-wall "
            "vertices, where fMRI signal is not meaningful."
        ),
    }
    return {
        "available": True,
        "artifact": name,
        "source": "nilearn.datasets.fetch_atlas_surf_destrieux ('Unknown' label)",
        "n_masked_vertices": int(mask.sum()),
        "usage": "Exclude these columns before interpreting or ranking regional responses.",
    }
