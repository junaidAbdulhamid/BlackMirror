"""Whether a parameter that looks inert to a whole-cortex mean is actually inert.

WHY THIS EXISTS
    The wide-space range measurement (`measure_space_range.py`) found that a
    30 dB swing in audio gain moved the predicted whole-cortex mean by 1-3% of
    the nuisance floor, while the same swing in brightness moved it by ~100%.
    Read literally that says loudness does nothing.

    A whole-cortex mean averages 20,484 vertices. An effect confined to a small
    region, or one with opposite signs in different places, cancels in that
    average while being perfectly real in the predictions. So before concluding
    that a parameter does nothing, the same contrast is taken per region.

WHAT IT REPORTS, AND WHAT IT DOES NOT
    Per-ROI mean differences for a pair of runs, the ROIs with the largest
    effects, and the ratio of mean absolute per-vertex change to the change in
    the whole-cortex mean. A large ratio means cancellation: the parameter moved
    the prediction, the summary hid it.

    It does not report a significance test. There is one clip and one run per
    condition, and the pipeline is deterministic, so there is nothing to average
    over and no scatter to estimate. What makes the output worth reading is
    whether the regions that move are the ones a manipulation of that modality
    would be expected to move, and whether that holds across both levels of the
    other parameter. That is a coherence check, not a p-value.

    Note also that the 0.0157 nuisance floor was measured as a *whole-cortex*
    mean shift. An ROI mean over a few dozen vertices is a noisier quantity than
    a mean over twenty thousand, so the ROI-level floor is almost certainly
    higher and has not been measured. Ratios against the whole-cortex floor are
    printed as orientation, not as a threshold anything passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blackmirror.analytics.atlas import DestrieuxAtlas
from blackmirror.search.schemas import MEASURED_NUISANCE_FLOOR

MIN_VERTICES = 20


def _mean_response(runs_root: Path, run_id: str) -> np.ndarray:
    """Time-averaged predicted response per vertex, as the objective sees it."""
    path = runs_root / run_id / "predictions.npy"
    if not path.is_file():
        raise SystemExit(f"no predictions at {path}")
    return np.asarray(np.load(path)).mean(axis=0)


def _contrast(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    cortex: np.ndarray,
    lookup: dict[int, tuple[str, str]],
    *,
    top: int,
) -> dict[str, object]:
    difference = left - right
    whole = float(difference[cortex].mean())
    absolute = float(np.abs(difference[cortex]).mean())

    rows: list[tuple[float, int, int]] = []
    for region in np.unique(labels[cortex]):
        selected = cortex & (labels == region)
        if int(selected.sum()) >= MIN_VERTICES:
            rows.append((float(difference[selected].mean()), int(region), int(selected.sum())))
    rows.sort(key=lambda item: abs(item[0]), reverse=True)

    return {
        "whole_cortex_mean_difference": whole,
        "mean_absolute_vertex_difference": absolute,
        # How much the whole-cortex mean hides. 1.0 would mean every vertex moved
        # the same way; a large value means the effect cancels against itself.
        "cancellation_ratio": (absolute / abs(whole)) if whole else None,
        "whole_cortex_over_floor": abs(whole) / MEASURED_NUISANCE_FLOOR,
        "largest_roi_effect": abs(rows[0][0]) if rows else None,
        "largest_roi_over_whole_cortex_floor": (
            abs(rows[0][0]) / MEASURED_NUISANCE_FLOOR if rows else None
        ),
        "regions_considered": len(rows),
        "top_regions": [
            {
                "name": lookup.get(region, (f"region {region}", "?"))[0],
                "hemisphere": lookup.get(region, ("", "?"))[1],
                "mean_difference": value,
                "vertices": count,
            }
            for value, region, count in rows[:top]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contrast",
        action="append",
        required=True,
        metavar="NAME=RUN_A,RUN_B",
        help="a named contrast; the reported difference is RUN_A minus RUN_B",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--mesh-dir", type=Path, default=Path("artifacts/mesh/fsaverage5"))
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmarks/roi_effects.json"))
    args = parser.parse_args()

    mapping = DestrieuxAtlas().load(args.mesh_dir)
    metadata = mapping.metadata.model_dump()
    lookup = {
        region["region_id"]: (region["name"], str(region["hemisphere"].value))
        for region in metadata["regions"]
    }
    labels = mapping.vertex_to_region
    cortex = ~mapping.medial_wall_mask

    results: dict[str, object] = {}
    for raw in args.contrast:
        name, _, pair = raw.partition("=")
        left_id, _, right_id = pair.partition(",")
        if not (name and left_id and right_id):
            raise SystemExit(f"malformed --contrast {raw!r}; expected NAME=RUN_A,RUN_B")
        left = _mean_response(args.runs_root, left_id.strip())
        right = _mean_response(args.runs_root, right_id.strip())
        outcome = _contrast(left, right, labels, cortex, lookup, top=args.top)
        outcome["runs"] = {"left": left_id.strip(), "right": right_id.strip()}
        results[name] = outcome

        print(f"\n{name}  ({left_id.strip()} minus {right_id.strip()})")
        print(f"  whole-cortex mean difference : {outcome['whole_cortex_mean_difference']:+.6f}"
              f"  ({outcome['whole_cortex_over_floor']:.2f}x the whole-cortex floor)")
        print(f"  mean |per-vertex difference| : "
              f"{outcome['mean_absolute_vertex_difference']:.6f}")
        ratio = outcome["cancellation_ratio"]
        if isinstance(ratio, float):
            print(f"  cancellation ratio           : {ratio:.1f}x")
        print(f"  largest ROI effect           : {outcome['largest_roi_effect']:.6f}")
        for region in outcome["top_regions"]:  # type: ignore[index]
            print(
                f"     {region['mean_difference']:+.6f}  "
                f"{region['hemisphere'][:1].upper()}H {region['name']}  "
                f"({region['vertices']} vtx)"
            )

    payload = {
        "nuisance_floor_whole_cortex": MEASURED_NUISANCE_FLOOR,
        "minimum_vertices_per_region": MIN_VERTICES,
        "atlas": metadata["name"],
        "caveat": (
            "Predicted responses, not measurements. One clip, one deterministic "
            "run per condition, so no significance test is possible. The 0.0157 "
            "floor is a whole-cortex quantity; the ROI-level floor is higher and "
            "has not been measured."
        ),
        "contrasts": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
