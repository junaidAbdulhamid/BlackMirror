"""Does a wider search space produce differences the pipeline can resolve?

THE QUESTION THIS ANSWERS
    Phase 9's real benchmark produced five candidates whose objective values
    spanned 0.004, against a measured nuisance floor of 0.0157. Everything the
    search could distinguish was smaller than the amount the whole pipeline
    moves when an ostensibly equivalent model mirror is swapped. No surrogate
    trained on that can demonstrate anything, and no sample-efficiency claim
    built on it would mean anything.

    The cheapest fix is a wider space. This evaluates the corners of one, which
    is the smallest experiment that bounds the achievable spread: if the
    extremes of a space cannot separate by more than the floor, nothing inside
    it can either.

WHY CORNERS
    Four evaluations rather than a grid. The corners of a box maximise pairwise
    distance in the space, so the spread they produce is an upper bound on what
    any point set inside the box could produce. A negative result here is
    therefore conclusive, which is what makes it worth an hour and a half.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from blackmirror.api.scoring_loader import ScoringLoader
from blackmirror.inference.service import InferenceService
from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.resimulation.backend import ExistingPipelineBackend
from blackmirror.resimulation.orchestrator import ResimulationOrchestrator
from blackmirror.search.cache import CacheIdentity, CachingEvaluator, SearchEvaluationCache
from blackmirror.search.decode import GenomeDecoder
from blackmirror.search.evaluator import CandidateEvaluator
from blackmirror.search.genome import CandidateGenome, GenomeOrigin
from blackmirror.search.schemas import MEASURED_NUISANCE_FLOOR
from blackmirror.search.space import ContentSearchSpace, continuous

ARTIFACTS = Path("artifacts")
EXPERIMENT = "search-bench"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="20260906T225022Z-630b9b86")
    parser.add_argument("--gain", type=float, default=15.0, help="+/- dB at the corners.")
    parser.add_argument("--brightness", type=float, default=0.35)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/benchmarks/space_range.json")
    )
    args = parser.parse_args()

    manifest = json.loads((ARTIFACTS / "runs" / args.root / "manifest.json").read_text())
    media = Path(manifest["stimulus"]["path"])
    duration = float(manifest["stimulus"]["duration_seconds"])

    scores = ScoringLoader(ARTIFACTS)
    stored = [
        item
        for item in scores.store.list_for_experiment(EXPERIMENT)
        if args.root in {variant.variant_id for variant in item.variant_scores}
    ]
    if not stored:
        raise SystemExit(f"no stored score for {EXPERIMENT} containing {args.root}")
    score = stored[0]
    objective = score.objectives[0]
    root_value = score.variant_scores[0].objective_scores[0].raw_value

    space = ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -args.gain, args.gain, resolution=0.5),
            continuous(
                "brightness", Op.BRIGHTNESS, -args.brightness, args.brightness,
                resolution=0.01,
            ),
        )
    )
    search_id = f"range-g{args.gain:g}-b{args.brightness:g}"
    decoder = GenomeDecoder(
        search_id=search_id,
        experiment_id=EXPERIMENT,
        space=space,
        parent_variant_id=args.root,
        root_media_path=str(media),
        root_duration_seconds=duration,
        objective=objective,
    )
    evaluator = CandidateEvaluator(
        ARTIFACTS,
        decoder,
        ResimulationOrchestrator(ARTIFACTS, ExistingPipelineBackend(ARTIFACTS, InferenceService())),
        objectives=score.objectives,
        objective_set_hash=score.metadata.objective_set_hash,
        objective_definition_hash=score.metadata.objective_definition_hash,
        candidate_dir=ARTIFACTS / "search" / search_id / "candidates",
    )
    cached = CachingEvaluator(
        evaluator,
        SearchEvaluationCache(
            ARTIFACTS,
            CacheIdentity(
                root_media_sha256=manifest["stimulus"]["sha256"],
                objective_definition_hash=score.metadata.objective_definition_hash,
                space_version=space.search_space_version,
                materialization_version="1.0",
                scoring_version=score.metadata.scoring_version,
                model_fingerprint=manifest["provenance"]["model_fingerprint"],
            ),
        ),
        space,
    )

    corners = list(itertools.product((-args.gain, args.gain), (-args.brightness, args.brightness)))
    print(f"root fitness {root_value:+.6f}; evaluating {len(corners)} corners", flush=True)

    rows: list[dict[str, object]] = []
    for index, (gain, brightness) in enumerate(corners, start=1):
        genome = CandidateGenome(
            genome_id=f"corner{index}",
            values={"gain": gain, "brightness": brightness},
            origin=GenomeOrigin.MANUAL,
            proposed_by="range_finding",
        )
        started = time.perf_counter()
        fitness = cached.evaluate(genome)
        rows.append(
            {
                "candidate_id": genome.genome_id,
                "gain": gain,
                "brightness": brightness,
                "fitness": fitness.scalar_fitness,
                "status": fitness.status.value,
                "reason": fitness.reason,
                "wall_seconds": time.perf_counter() - started,
                "cached": fitness.compute.served_from_cache,
            }
        )
        print(
            f"  corner{index} gain={gain:+g} brightness={brightness:+g} "
            f"-> {fitness.scalar_fitness} ({fitness.status.value}, "
            f"{(time.perf_counter() - started) / 60:.1f} min)",
            flush=True,
        )

    values = [row["fitness"] for row in rows if isinstance(row["fitness"], float)]
    spread = (max(values) - min(values)) if len(values) > 1 else 0.0
    payload = {
        "root_run": args.root,
        "root_fitness": root_value,
        "gain_extent": args.gain,
        "brightness_extent": args.brightness,
        "corners": rows,
        "spread": spread,
        "nuisance_floor": MEASURED_NUISANCE_FLOOR,
        "spread_over_floor": spread / MEASURED_NUISANCE_FLOOR,
        "resolvable": spread > MEASURED_NUISANCE_FLOOR,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"spread          : {spread:.6f}")
    print(f"nuisance floor  : {MEASURED_NUISANCE_FLOOR}")
    print(f"spread / floor  : {spread / MEASURED_NUISANCE_FLOOR:.2f}x")
    print(
        "RESOLVABLE: a search over this space can produce differences larger "
        "than the floor."
        if spread > MEASURED_NUISANCE_FLOOR
        else "NOT RESOLVABLE: even the corners of this space differ by less than "
        "the floor, so no point inside it can be separated either."
    )
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
