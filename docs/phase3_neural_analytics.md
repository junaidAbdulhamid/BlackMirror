# Phase 3 — Neural Analytics and Interpretation Engine

Phase 3 creates a strict derived-data layer:

`raw T x V prediction -> verified atlas -> T x ROI -> metrics -> events`

Raw `predictions.npy` is never changed. Analytics are written beside it under
`runs/<run_id>/analytics/`, versioned independently at analytics version 1.1:

- `metadata.json`: atlas provenance, validation, ROI summaries, events, parameters
- `timeseries.npz`: stimulus time, ROI/global/hemisphere series, changes,
  concentration, and ROI response correlation
- `atlas_mapping.npz`: dense vertex-to-region mapping and medial-wall mask

## Architecture

`DestrieuxAtlas` verifies surface identity and builds a reusable mapping.
`validate_atlas_mapping` reports counts, hemispheres, wall, unknown, and invalid
vertices. `NeuralAnalyticsEngine` performs vectorized aggregation and numerical
metrics. `NeuralEventDetector` applies transparent peak rules.
`AnalyticsStore` writes versioned binary/JSON artifacts atomically.

The CLI command `blackmirror analyze [RUN_ID]` derives analytics from an
existing completed run. It uses `stimulus_time_seconds`, including sparse
timelines; it never assumes `row_index * TR`.

The read-only API exposes:

- `GET /api/runs/{id}/analytics`
- `GET /api/runs/{id}/analytics/arrays/{key}`

Bulk arrays use binary transport. Unknown/uncomputed analytics return 404;
opening a visualization never triggers model inference or analytics computation.

## Numerical and performance design

ROI means use indexed NumPy reductions with no nested Python time/region/vertex
loop. Nonfinite values are omitted per timestep and ROI and remain visible in
Phase 1 validation. Whole-cortex, hemisphere, change, and concentration metrics
also exclude the medial wall. The fsaverage5 benchmark script is
`scripts/benchmark_analytics.py`; its recorded output is
`artifacts/benchmarks/phase3_analytics.json`.

## Scientific boundary

The atlas answers *where*. Metrics quantify predicted response and change.
Neither supplies causal or psychological interpretation. The implementation
does not contain emotion, engagement, memory, purchase-intent, or attention
scores. Functional-network aggregation was deferred until a network atlas could
be independently verified in the exact surface order; that verification now
exists in Phase 5 (`blackmirror.comparison.network`), which checks the Yeo 2011
files against a pinned manifest checksum, requires the atlas source surface
coordinates to equal this project's mesh exactly, and requires its medial wall
to match the project mask before any aggregation is permitted. Phase 3 itself
still emits no network series; the mapping is applied at comparison time.

Phase 4 may consume stimulus-aligned timestamps, ranked mathematical events,
ROI series, and global/change series to align content events. It must preserve
the same measurement-versus-interpretation boundary.
