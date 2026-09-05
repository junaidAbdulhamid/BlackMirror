# Data Contracts

What Phase 2+ can rely on. Everything here is stable within `schema_version 1.0`;
a breaking change bumps that field.

The governing rule: **arrays live in binary files, metadata lives in JSON.** No
large numeric array is ever embedded in a manifest.

---

## `StimulusInput`

`blackmirror.schemas.stimulus.StimulusInput` — a validated, content-addressed
piece of content.

| Field | Type | Notes |
|---|---|---|
| `stimulus_id` | `str` | Random hex; identifies an *ingestion*, not the content |
| `path` | `Path` | Absolute path to the source file. **Never copied into artifacts** |
| `filename` | `str` | |
| `media_type` | `"video" \| "audio" \| "text"` | |
| `mime_type` | `str \| None` | Best-effort from the extension |
| `file_size_bytes` | `int` | |
| `duration_seconds` | `float \| None` | From `ffprobe`. `None` for text or when ffprobe is absent — never inferred from the prediction |
| `sha256` | `str` (64 hex) | **The content identity.** Same bytes ⇒ same hash, regardless of filename |
| `created_at` | `datetime` (UTC) | Ingestion time, not file mtime |

`sha256` is what makes an experiment reproducible and is one of the three
components of the run cache key.

---

## `PredictionResult`

`blackmirror.schemas.prediction.PredictionResult` — serialized verbatim to
`manifest.json`. Loadable with nothing installed but `pydantic`:

```python
from blackmirror.schemas.prediction import PredictionResult
result = PredictionResult.model_validate_json(manifest_path.read_text())
```

Top-level: `schema_version`, `run_id`, `created_at`, `status`, `stimulus`,
`model`, `prediction`, `temporal`, `cortical`, `validation`, `performance`,
`provenance`, `artifacts`, `interpretation_notice`.

`status` is `completed`, `completed_with_warnings`, or `failed`.

### `prediction` — `PredictionArrayMetadata`

Describes `predictions.npy`.

| Field | Meaning |
|---|---|
| `shape` | `(T, V)` |
| `dtype` | e.g. `float32` |
| `axis_names` | `("time", "vertex")` |
| `temporal_samples` | `T` |
| `cortical_features` | `V` |
| `units` | `null` for TRIBE v2 — upstream publishes no physical unit |
| `normalization` | Normalization the **model** applies, where documented |
| `semantics` | Plain-language meaning of one value |
| `is_raw_model_output` | `true` — the file is byte-for-byte what the model returned |
| `artifact_path` | Run-relative path (`predictions.npy`) |

### `temporal` — `TemporalMetadata`

**The part Phase 2 cannot work without.**

| Field | Meaning |
|---|---|
| `n_time_points` | `T` |
| `tr_seconds` | Duration one row covers (TRIBE v2: `1.0`) |
| `sampling_rate_hz` | `1 / tr_seconds` |
| `timeline_is_contiguous` | `true` only if rows tile the stimulus at exactly TR spacing |
| `segments_were_filtered` | Whether event-free segments were dropped |
| `n_segments_total` / `n_segments_kept` | Segments before and after filtering. `n_segments_total` is recovered from TRIBE's own log record and validated; `null` means it could not be trusted, never a guess |
| `first_segment_start_seconds`, `last_segment_end_seconds`, `covered_seconds` | Timeline coverage |
| `hemodynamic_offset_seconds` | The lag the model was trained to compensate. `5.0` for TRIBE v2, read from `data.neuro.offset` |
| `hemodynamic_offset_verified` | `true` for TRIBE v2 — established from the model's config and training code, not prose |
| `hemodynamic_offset_source` | How the value and its sign were established |
| `output_is_stimulus_aligned` | `true` for TRIBE v2: a row's segment start **is** the stimulus content time, because the model already compensated the lag |
| `hemodynamic_offset_applied_to_raw` | Always `false` — the raw matrix is never shifted |
| `notes` | Caveats a consumer must see (e.g. non-physical derived times) |
| `arrays_artifact_path` | `temporal.npz` |
| `array_keys` | Key → description for each array in that file |
| `event_summary` | Counts and types of stimulus events |

> **Do not compute time as `row_index × TR`.** When
> `timeline_is_contiguous` is `false`, rows were dropped and that formula
> silently desynchronises playback from the response. Always read
> `segment_start_seconds`.

### `cortical` — `CorticalMetadata`

| Field | Meaning |
|---|---|
| `surface_space` | `fsaverage5` |
| `vertex_count` | `20484` |
| `vertices_per_hemisphere` | `10242` |
| `hemisphere_order` | `("left", "right")` |
| `hemisphere_index_ranges` | `{"left": [0, 10242], "right": [10242, 20484]}` — half-open |
| `is_surface_based` / `includes_subcortex` | `true` / `false` |
| `mesh_artifact_dir` | Exported mesh directory, when present |
| `medial_wall_handling` | How medial-wall vertices are treated |
| `atlas` | Parcellation, if any (`null` in Phase 1) |
| `mapping_notes` | Prose statement of the column → vertex rule |

**The mapping rule:** column `j` belongs to the hemisphere whose half-open range
contains `j`; its index within that hemisphere's mesh is `j - range_start`.

### `validation` — `PredictionValidation`

`valid`, `shape`, `dtype`, `nan_count`, `inf_count`, `finite_count`, `min`,
`max`, `mean`, `std`, `p01`, `p50`, `p99`, `constant_vertices`,
`temporal_variance_mean`, `warnings`, `errors`.

Statistics are computed over **finite values only**. Anomalies are reported and
the raw data is preserved; `valid=false` is reserved for *structural* failures
(wrong rank, zero-length axis, vertex-count mismatch), which abort the run.

### `performance` — `PerformanceMetrics`

Per-stage seconds (`preprocessing`, `model_load`, `inference`, `postprocessing`,
`validation`, `persistence`, `total`), peak memory where the platform reports
it, and `realtime_factor` (total seconds per second of content).

### `provenance` — `RunProvenance`

`run_id`, `created_at`, `blackmirror_version`, `python_version`, `platform`,
`torch_version`, `numpy_version`, `backend_package_version`, `device`, `dtype`,
`random_seed`, `stimulus_sha256`, `model_fingerprint`, `preprocessing_config`,
`cache_key`, `applied_compat_patches`.

**Before comparing two runs**, assert that `model_fingerprint`,
`preprocessing_config` and `applied_compat_patches` match. Otherwise it is not a
controlled comparison.

### `model` — `ModelMetadata`

`name`, `backend`, `model_id`, `source`, `checkpoint`, `version`, `license`,
`modalities`, `feature_extractors`, `loaded_device`, `dtype`,
`parameter_count`, `subject_conditioning`, `is_synthetic`, `extra`.

> Always check **`is_synthetic`** before interpreting anything. `true` means the
> mock backend produced test data, not a brain prediction.

---

## Artifact layout

```
artifacts/
├── index.jsonl                  # append-only run index (cache-key lookup)
├── runs/<run_id>/
│   ├── manifest.json            # the full PredictionResult
│   ├── stimulus.json
│   ├── predictions.npy          # [T, V] raw model output — IMMUTABLE
│   ├── temporal.npz             # per-row timing arrays
│   ├── events.parquet           # backend event table (when produced)
│   ├── events_summary.json
│   ├── prediction_metadata.json
│   ├── validation.json
│   ├── performance.json
│   ├── provenance.json
│   └── diagnostics/
│       └── global_mean_timecourse.png
├── mesh/fsaverage5/             # shared across runs
└── benchmarks/
```

`run_id` is `<YYYYmmddTHHMMSSZ>-<8 hex>`; the fixed-width UTC prefix makes
lexical order chronological.

### `predictions.npy`

`np.load(...)` → `(T, V)` array, exactly as the model returned it. **Never**
normalized, clipped, NaN-stripped, smoothed, resampled or reordered. NaN and Inf
values, if any, are preserved.

### `temporal.npz`

All arrays have length `T` and are aligned row-for-row with `predictions.npy`.

| Key | dtype | Meaning |
|---|---|---|
| `segment_start_seconds` | `float64` | **Ground truth.** Start of each row's segment on the stimulus timeline |
| `segment_duration_seconds` | `float64` | Duration each row covers |
| `segment_n_events` | `int64` | Stimulus events inside each segment |
| `stimulus_time_seconds` | `float64` | **Derived**: the content time each row describes. Equals `segment_start_seconds` when `output_is_stimulus_aligned` (TRIBE v2). **Sync playback on this** |
| `bold_acquisition_time_seconds` | `float64` | **Derived**: `stimulus_time + hemodynamic_offset` — when BOLD would physically be measured. Not a playback clock |

### `mesh/fsaverage5/`

Exported once by `blackmirror export-mesh`; shared by every run.

| File | Shape | Meaning |
|---|---|---|
| `left_pial_vertices.npy` | `(10242, 3)` | Left pial coordinates (mm) |
| `left_inflated_vertices.npy` | `(10242, 3)` | Left inflated coordinates |
| `left_faces.npy` | `(F, 3)` | Left triangles, **hemisphere-local** 0-based indices |
| `right_*` | same | Right hemisphere |
| `medial_wall_mask.npy` | `(20484,)` bool | `True` = medial wall; mask before interpreting |
| `mapping.json` | — | Index ranges, provenance, caveats |

Faces index into that hemisphere's own coordinate array. To colour the right
hemisphere, slice `row[10242:20484]` and index it directly.

---

## Consuming a run (Phase 2 recipe)

```python
import numpy as np
from blackmirror.schemas.prediction import PredictionResult

result = PredictionResult.model_validate_json((run_dir / "manifest.json").read_text())
assert not result.model.is_synthetic

predictions = np.load(run_dir / result.prediction.artifact_path)      # (T, V)
with np.load(run_dir / result.temporal.arrays_artifact_path) as t:
    times = t["stimulus_time_seconds"]        # (T,) content time — sync playback here
assert result.temporal.output_is_stimulus_aligned  # else re-check the convention

left_start, left_end = result.cortical.hemisphere_index_ranges["left"]
left_frame = predictions[0, left_start:left_end]                      # (10242,)

mesh = result.cortical.mesh_artifact_dir
vertices = np.load(mesh / "left_inflated_vertices.npy")               # (10242, 3)
faces = np.load(mesh / "left_faces.npy")
medial_wall = np.load(mesh / "medial_wall_mask.npy")[left_start:left_end]
```

Note this snippet imports **no** model dependency — no `tribev2`, no `torch`.
That is the abstraction boundary working.
