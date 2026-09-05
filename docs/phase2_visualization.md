# Phase 2 — Interactive Cortical Visualization

Turns the raw TRIBE v2 predictions from Phase 1 into an interactive 3D cortical
surface, synchronised with the original stimulus.

> **The frontend never runs TRIBE.** Every byte it displays comes from artifacts
> a completed Phase 1 run already wrote. Rotating the brain or dragging the
> timeline touches no model.

---

## Architecture

```
Phase 1 artifacts (predictions.npy, temporal.npz, mesh/, manifest.json)
        │
        ▼
VisualizationLoader        reuses ArtifactStore + PredictionResult
        │                  validates prediction↔mesh alignment
        ▼
FastAPI  (read-only)       JSON metadata + raw binary arrays
        │
        ▼
lib/api.ts                 fetch -> typed arrays, re-validates
        │
        ▼
neuralVisualizationStore   ONE source of truth for time
        │
        ▼
BrainViewer -> BrainScene -> CorticalMesh -> Hemisphere
        │
        ▼
BufferGeometry colour attribute -> GPU
```

Four concerns kept apart: **data** (`lib/api.ts`), **rendering**
(`components/brain/`), **playback** (`hooks/usePlaybackSync.ts`), and **UI**.

---

## Backend

`src/blackmirror/api/` — three modules:

| Module | Role |
|---|---|
| `contracts.py` | JSON-safe view models. No numeric arrays. |
| `loader.py` | Reads artifacts via `ArtifactStore`; validates alignment. |
| `app.py` | HTTP routes. |

### Endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | Completed runs, newest first |
| `GET /api/runs/{id}` | `RunSummary` — metadata, scale statistics, array descriptors |
| `GET /api/runs/{id}/predictions` | `[T, V]` float32, raw |
| `GET /api/runs/{id}/timeline` | `[T]` float32 — `stimulus_time_seconds` |
| `GET /api/runs/{id}/mesh` | Mesh manifest |
| `GET /api/runs/{id}/mesh/{hemi}/{surface}/vertices` | `[N, 3]` float32 |
| `GET /api/runs/{id}/mesh/{hemi}/faces` | `[F, 3]` int32, hemisphere-local |
| `GET /api/runs/{id}/mesh/medial-wall` | `[V]` uint8 |
| `GET /api/runs/{id}/stimulus` | The original media |

Run it with:

```bash
uvicorn blackmirror.api.app:app --port 8000
```

### Transport: binary, not JSON

Arrays are served as raw little-endian binary with `X-Array-Dtype` and
`X-Array-Shape` headers (CORS-exposed so the browser can read them).

Measured on the reference run: predictions are **320 KB** binary versus **~960 KB**
as JSON — 3× smaller, and it lands directly in a `Float32Array` with no parse
step. Whole-run payload is **1.06 MB** (mesh 740 KB + predictions 320 KB).

**Tradeoff.** At this size everything is preloaded, so playback never stalls and
scrubbing is instant. The design scales linearly: a 10-minute stimulus would be
~49 MB of predictions, at which point per-timestep or chunked fetching becomes
worthwhile. The endpoints are already shaped for that — `predictions` would gain
a range parameter — but adding it now would be unmeasured complexity.

---

## Cortical mesh and vertex mapping

**Surface:** fsaverage5 (FreeSurfer, via nilearn), exported by Phase 1's
`blackmirror export-mesh`. Not a decorative model — the same standard surface
TRIBE predicts onto.

| Property | Value |
|---|---|
| Vertices per hemisphere | 10,242 |
| Total | 20,484 |
| Faces per hemisphere | 20,480 |
| Left hemisphere | prediction columns `[0, 10242)` |
| Right hemisphere | prediction columns `[10242, 20484)` |
| Medial wall | 1,769 vertices (8.6%) |

**The mapping rule.** Prediction column `j` belongs to the hemisphere whose
half-open range contains `j`, at local index `j − range_start`.

### Alignment is verified, not assumed

No single check proves the mapping. `tests/unit/test_mesh_alignment.py` uses
three complementary ones, and it matters to be precise about what each does and
does not establish.

**1. Byte-identity of the exported mesh (proof by construction).**
Our surface comes from `nilearn.datasets.load_fsaverage("fsaverage5")` — the
*same call* TRIBE's own plotting makes (`tribev2/plotting/cortical.py:182`).
The export is asserted byte-identical to a fresh load, so vertex **ordering** is
preserved end to end. This is not statistical. It does not by itself prove the
model indexes its output the same way; that comes from upstream source, cited in
`tribe_integration.md`.

**2. Spatial smoothness (detects reordering).**
BOLD varies smoothly across cortex, so edge-connected vertices should be more
similar than random pairs. Measured on the real run:

| Mapping | Smoothness ratio |
|---|---|
| **Correct** | **2.99×** |
| Rotated by 1 | 1.83× |
| Rotated by 101 | 1.21× |
| Reversed | 1.15× |
| Shuffled | 1.00× |

**3. Medial-wall variance (detects hemisphere assignment).**
Smoothness has a genuine blind spot: **it cannot detect a hemisphere swap**.
Swapping LH and RH scored **3.20×** — *higher* than correct — because each
hemisphere is internally smooth either way. The medial-wall mask closes that
gap. It comes from an anatomical atlas rather than the model, and the LH/RH wall
index sets overlap by only ~3%, so:

| Assignment | Wall/cortex variance ratio |
|---|---|
| **Correct** | **0.51×** (trimodal run) / 0.40× (reduced run) |
| Hemispheres swapped | 1.25× / 0.80× |

Together these pin ordering, orientation and hemisphere assignment. The
smoothness blind spot is itself asserted by a test, so if smoothness ever
becomes swap-sensitive this documentation gets revisited rather than quietly
drifting out of date.

### A trap worth knowing

**Inflated hemispheres are both centred on x ≈ 0** and geometrically overlap.
The renderer applies a ±48 mm lateral offset (`INFLATED_SEPARATION_MM`). Pial
coordinates are already lateralised and get no offset. Pinned by a test.

### Failing safely

If the prediction vertex count does not match the mesh, the API returns **409**
and the UI shows an error. It never renders a misaligned brain — a plausible but
wrong picture is the worst possible failure for a scientific tool.

Mesh lookups follow the **run's declared** `surface_space`, never a hard-coded
default. A run predicting onto a space we have not exported fails with a clear
error naming that space, rather than being silently described with fsaverage5
geometry.

---

## Temporal mapping

**Never `index = time / TR`.** TRIBE drops event-free segments; the reference run
kept **4 rows of 100** (96% dropped). The timeline is sparse and can be gappy.

`mapTimeToPredictionIndex(seconds, timeline)` binary-searches the per-row
`stimulus_time_seconds` array and returns the **nearest** row, clamped to
`[0, T-1]` — never `-1`, never `T`.

Because Phase 1 established `output_is_stimulus_aligned = true` (TRIBE
compensated the 5 s hemodynamic lag during training by shifting the fMRI
recording to `start − offset`), a row's segment time *is* the stimulus content
time. **No further shift is applied here.**

### Coverage is interval membership, not proximity

"Which row is nearest" and "which row covers this moment" are different
questions, and only the second is honest. Row `i` covers
`[timeline[i], timeline[i] + durations[i])` — half-open, so a moment exactly on
the next row's start belongs to that next row.

Proximity is misleading in a specific way: a moment 0.4 s *before* a row is near
it but is not described by it. With a 96% segment drop rate those gaps are
common, so the API serves `segment_duration_seconds` alongside the timeline and
`coveringPredictionIndex()` does true interval membership.

When nothing covers the current moment the cortex is drawn **inert grey** and
the UI says so explicitly. It does not show the nearest row's values, which
describe a different part of the stimulus.

---

## Normalization

Display-only. **The stored prediction array is never modified**, and the active
mode is always shown in the legend.

| Mode | Range | Use |
|---|---|---|
| **Robust global** (default) | ±max(\|p01\|, \|p99\|) | Outlier-resistant, comparable across time |
| Global | ±max(\|min\|, \|max\|) | True extremes |
| Per-frame | ±max of current row | Max contrast; **not comparable between timesteps** |

Robust global is the default because a fixed range is what makes watching the
response evolve meaningful — with per-frame scaling every frame looks equally
intense and change becomes invisible.

Percentiles are reused from Phase 1's `validation` block rather than recomputed,
so the API and the CLI report the same numbers.

---

## Colour mapping

**47.6%** of the reference run's values are negative, so the default is a
**zero-centred diverging** scale (cool below baseline → neutral at zero → warm
above). The API computes `diverging_recommended` from the data rather than
hard-coding it.

Both arms carry comparable lightness so neither sign reads as "stronger".
**Negative does not mean bad**, and the UI never implies otherwise.

Three colours sit deliberately **outside** the response scale, so they can never
be read as a response value:

| Sentinel | Meaning |
|---|---|
| Inert grey (`MEDIAL_WALL_COLOR`) | Medial wall — the model emits values, but fMRI signal is not meaningful |
| **Magenta** (`INVALID_VALUE_COLOR`) | The prediction is NaN or infinite |
| Dark grey (`NO_COVERAGE_COLOR`) | No prediction row covers this moment |

The magenta case matters more than it looks. A non-finite value normalised to
the midpoint of a zero-centred scale renders **identically to a genuine zero
response** — broken data that looks like ordinary data. `fillVertexColors`
screens for finiteness before colouring, returns a count of affected vertices,
and the UI reports it. The tooltip shows "no valid value" rather than a number.

Phase 1 validation reports 0 NaN/Inf on every real run to date, so this path is
covered by unit tests rather than observed in the browser.

A 256-entry LUT is precomputed once; per-vertex colouring is an array index, not
an interpolation.

---

## Rendering strategy

Geometry is built **once** per hemisphere as a `BufferGeometry`
(positions, indices, per-vertex colours, computed normals). A timestep change
rewrites only the colour attribute in place and sets `needsUpdate = true` — one
~123 KB upload, no mesh rebuild.

**React never touches the 61,452 colour floats.** They live in a ref, are written
by a plain loop in `fillVertexColors`, and go straight to the GPU. Putting them
in React state would reconcile them through the virtual DOM every frame.

The current approach is CPU-side colouring with a GPU upload. At 20,484 vertices
that measures well under a millisecond. A shader-based path (upload the response
as a texture, colour-map in GLSL) is the next step if vertex counts grow by an
order of magnitude — the seam is `fillVertexColors` plus the material.

---

## Reusability: A/B and delta maps

`BrainViewer` takes `values: Float32Array` and knows nothing about experiments
or TRIBE. Phase 5 mounts two instances sharing one `currentTime`:

```tsx
<BrainViewer values={variantA.predictions} … />
<BrainViewer values={variantB.predictions} … />
```

A difference map `ΔR = R_B − R_A` is just another `values` array. Because the
scale is already diverging and zero-centred, a delta map renders correctly with
no changes — zero difference sits at the neutral midpoint, which is exactly the
right default. `ResponseScale` is passed in, so both viewers can share one scale.

---

## Scientific considerations

- Language is constrained throughout: *"Predicted Cortical Response"*, never
  "brain activity" unqualified, never "brain scan".
- Every result carries Phase 1's `interpretation_notice`, surfaced in the
  "What am I seeing?" panel.
- Synthetic (mock backend) runs are badged **Synthetic test data** in the run
  list and on the experiment page.
- **No anatomical regions are named.** Phase 1 ships no atlas; inventing a region
  label would be fabrication. The tooltip shows only vertex index, hemisphere,
  raw value and time. Phase 3 adds ROI metadata.
- Peak markers are labelled **"Large response change"**, never "emotional moment".
- No temporal interpolation. With 3–4 samples the cortex changes in visible
  steps; smoothing between them would invent data. Timeline ticks show exactly
  where real samples are.

---

## Known limitations

1. **Very short timelines.** Real runs have 3–4 samples, so playback is a few
   discrete steps. That is a property of the CPU-bound Phase 1 runs, not the
   renderer.
2. **No temporal interpolation** — deliberate (see above), but it means playback
   looks stepped.
3. **Hover uses `face.a`**, the first vertex of the hit triangle, rather than the
   nearest vertex to the cursor. Off by at most one triangle edge.
7. **Model-internal indexing is taken on trust.** The three alignment checks
   constrain ordering, orientation and hemisphere assignment, but the claim that
   TRIBE emits `[LH, RH]` in fsaverage5 order ultimately rests on upstream
   source (`tribev2/plotting/cortical.py`), not on anything measurable here.
4. **Snapshot export is not implemented.** The canvas is created with
   `preserveDrawingBuffer: true` so it is a small addition.
5. **Single surface at a time.** Switching inflated/pial refetches geometry.
6. **No pial/inflated morph animation.**

---

## Testing

**Backend (28 new tests):**
`tests/unit/test_mesh_alignment.py` (14) — geometry invariants, spatial-smoothness
alignment proof, shuffled control, medial-wall corroboration, inflated-overlap trap.
`tests/unit/test_visualization_api.py` (14) — contract serialization, binary
transport byte-fidelity, sparse-timeline correctness, 404/409 failure modes,
"no bulk arrays in JSON".

**Frontend (54 tests, Vitest):**
`temporalMapping.test.ts` (16) — clamping, nearest-sample, gappy timelines,
round-trips, agreement with a linear scan over 500 random queries.
`normalization.test.ts` (22) — mode ranges, clamping, order preservation,
input immutability, medial-wall exclusion, peak detection.
`colorMapping.test.ts` (16) — ramp bounds, diverging neutrality, LUT fidelity,
and the `fillVertexColors` hot path including hemisphere offset indexing.

---

## Running it

```bash
# terminal 1 — visualization API
uvicorn blackmirror.api.app:app --port 8000

# terminal 2 — frontend
cd frontend && npm run dev
```

Then open <http://localhost:3000>. Press `d` for the developer overlay
(FPS, prediction index, frame min/max, buffer update time, alignment check);
space toggles playback.

---

## Measured performance

Apple M5, 16 GB, Chrome (ANGLE Metal renderer), 1680×1050 @ DPR 2, run
`20260903T160907Z-63dfdbb8` (real Sintel trailer clip, 4 × 20,484).

| Metric | Value |
|---|---|
| Mesh vertices rendered | 20,484 (2 × 10,242) |
| Triangles | 40,960 |
| Prediction samples | 4 |
| Prediction payload | 320 KB binary (≈960 KB as JSON) |
| Total transfer | 1.06 MB (mesh 740 KB + predictions 320 KB + metadata 4 KB) |
| API response time | 42 ms sequential for all buffers |
| **Initial page load to rendered cortex** | **≈3.3 s** |
| **Average FPS during playback** | **60.2** |
| Colour buffer fill, one hemisphere | mean 0.084 ms, p95 0.087 ms, p99 0.091 ms |
| **Both hemispheres per timestep** | **0.169 ms** (~1% of a 16.7 ms frame budget) |
| In-browser reported buffer update | 0.10 ms |
| Colour buffer per hemisphere | 123 KB |

The colouring cost is ~1% of the frame budget, so rendering is GPU-bound rather
than limited by the CPU-side colour computation. That is the headroom that makes
a shader-based path unnecessary at this vertex count.

### Visual validation

`frontend/scripts/validate-visualization.mjs` drives a real browser against a
real run and asserts what unit tests cannot:

| Check | Result |
|---|---|
| WebGL painted cortex | ✓ (non-background pixels, 15 distinct colours sampled) |
| Both hemispheres visible, neither occluded | ✓ left 55,734 px / right 54,833 px |
| Changing timestep repaints the surface | ✓ 142,750 pixels changed between t=0 and t=3 |
| Prediction vertices == mesh vertices | ✓ 20,484 == 20,484 (in-browser debug overlay) |
| Frame min/max match the source data | ✓ −1.0167 / +0.6680 at t=0 |
| Console errors / failed requests | ✓ none |
| Average FPS | ✓ 60.2 |

It exits non-zero on failure, so it can gate a release.

### A bug this caught

The first browser run rendered only one hemisphere in **Both** view. The camera
preset placed the camera on the **X** axis while the inflated hemispheres are
separated along **X** — so the near hemisphere completely occluded the far one,
and the picture looked entirely plausible.

Fixed with view-aware camera presets (`cameraPosition(preset, paired)`): paired
views use off-axis positions, and the default for both hemispheres is now the
dorsal view. Coverage is now balanced left/right, which the validator asserts.

This is exactly the class of error the project's rule about scientific
correctness exists for: nothing crashed, no test failed, and half the data was
silently invisible.

---

## Correctness review — six issues found and fixed

A post-implementation review surfaced six defects. All are fixed and regression-tested.

| # | Issue | Fix | Test |
|---|---|---|---|
| 1 | Mesh descriptors defaulted to `fsaverage5` instead of the run's declared space | Thread `result.cortical.surface_space` through every mesh lookup in `build_summary` | `test_mesh_descriptors_use_the_runs_declared_space`, `test_missing_mesh_for_the_declared_space_is_refused` |
| 2 | Coverage used nearest-row distance instead of `[start, start + duration)` | New `coveringPredictionIndex()`; API now serves `segment_duration_seconds` | 9 tests incl. `does NOT count a moment just before a row as covered` |
| 3 | UI showed the nearest prediction during a genuine gap | Cortex drawn inert with an explicit banner; `useIsCovered()` derives from interval membership | Browser-verified: covered `yes` at 1.5 s, `NO — inert surface` at 4.0 s |
| 4 | NaN/Inf rendered at the neutral midpoint, indistinguishable from a real zero | Screened before colouring, painted magenta, counted and surfaced | `paints NaN and Infinity distinctly, never at the neutral midpoint` |
| 5 | Segment-count log capture could cross-talk between concurrent predictions | Per-capture thread affinity + reference-counted logger mutation under a lock | `test_concurrent_captures_do_not_cross_talk` (verified to fail without the fix) |
| 6 | Smoothness was treated as proof of exact vertex identity | Added byte-identity proof and a hemisphere-swap detector; documented the blind spot | `test_export_matches_nilearn_exactly`, `test_medial_wall_detects_hemisphere_assignment` |

Two further defects were found while verifying the above:

- **Camera occlusion.** In *Both* view the camera sat on the X axis while
  hemispheres are separated along X, so the near one completely hid the far one.
  Fixed with view-aware presets; the validator now asserts balanced left/right
  pixel coverage.
- **Silent scrubber failure.** `setPointerCapture` can throw for a pointer id the
  element does not own, which aborted the handler before the seek ran. Now
  guarded, with `pointercancel` handled.

Issue 5 is the one worth dwelling on: the test was confirmed to **fail** with the
fix reverted. A test for a concurrency bug that has never been observed to fail
is not evidence of anything.
