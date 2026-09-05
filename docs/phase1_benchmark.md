# Phase 1 Performance Baseline

Measured on the development machine. These are the numbers later optimization
must beat, and the basis for judging whether a Phase 5 variant sweep is
affordable.

Records are machine-readable in `artifacts/benchmarks/<run_id>.json`; regenerate
any of them with `blackmirror benchmark <run_id>`.

## Machine

| | |
|---|---|
| Hardware | Apple M5, 8-core GPU, **16 GB unified memory** |
| OS | macOS 26.5.1 (arm64) |
| Python | 3.12.14 |
| torch | 2.6.0 (**CPU** — TRIBE v2 has no MPS path) |
| Device used | `cpu` |

## Full trimodal run — real content

Stimulus: `sintel_speech_4s.mp4` — 4.0 s at 854×480, cut from the **Sintel
trailer** (Blender Foundation, CC-BY 3.0) at 41.5 s, the densest speech window
("I've been alone for as long as I can remember"). All three modalities active:
10 `Word` events + `Sentence` + `Text` + `Audio` + `Video`.

| Stage | Cold cache | Warm cache |
|---|---|---|
| Preprocessing (incl. WhisperX) | 31.11 s | 0.4 s |
| **Inference** (incl. all 4 encoders) | **1553.56 s** | **48.06 s** |
| **Total** | **~1800 s** (30 min) | **56.66 s** |

Output `(4, 20484)` float32; mean −0.016, std 0.213, 0 NaN, 0 Inf.
Segments: **4 kept of 100** (96% dropped).

Text features ran through `unsloth/Llama-3.2-3B` via
`BLACKMIRROR_TEXT_ENCODER_ID` (see `docs/tribe_integration.md`); the encoder
download added ~6 GB, bringing the HuggingFace cache to 16 GB.

Cost breakdown for the cold run: WhisperX 31 s, Llama-3.2-3B download + text
features ~110 s, audio ~10 s, **V-JEPA2 video ~1320 s (8 clips)**, DINOv2 +
head the remainder. Video encoding is ~85% of the total.

## Reduced-modality runs

Stimulus: `synthetic_2s.mp4` — 2.08 s, 320×240, 24 fps, 93.6 kB, generated with
ffmpeg (`testsrc2` + 440 Hz sine). Transcription disabled, so
audio + video only, no text stream.

| Stage | Cold cache | Warm cache |
|---|---|---|
| Preprocessing | 0.37 s | 0.37 s |
| Model load | 6.87 s | 6.63 s |
| **Inference** (incl. feature extraction) | **688.41 s** | **28.97 s** |
| Postprocessing | 0.00 s | 0.00 s |
| Validation | 0.02 s | 0.02 s |
| Persistence | 0.22 s | 0.22 s |
| **Total** | **695.90 s** | **36.06 s** |
| Realtime factor | **334×** content duration | 17.3× |
| Peak host RSS | 5.72 GB | 1.05 GB |

Output: `(3, 20484)` float32 — 3 time points × 20,484 fsaverage5 vertices.
`predictions.npy` 246 kB; whole run directory 349 kB.

### What dominates

Almost everything is **frozen feature extraction**, not the TRIBE head (177 M
parameters, whose forward pass is negligible). V-JEPA2 ViT-g encodes one 4-second
clip every 0.5 s of video — about **150 s per clip on CPU** — so cost scales with
content duration, not resolution.

TRIBE caches extracted features via `exca` under `BLACKMIRROR_MODEL_CACHE_DIR`.
The **24× cold-to-warm speedup** is that cache: re-running the same stimulus
skips extraction entirely. This is why the cache directory is worth keeping, and
why Phase 5's repeated variant scoring is more affordable than the cold number
suggests — for *unchanged* content.

### Segment filtering, observed

Both runs pad to a 100-segment window and keep only the segments containing
events: **3 of 100** for the 2 s clip, **4 of 100** for the 4 s clip — a 96-97%
drop rate, now recorded in the contract as `n_segments_total`.

The survivors happened to be the leading segments, so both timelines are
contiguous — but that is luck, not a guarantee. It is the empirical
justification for persisting `segment_start_seconds` rather than computing time
as `row_index × TR`.

## Failed attempt: 8-second clip

Aborted after **5 h 43 m wall clock**, having completed 13 of 16 V-JEPA2 clips.

The instructive part is the CPU-time-to-wall-time ratio: in the final 83 minutes
of wall time the process accumulated only **5.7 minutes of CPU**. System swap was
at **27.0 GB of 27.6 GB**. The process was not computing — it was paging.

**Conclusion: on a 16 GB machine, CPU inference is viable only for stimuli of a
few seconds.** The limit is memory pressure, not raw compute. The extractor
download footprint alone exceeds 10 GB.

## Mock backend (contract layer only)

| Stimulus | Shape | Total | Notes |
|---|---|---|---|
| 20 s video | `(20, 20484)` | 0.05 s | no model, no torch |

The mock path is ~14,000× faster than cold TRIBE, which is what makes the unit
suite (94 tests, ~9 s) practical.

## Cortical mesh export

One-off, shared by every run, independent of the model:
`artifacts/mesh/fsaverage5/` — 7 arrays, 20,484 vertices, 20,480 faces per
hemisphere, 1,769 medial-wall vertices (8.6%).

## Implications

1. **A GPU is required for real work.** ~450× realtime on CPU for the trimodal
   path means a 30-second ad would take ~3.7 hours cold. Phase 5 A/B testing is
   impractical without CUDA.
2. **Feature extraction is the optimization target**, not the TRIBE head.
3. **The feature cache is load-bearing** — preserve it across runs.
4. **Memory, not FLOPs, is the binding constraint** on 16 GB.
5. Artifacts are cheap: ~350 kB per run at this length. Scaling linearly, a
   10-minute video is roughly 50 MB of predictions — large but manageable.

## Reproducing

```bash
blackmirror predict examples/media/synthetic_2s.mp4
blackmirror benchmark              # writes artifacts/benchmarks/<run_id>.json
```
