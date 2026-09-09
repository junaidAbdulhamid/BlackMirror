# BlackMirror

**In-silico neural content experimentation.** BlackMirror runs multimodal
content — video, audio, or text — through Meta FAIR's TRIBE v2 brain-encoding
model and returns **predicted cortical fMRI responses** in a standardized,
reproducible form.

> These are **model-predicted** cortical responses for an average subject. They
> are not measurements from any real viewer, and they do not establish emotion,
> memory, attention, preference or purchase intent.
> **Read [`docs/scientific_limitations.md`](docs/scientific_limitations.md).**

---

## What BlackMirror is

The long-term goal is neural A/B testing: upload two versions of a piece of
content, run both through an identical inference pipeline, and compare the
predicted cortical responses rather than only clicks or surveys.

**Phase 1 — the current phase — builds none of that.** It builds the foundation
everything else needs: a reliable, reproducible, modular inference layer that
turns a content file into a validated `PredictionResult` plus a persistent
artifact trail.

The rest of BlackMirror never needs to understand TRIBE's internals. Phase 1 is
the abstraction boundary.

## Current status

**Phase 1 ✓ TRIBE Core** — model loading, stimulus validation, preprocessing,
inference, prediction validation, cortical mesh export, artifact persistence,
provenance, CLI, tests.

**Phase 2 ✓ Interactive Cortical Visualization** — read-only FastAPI over Phase 1
artifacts, plus a Next.js + React Three Fiber cortical renderer with stimulus
playback synchronised to the predicted response timeline.
See [`docs/phase2_visualization.md`](docs/phase2_visualization.md).

**Phase 3 ✓ Neural Analytics** — Destrieux atlas mapping, ROI time-series,
whole-cortex metrics, and non-semantic neural event detection.
See [`docs/phase3_neural_analytics.md`](docs/phase3_neural_analytics.md).

**Phase 4 ✓ Multimodal Content Intelligence** — shot/scene segmentation, CLIP
visual semantics, OCR, audio signals, transcript reuse, multimodal event fusion,
and non-causal alignment of content to predicted neural events.
See [`docs/phase4_content_intelligence.md`](docs/phase4_content_intelligence.md).

**Phase 5 ✓ Neural A/B Comparison** — comparability gating, temporal alignment
across variants, per-vertex and per-ROI contrasts with multiple-comparison
correction. See [`docs/phase5_neural_comparison.md`](docs/phase5_neural_comparison.md).

**Phase 6 ✓ Goal-Conditioned Scoring** — explicitly declared objectives over
predicted responses, a metric registry, per-sample decomposition, effective
weights, and normalization that never hides a negligible raw difference.
See [`docs/phase6_goal_scoring.md`](docs/phase6_goal_scoring.md).

**Phase 7 ✓ Neural Content Optimization Agent** — locates where a variant
underperforms its objective, measures how higher-scoring variants differ there,
and proposes candidate interventions. Every output is a hypothesis, and a
recommendation becomes a candidate only after a recorded human decision.
See [`docs/phase7_optimization_agent.md`](docs/phase7_optimization_agent.md).

**Phase 10 ✓ Surrogate Modelling and Bayesian Optimization** — a cheap learned
approximation of the expensive objective, used to screen thousands of candidate
variants and choose which few deserve a real evaluation. Gaussian Process and
tree-ensemble surrogates, expected improvement and upper confidence bound, a
trust gate that refuses to let an unproven model spend budget, and predicted
against actual recorded every round. Measured on synthetic surfaces: **about
half the evaluations** to reach the same score as random search.
See [`docs/phase10_surrogate_optimization.md`](docs/phase10_surrogate_optimization.md),
[`docs/bayesian_optimization.md`](docs/bayesian_optimization.md) and
[`docs/surrogate_evaluation.md`](docs/surrogate_evaluation.md).

**Phase 9 ✓ Automated Neural Search** — a budgeted, sample-efficient search over
content variants: a declared parameter space, deterministic candidate
materialization, seven interchangeable strategies, guardrails that veto an
inadmissible winner, Pareto search across objectives, a persistent evaluation
cache, checkpointing and resume. Reports the best *observed* candidate and
whether its lead clears the pipeline's measured noise floor.
See [`docs/phase9_neural_search.md`](docs/phase9_neural_search.md),
[`docs/search_algorithms.md`](docs/search_algorithms.md) and
[`docs/search_benchmarks.md`](docs/search_benchmarks.md).

**Phase 8 ✓ Bounded Re-Simulation** — binds an approved candidate to media a
person built and measures it: a durable, resumable state machine through
inference, analytics, content, scoring and comparison, with direction-aware
outcomes and separate hypothesis verdicts. The only part of the product that
runs the model on demand.
See [`docs/phase8_resimulation_loop.md`](docs/phase8_resimulation_loop.md).

## Architecture

```
       Stimulus  (video / audio / text)
          │
          ▼
      Validation  ── SHA-256, media type, size, readability
          │
          ▼
    Preprocessing  ── audio extraction, transcription, event table
          │
          ▼
       TRIBE v2    ── frozen Llama-3.2 + V-JEPA2 + W2V-BERT + DINOv2
          │           into a trained cortical head
          ▼
  Predicted cortical responses   [T time points x 20,484 fsaverage5 vertices]
          │
          ▼
      Validation  ── shape, dtype, NaN/Inf, distribution, temporal variance
          │
          ▼
       Artifacts  ── raw matrix, temporal alignment, provenance, manifest
```

Phase 2 consumes those artifacts without ever running the model:

```
Saved artifacts
     │
     ▼
Visualization API  (FastAPI, read-only)
     │  JSON metadata + raw binary arrays
     ▼
Next.js + React Three Fiber
     │
     ▼
fsaverage5 cortical surface, coloured per vertex
```

Full detail: [`docs/architecture.md`](docs/architecture.md) and
[`docs/phase2_visualization.md`](docs/phase2_visualization.md).

## Installation

Requires **Python 3.11+** (3.12 recommended), `ffmpeg`, and `uv`.

```bash
# 1. Toolchain (macOS)
brew install python@3.12 ffmpeg uv

# 2. Virtual environment
uv venv --python python3.12 .venv
source .venv/bin/activate

# 3. BlackMirror core (no ML stack — the contract layer alone)
uv pip install -e ".[dev,viz]"

# 4. TRIBE v2 itself (CC-BY-NC-4.0, NON-COMMERCIAL)
uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

> **Do not `pip install whisperx`.** TRIBE runs transcription via
> `uvx whisperx` in its own ephemeral environment. Installing it here drags
> `torch` to 2.8, outside TRIBE's supported `>=2.5.1,<2.7` range.

### Configuration

```bash
cp .env.example .env
```

## Model setup

The TRIBE head downloads automatically on first use (~709 MB). Pre-fetch it:

```bash
python scripts/download_model.py
```

### HuggingFace access (required for the text stream)

TRIBE's text encoder is the **gated** `meta-llama/Llama-3.2-3B`:

1. Accept the license at <https://huggingface.co/meta-llama/Llama-3.2-3B>
2. Create a read token at <https://huggingface.co/settings/tokens>
3. Put it in `.env` as `HF_TOKEN=...` (or run `huggingface-cli login`)

Content with **no speech** never reaches the text encoder — TRIBE drops the text
extractor when there are no `Word` events — so silent clips run without this.
Anything with dialogue needs it.

If you cannot use the gated repo, `BLACKMIRROR_TEXT_ENCODER_ID` repoints the
frozen text encoder at a repo you can read:

```bash
BLACKMIRROR_TEXT_ENCODER_ID=unsloth/Llama-3.2-3B blackmirror predict clip.mp4
```

It must be the **same model** — the trained head expects Llama-3.2-3B's hidden
dimensions. The override is recorded in the model fingerprint, so runs using
different text encoders are treated as non-comparable. It does not grant you a
license: Llama 3.2 remains under Meta's terms whatever repo the weights come from.

### GPU requirements

| Environment | Behaviour |
|---|---|
| **CUDA** | Supported and strongly recommended |
| **Apple Silicon / MPS** | **Not supported by TRIBE.** `device="auto"` resolves to CPU; `--device mps` is refused rather than half-honoured |
| **CPU** | Works, but feature extraction (V-JEPA2 ViT-g, Llama-3.2-3B) dominates — budget tens of minutes for a ~10 s clip, and ~16 GB RAM |

The frozen extractors total well over 10 GB of downloads on first use.

## Verify the environment

```bash
blackmirror verify-env
```

Prints a PASS/WARN/FAIL table with a concrete remedy for every failure — Python
version, torch range, ffmpeg, `uvx`, gated HuggingFace access, checkpoint cache,
writable directories.

## Running inference

`examples/media/` is gitignored — generate a test clip first
(see [`examples/README.md`](examples/README.md)):

```bash
mkdir -p examples/media
ffmpeg -y -f lavfi -i "testsrc2=size=320x240:rate=24:duration=20" \
       -f lavfi -i "sine=frequency=440:duration=20" \
       -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
       examples/media/synthetic_20s.mp4
```

```bash
# Real prediction
blackmirror predict examples/media/synthetic_20s.mp4

# Synthetic run — no model needed; proves the pipeline end to end
blackmirror predict --backend mock examples/media/synthetic_20s.mp4

blackmirror inspect-model            # model provenance
blackmirror list-runs                # stored runs
blackmirror inspect-run <run_id>     # summary for one run
blackmirror export-mesh              # fsaverage5 vertices/faces/medial wall
```

Programmatically — this is the API a future FastAPI layer will call:

```python
from blackmirror import InferenceService

service = InferenceService()
result = service.predict("examples/media/synthetic_20s.mp4")

print(result.prediction.shape)                  # (T, 20484)
print(result.temporal.timeline_is_contiguous)
print(result.artifacts.run_dir)

# Many variants, one loaded model — the basis for Phase 5
results = service.predict_many(["a.mp4", "b.mp4"])
```

## Output structure

```
artifacts/runs/<run_id>/
├── manifest.json              # the full PredictionResult contract
├── predictions.npy            # [T, V] raw model output — IMMUTABLE
├── temporal.npz               # per-row segment times + derived stimulus times
├── events.parquet             # stimulus event table
├── stimulus.json  prediction_metadata.json  validation.json
├── performance.json  provenance.json  events_summary.json
└── diagnostics/global_mean_timecourse.png
```

Source media is **never copied** — it is referenced by path, filename and
SHA-256.

> **The prediction time axis is not necessarily uniform.** TRIBE drops
> event-free segments, so row `i` is *not* time `i × TR`. Always read
> `segment_start_seconds` from `temporal.npz`. Full contract:
> [`docs/data_contracts.md`](docs/data_contracts.md).

## Visualization (Phase 2)

```bash
# terminal 1 — read-only API over saved runs
uvicorn blackmirror.api.app:app --port 8000

# terminal 2 — frontend
cd frontend && npm install && npm run dev
```

Open <http://localhost:3000>, pick a run, and the cortex renders beside the
stimulus on a shared timeline. Press `d` for the developer overlay (FPS,
prediction index, frame min/max, buffer update time, alignment check).

The frontend **never runs inference** — it reads artifacts a completed run
already wrote.

## Content intelligence (Phase 4)

```bash
blackmirror analyze <run_id>           # Phase 3 neural analytics (needed for associations)
blackmirror analyze-content <run_id>   # Phase 4 content analysis
```

Results are cached on a key covering the stimulus hash, every model id, the
analysis version and the configuration — measured **16.8 s cold, 0.42 s cached**.
The API never triggers the pipeline; an un-analysed run 404s with instructions.

## Scoring, optimization and re-simulation (Phases 6–8)

```bash
# Phase 8 — measure a variant you built against an approved hypothesis.
blackmirror bind-resimulation loop-1 \
    --experiment exp --optimization <key> --candidate cand-1 \
    --media path/to/variant.mp4 --out request.json
blackmirror resimulate request.json     # runs inline; a 10 s variant took ~2 h
blackmirror list-resimulations
blackmirror resume-resimulation loop-1  # verifies checksums, continues
blackmirror stop-resimulation loop-1    # observed at the next stage boundary

# Phase 9 — automated search over a declared space.
blackmirror list-searches
blackmirror search-trajectory <search_id>   # the sample-efficiency curve
blackmirror search-report <search_id>
python scripts/benchmark_search_strategies.py   # synthetic, free, seconds
python scripts/benchmark_search_pipeline.py     # real, hours

# Phase 10 — surrogate-assisted optimization.
python scripts/benchmark_surrogate.py           # random vs beam vs bayesian
python scripts/measure_space_range.py           # does a space clear the noise floor?
```

The surrogate never replaces the real evaluator. It ranks candidates cheaply so
the expensive evaluations go where they are most likely to be worth spending,
and every number it produces is labelled an estimate until the real pipeline has
measured it. A surrogate that cannot demonstrate rank skill on held-out data is
not permitted to choose anything.

A search evaluates candidates through the full Phase 8 loop, so its cost is
measured in evaluations rather than seconds. It reports the strongest candidate
it **observed**, never an optimum, and states whether that lead is larger than
the pipeline's own sensitivity to choices that should not matter.

The request is *derived*, not typed: the objective set, both of its hashes and
the per-hypothesis direction map are read from stored Phase 6 and Phase 7
records, so a pass cannot end up measuring an objective nobody approved. The
same flow is available at `/scoring`, `/optimize` and `/resimulation` in the UI.

Phase 8 is the one part of the product that runs the model on demand, and the
only route family in the API that can reach one. It never edits media: it reads
the file where it lies, records its hash, and refuses a variant whose bytes
match the source.

## Testing

```bash
pytest                      # unit suite: no checkpoint, no GPU, no network
pytest -m tribe_integration # opt-in; needs the real model

cd frontend && npm test     # temporal mapping, normalization, colour mapping
```

## Scientific limitations

Non-negotiable reading before interpreting any output:
[`docs/scientific_limitations.md`](docs/scientific_limitations.md). In short —
predictions are model output for an *average* subject, BOLD is not neural
firing, activation implies no mental state, and none of this is for medical use.

## Licensing

BlackMirror's own code is MIT. **TRIBE v2 is CC-BY-NC-4.0 — non-commercial use
only** — which constrains any commercial deployment built on it. See
[`docs/tribe_integration.md`](docs/tribe_integration.md).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | **TRIBE Core** — inference foundation, contracts, artifacts | ✓ |
| 2 | **Interactive Cortical Visualization** — 3D cortex, playback sync | ✓ |
| 3 | **Neural Analytics** — atlas ROIs, temporal events | ✓ |
| 4 | **Multimodal Content Intelligence** — content events, neural association | ✓ |
| 5 | **Neural A/B testing** — comparability gating, aligned contrasts | ✓ |
| 6 | **Goal-conditioned scoring** — declared objectives, decomposition | ✓ |
| 7 | **Optimization agent** — evidence-backed candidate interventions | ✓ |
| 8 | **Re-simulation loop** — measure an approved candidate | ✓ |
| 9 | **A/B/N search** — budgeted, sample-efficient optimization | ✓ |
| 10 | **Surrogate optimization** — learned screening, Bayesian search | ✓ |
| 10 | Production infrastructure | |
