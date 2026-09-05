# Architecture

## The one idea

Phase 1 exists to create an **abstraction boundary**. Everything above the
boundary speaks a stable contract; everything below it is a swappable model.

```
                      StimulusInput
                            │
                            ▼
                     StimulusValidator
                            │
                            ▼
                    InferenceService  ◀── the public API
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  preprocess()          infer()        get_model_metadata()
        └───────────────────┼───────────────────┘
                            ▼
              CorticalPredictorBackend  (Protocol)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        TribeV2Backend               MockBackend
      (lazy `import tribev2`)    (labelled synthetic)
                            │
                            ▼
                     RawPrediction
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     PredictionValidator          Metadata builders
              └─────────────┬─────────────┘
                            ▼
                      ArtifactStore
                            │
                            ▼
                    PredictionResult
```

The success test: *if TRIBE v2 were replaced by another cortical prediction
model, most of BlackMirror should remain unchanged.* Concretely, only
`inference/tribe_backend.py` and `inference/tribe_loader.py` would be rewritten.

This is enforced mechanically —
`tests/unit/test_pipeline.py::test_pipeline_runs_without_importing_tribe`
asserts `tribev2` never enters `sys.modules` during a full run.

## Module responsibilities

| Module | Responsibility | Must NOT |
|---|---|---|
| `schemas/` | The data contracts | Import anything heavier than pydantic/numpy |
| `config/settings.py` | All configuration | Be bypassed by `os.environ` reads elsewhere |
| `errors.py` | Typed failures | — |
| `inference/backend.py` | The Protocol + transport types | Import any model |
| `inference/tribe_loader.py` | **Model lifecycle only** | Preprocess content or analyse results |
| `inference/tribe_backend.py` | TRIBE ⇄ contract translation | Leak TRIBE types upward |
| `inference/preprocessing.py` | Wrap TRIBE's event pipeline | Reimplement it |
| `inference/postprocessing.py` | Temporal + cortical metadata | Modify the prediction matrix |
| `inference/validation.py` | Stimulus + prediction checks | Repair data |
| `inference/mock_backend.py` | Synthetic backend | Ever be mistaken for real output |
| `inference/compat_macos.py` | Documented platform patches | Patch silently |
| `inference/service.py` | Orchestration | Know that TRIBE exists |
| `storage/artifact_store.py` | Persistence | Interpret content |
| `cortical/mesh.py` | Surface geometry | Depend on the model |
| `diagnostics.py` | Env checks + sanity plot | Be on the inference path |
| `cli.py` | Developer interface | Contain inference logic |

## Key design decisions

### 1. The backend is a Protocol, not a base class

`CorticalPredictorBackend` is a `typing.Protocol`. A new backend needs no
inheritance and no import of our internals — it just has to have the right
methods. This keeps the coupling structural rather than nominal.

### 2. Model lifecycle is independent of any stimulus

`TribeModelLoader` holds the model; `InferenceService.ensure_model_loaded()` is
idempotent. `predict_many()` scores a batch of variants against one loaded
model, which is what makes Phase 5's A/B/N comparison both affordable and
*fair* — every variant shares one model instance and one provenance record.

### 3. The mock backend is production code, not a fixture

It proves the abstraction, keeps the unit suite free of a 709 MB checkpoint and
gated model access, and unblocks Phase 2 against realistically-shaped output.
Its cost is one small module; its output is stamped `is_synthetic=True` at every
layer, including inside `manifest.json`.

### 4. Raw predictions are immutable

`predictions.npy` is written once, byte-for-byte. Derived quantities (the
`stimulus_time_seconds` column, diagnostic plots) go in separate files with
their derivation recorded. See `docs/scientific_limitations.md`.

### 5. Failures are typed and chained

Every stage wraps its exception in a specific error naming what failed, with
`raise ... from exc` preserving the cause. Failure never leaves a partial run
directory: artifacts are written only after inference and validation succeed,
and each file lands via temp-file + atomic rename.

### 6. No silent GPU → CPU fallback for real models

TRIBE v2 on CPU is orders of magnitude slower than on CUDA. A silent fallback
turns a 40-second job into hours and reads as a hang.
`BLACKMIRROR_ALLOW_CPU_FALLBACK=false` makes an unsatisfiable device request an
error. The device utility also knows which devices each backend *actually*
supports, so `--device mps` for TRIBE is refused rather than half-honoured.

### 7. Compatibility patches are recorded, never hidden

Platform patches live in one module, each documenting problem / fix /
scientific impact, and every applied patch is written into
`provenance.applied_compat_patches`. A patched run can never be mistaken for a
stock upstream run, and two runs are only comparable if their patch lists match.

### 8. Caching is a seam, not a system

The run cache key is
`sha256(stimulus_sha256 | model_fingerprint | preprocessing_config)`, stored in
an append-only `index.jsonl` and looked up by linear scan. Deliberately trivial:
the point is that `ArtifactStore.find_by_cache_key()` exists as a seam a real
cache can slot behind later.

## Ready for an API

The CLI holds no inference logic. A future FastAPI layer calls the same object:

```
CLI     ─┐
         ├─▶ InferenceService ─▶ CorticalPredictorBackend
FastAPI ─┘
```

`InferenceService` has no CLI state, no global mutable state, and returns a
pydantic model that is already a valid response body.

## Deliberately not built in Phase 1

3D cortex, ROI interpretation, emotion/memory/engagement scores, A/B comparison,
optimization agents, job queues, auth, frontend. Phase 1 stops at a reliable
prediction and a persisted, reproducible artifact.
