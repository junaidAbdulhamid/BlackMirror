# Phase 9 Integration Report

Written before any Phase 9 code, from an inspection of the Phase 1–8 source as
it actually stands rather than as the brief describes it. Where the two differ,
this report follows the code and says so.

---

## 1. Existing candidate architecture

There is no `CandidateVariant` type. The nearest equivalents are:

| Type | Module | What it is |
| --- | --- | --- |
| `ProposedVariantSpec` | `optimization/schemas.py` | A description of a variant **that does not exist yet** |
| `TransformationBinding` | `resimulation/schemas.py` | A content-addressed link between a source file and a variant file that **already exists** |
| `ResimulationResult` | `resimulation/schemas.py` | The durable record of evaluating one such variant |

The gap between those first two rows is the central fact of this report. A
`ProposedVariantSpec` carries interventions and edit instructions. A
`TransformationBinding` carries two SHA-256 hashes and refuses to accept a
variant whose bytes match the source. **Nothing in the codebase turns the first
into the second.** Phase 8's only registered adapter is
`UserSuppliedVariantAdapter`, whose docstring is explicit: "Authorized
transformation bindings; no media generation occurs implicitly."

That was the correct decision for Phase 8, where a human builds the variant and
the system measures it. It is a hard blocker for Phase 9, where the whole point
is that the system proposes and evaluates variants without a human in the loop
for each one.

### Absent from the code, present in the brief

The Phase 8 summary in the brief lists several things that were never built:

| Named in brief | Status |
| --- | --- |
| `CandidateVariant` | absent |
| `VariantLineage` | absent (lineage exists as fields, see §8) |
| `CandidateAcceptancePolicy` | absent |
| guardrails | absent — no occurrence of the word in `src/` |
| off-target analysis | absent |
| candidate promotion | absent |
| candidate materialization | absent |
| cost tracking | partial — wall-clock only, see §5 |

`SUPPORTED / REJECTED / INCONCLUSIVE` does exist, as
`HypothesisVerdict.PASS / FAIL / INCONCLUSIVE`. Resumability and idempotency
exist and are well covered by tests.

---

## 2. Existing search-compatible interfaces

These are genuinely reusable and Phase 9 should build directly on them.

**Evaluation.** `ResimulationOrchestrator` is already the evaluation function
Phase 9 needs. `prepare(request)` binds a run durably without executing;
`start(request)` runs the five stages; `resume(id)` continues after the last
durable stage. It is idempotent on restart and checkpoints after every stage,
which is exactly the property a resumable search requires.

**Concurrency and admission.** `ResimulationLoader` already wraps the
orchestrator with a single-worker executor, an in-flight job map, and a durable
per-run lock. Phase 9's `CandidateScheduler` should extend this rather than
introduce a second job system.

**Scoring.** `ScoringLoader.create(experiment_id, run_ids, objectives, ...)`
computes a full `ExperimentScoreResult` including ranking, contributions,
Pareto and weight sensitivity. It caches on a digest that already includes the
run set, so scoring N candidates does not overwrite earlier scores.

**Pareto.** `analyze_pareto` in `scoring/engine.py` implements strict dominance
over normalized per-objective scores where higher is better, returns
`non_dominated`, `dominated`, and `dominated_by`, and warns when a variant was
excluded for missing a score. This is directly usable for Step 58–60. It is
O(n²) in variants, which is irrelevant at the population sizes Phase 9 can
afford.

**Determinism.** Measured this session: two independent inference runs of the
same stimulus under the same configuration produced **bit-identical**
predictions. Reproducible search is therefore achievable; the randomness Phase 9
must seed is its own, not the model's.

---

## 3. Available intervention space

`InterventionType` has eleven members and `EditInstruction.operation` has seven:

    move_event, insert_speech, adjust_feature, reorder_events,
    extend_event, replace_text, adjust_pace

These are machine-readable and were designed for "Phase 8 to execute or a human
to apply". **Phase 8 executes none of them.** For Phase 9 they divide sharply:

| Operation | Mechanically realizable with ffmpeg? |
| --- | --- |
| `adjust_feature` (audio gain, brightness, contrast) | Yes, deterministically |
| `adjust_pace` (speed factor) | Yes, with `setpts` / `atempo` |
| `move_event`, `extend_event` (timing of an existing segment) | Yes, by cut and reassemble |
| `reorder_events` | Yes, by cut and reassemble |
| `insert_speech` | No — requires speech synthesis |
| `replace_text` | No — requires rendering or generation |

ffmpeg 9.0.1 is on PATH, and `content/media.py` already has the right shape for
this: a `_tool()` PATH check and a `_run()` subprocess wrapper with timeout and
error translation, used for `extract_audio` and `extract_frame`.

**Recommendation.** Phase 9's search space should cover only the top four rows.
The bottom two need generative models, which Step 95 explicitly defers. A search
space restricted to deterministic, parameterizable edits is also the more
defensible one scientifically, because every candidate differs from its parent
in a way that can be stated exactly.

### A schema friction point

`ContentIntervention.evidence_ids` has `min_length=1`, and every `EvidenceKind`
member records a *measured comparison between variants*:
`CONTENT_FEATURE_DELTA`, `OBJECTIVE_GAP`, `CONTRIBUTION_SHARE`,
`CONTENT_EVENT_PRESENCE`, `FEATURE_CONSISTENCY`, `REGIONAL_CONTRIBUTION`.

A search-generated candidate has no such measurement behind it. Its parameter
value came from a strategy sampling a declared space. Reusing a Phase 7
evidence kind to satisfy the schema would make a sampled guess look like a
grounded finding, which is precisely the confusion the evidence model exists to
prevent.

Phase 9 needs its own provenance path — a distinct evidence kind such as
`SEARCH_SPACE_SAMPLE` whose description states that the value was chosen by a
named strategy from a declared range, not measured. This must be decided before
`GenomeDecoder` is written.

---

## 4. Existing objective interface

`NeuralObjective` is complete and directly usable as Phase 9's fitness
definition:

- `metric` resolved through a registry (`scoring/metrics.py`), seven metrics
- `target`: whole cortex, ROI, or Yeo network
- `temporal_scope`: full stimulus, window, or content event
- `direction`: `MAXIMIZE`, `MINIMIZE`, `TARGET`
- `normalization` with an explicit `target_value` for `TARGET`
- `weight`, with `effective_weights()` reporting what the weights actually do

`ExperimentScoreResult` gives `variant_scores`, `ranking`, `pareto`,
`sensitivity` and `metadata.objective_definition_hash`. Phase 9's
`CandidateFitness` should wrap this rather than recompute anything.

Direction-aware outcome logic already exists in
`resimulation/evaluation.py::evaluate_objectives`, including the `TARGET` case
(reduced absolute distance from the declared target is an improvement) and a
tolerance band below which nothing is claimed. Step 45's target-based stop
should reuse this rather than reimplement the comparison.

---

## 5. Existing cost information

`PerformanceMetrics` records, per run: preprocessing, model load, inference,
postprocessing, validation, persistence and total seconds, plus peak GPU
allocated/reserved bytes, peak host RSS, and a `realtime_factor`.

There is **no** GPU-seconds accounting, no monetary cost, and no aggregation
across runs. `gpu_seconds`, `cost_usd` and `estimated_cost` do not appear
anywhere in `src/`.

On this machine that is not merely missing, it is not applicable: TRIBE v2 runs
on CPU, and `device="auto"` has no MPS path. A `SearchBudget` field named
`MAX_GPU_SECONDS` would be measuring something that does not happen. Phase 9
should track **wall-clock seconds** and name the field accordingly, and treat
monetary cost as an optional operator-supplied rate applied to wall-clock time,
clearly labelled as an estimate.

### Measured evaluation cost — the governing constraint

| Stimulus | First run, new bytes | Re-run, cached features |
| --- | --- | --- |
| sintel_speech_4s | 26.5 min | 2.2 min |
| sintel_10s | 57.4 min | 0.5 min |
| sintel_10s_vB | 45.8 min | 4.0 min |
| sintel_10s_vC | 52.8 min | 0.5 min |
| sintel_30s | 154.4 min | 2.3 min |

**A new candidate has new bytes and therefore pays the first-run column.** For a
10-second clip that is roughly **52 minutes of TRIBE inference**, plus 1–2
minutes of content analysis, per evaluation. Analytics is under a second and
scoring is milliseconds; they do not matter.

The consequences are severe and shape every design choice in Phase 9:

| Budget | Wall-clock |
| --- | --- |
| 5 evaluations | ~4.5 hours |
| 25 evaluations, the brief's example | ~22 hours |
| Random vs. beam benchmark, 10 each | ~18 hours |

Sample efficiency is not a feature of Phase 9. It is the only thing that makes
Phase 9 possible at all.

---

## 6. Existing caching

Four independent cache layers already exist:

1. **Feature extraction** (`exca` TaskInfra under `cache/`), keyed per extractor
   — text, audio, video separately. This is why re-running identical media is
   30x faster.
2. **Inference runs** — `build_cache_key(stimulus_sha256, model_fingerprint,
   preprocessing_fingerprint)` with `ArtifactStore.find_by_cache_key`. Gated by
   `reuse_cached_runs`, which **defaults to `False`**.
3. **Content analysis** — `compute_cache_key` over stimulus hash, every model
   id, analysis version and configuration.
4. **Scoring** — `objective_set_hash(objectives, cache_context={run_ids,
   baseline, with_networks, input_artifact_sha256})`.

Phase 8 adds a fifth: its own `cache_key` is a SHA-256 of the request excluding
`resimulation_id`, so the same request under a different id is recognized.

Step 38's `SearchEvaluationCache` should key on the genome hash and delegate
downward rather than duplicate any of this. Note that layer 2 is off by
default, which is the correct default for a research corpus but wrong for a
search: Phase 9 must enable it explicitly, or it will re-infer identical
candidates.

**Open question worth measuring first.** The feature cache is keyed per
extractor. If a candidate changes only the audio stream and the video stream is
copied bit-for-bit, the video extractor cache may still hit. Video features
(V-JEPA2 ViT-g) plausibly dominate the 52 minutes. If that cache hits, the cost
of an audio-only candidate could fall by most of its total. This single
measurement would change what search budgets are realistic, and it should be
run before any strategy code is written.

---

## 7. Existing concurrency

`ResimulationLoader` uses a module-level `ThreadPoolExecutor(max_workers=1)`,
with a documented rationale: inference is memory-bound before it is
compute-bound, so two concurrent passes on one machine contend for the same
weights and swap rather than finishing sooner. A durable per-run lock file
independently refuses a duplicate start from another process, with dead-PID
recovery.

`content/pipeline.py` uses `ThreadPoolExecutor(max_workers=2)` internally for
content stages.

Step 51 asks for concurrent candidate evaluation. On this hardware, 16 GB
unified memory shared with a ViT-g and a 3B language model, raising the worker
count will make the search slower. Phase 9 should expose
`max_concurrent_candidates` as a real knob, default it to 1, and record why in
the config rather than in a comment.

---

## 8. Existing lineage

Lineage exists as fields rather than as a type:

- `ProposedVariantSpec.parent_variant_id` — one parent per candidate
- `ProposedVariantSpec.experiment_id`
- `ResimulationRequest.previous_resimulation_id` — links iterations
- `ResimulationRequest.iteration_index` / `max_iterations` — bounded loop,
  validated so a later iteration must cite its predecessor
- `ResimulationResult.candidate_run_id` — the produced run

That is a linked list, and it is enough to reconstruct a chain. It is **not**
enough for Step 64's tree: there is no child list, no sibling ordering, no
generation number, and no place to record why a candidate was retained or
eliminated. Phase 9 needs a real tree structure over these fields, and it can
build one without modifying Phase 7 or Phase 8 schemas.

Crossover (Step 30) breaks the single-parent assumption. If crossover is
implemented, `parent_variant_id` cannot express a two-parent candidate and
Phase 9 will need its own parentage record.

---

## 9. Major architectural risks

**R1 — No materialization. Blocking.** Search cannot evaluate what it cannot
build. A deterministic ffmpeg-based materializer must exist before any strategy
is useful, and it constrains the search space to mechanically realizable edits.
This is the largest single piece of new work in Phase 9 and it is not mentioned
in the brief's implementation order.

**R2 — 52 minutes per evaluation.** Every budget number in the brief is
optimistic by an order of magnitude in wall-clock terms. The brief's example UI
shows 25 candidates; that is a full day of compute. Phase 9 must default to
single-digit budgets and the real benchmark of Step 88 must be scoped
accordingly.

**R3 — The improvement floor is not zero, and it is large.** Measured this
session: swapping between two mirrors of nominally identical Llama weights
shifted whole-cortex mean by an average of **0.0157**, while the differences
between the three real variants were 0.0117 to 0.0241. Two of three pairwise
comparisons were smaller than that nuisance shift, and one ranking reversed.

For Phase 9 this is decisive. A search that reports an improvement of 0.01 has
found nothing distinguishable from an implementation detail.
`minimum_improvement` must default above the measured floor, the search report
must state the floor beside every improvement it claims, and "best observed"
must mean best by a margin that clears it. Reporting a winner whose margin is
inside the noise would be the single most misleading thing this system could do.

**R4 — Guardrails do not exist.** Steps 61, 62 and 86 assume Phase 8 enforces
them. It does not. `OptimizationConstraints` provides the vocabulary —
`frozen_modalities`, `preserve_intervals`, `max_duration_change_seconds`,
`frozen_event_types`, `forbid_new_voiceover` — but nothing checks a produced
variant against it. Feasibility checking must be built, and it must run on the
materialized artifact, not on the genome, or it will only verify intent.

**R5 — Evidence semantics.** Satisfying `evidence_ids` with a Phase 7 evidence
kind would make a sampled parameter indistinguishable from a measured finding.
Needs a distinct kind before `GenomeDecoder` exists. See §3.

**R6 — Cost vocabulary that does not match the hardware.** Naming a budget field
`MAX_GPU_SECONDS` on a CPU-only pipeline invents a measurement. Use wall-clock.

**R7 — Score directory proliferation.** Each candidate produces a new scoring
key because `cache_context` includes the run set. Twenty candidates means twenty
score directories per objective set. Harmless but worth a cleanup policy.

**R8 — Determinism is a strength to protect.** Bit-identical reruns mean a
seeded search is genuinely reproducible. Any concurrency or nondeterministic
ordering introduced in Phase 9 must not compromise that, which argues for
recording RNG state per proposal rather than relying on global ordering.

---

## 10. Recommended deviation from the brief's order

The brief's implementation order begins at `SearchExperiment` and reaches
evaluation at step 8. Given R1 and R2, I recommend two changes:

1. **Measure the per-modality cache question (§6) first.** It is one
   materialized file and one inference run, and its answer determines whether
   realistic budgets are 5 candidates or 40.
2. **Build the materializer before the strategies.** Steps 1–7 define types that
   cannot be exercised end to end without it. Random search against a genome
   that cannot become a file proves nothing about the pipeline.

The synthetic algorithm tests of Steps 77–87 are cheap, need no materializer,
and should be built alongside the strategies as the brief specifies. They are
what actually proves the search logic; the real-pipeline benchmark proves the
integration, at a cost of several hours for a handful of evaluations.

---

## 11. What this report does not cover

I have not yet examined how `ContentSearchSpace` parameters would map onto the
Phase 4 feature names that Phase 7 evidence uses, which matters for closing the
loop between a search parameter and the measured feature it is supposed to move.
That mapping should be settled during Step 6.
