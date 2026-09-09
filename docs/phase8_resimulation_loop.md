# Phase 8 — bounded re-simulation loop

Phase 8 tests an approved Phase 7 hypothesis against an actual, content-addressed
variant. It never edits media implicitly. The default transformation adapter only
binds a user-supplied file; external generators must be explicitly registered by
the application and are not called by this package.

## Durable state machine

`BOUND → INFERENCE → ANALYTICS → CONTENT → SCORING → COMPARISON → EVALUATED`

Each successful stage produces an artifact beneath the configured artifact root
with a SHA-256, pipeline version, and the exact upstream hash map it consumed.
Every stage declares and hashes its complete source-owned file set (including
numeric arrays), not merely a JSON manifest.
The state is fsynced to a temporary file and atomically renamed after every stage.
Resume verifies the request, variant bytes, every durable artifact, and every
upstream edge before reusing work. Transitions are monotonic. A concurrent start
for the same id is rejected by an exclusive lock.

Failures are append-only records containing attempt, stage, exception type,
message, and timestamp. Successful inference/analytics/content artifacts remain
available when scoring or comparison fails. Retry resumes after the last durable
stage and loads the persisted score if scoring had already succeeded. Bounds and
stop reasons cover completion, failure, cancellation, timeout, resource limits,
maximum retry attempts, and maximum outer iterations. Cancellation uses a durable
side-channel marker checked between stages, so it does not wait on the worker's
exclusive lock. Dead-PID locks are recovered; live duplicate starts are refused.
Phase 8 evaluates one supplied candidate per request;
choosing or generating the next candidate is Phase 9 and is excluded.

## Lineage and outcomes

The request binds the original Phase 7 `ProposedVariantSpec`, parent run,
optimization request key, recommendation/intervention/hypothesis ids, exact
objective set and its stored hash, transformation adapter/version/method/tool/
model/parameters hash, source/variant paths, and source/variant content hashes.
Identical media is rejected except for the explicit `dry-run-test` adapter mode.

For every objective, the result preserves parent/candidate raw values and scores,
their separate deltas, and a direction-aware classification:

- maximize: positive candidate-minus-parent raw delta is improved;
- minimize: negative raw delta is improved;
- target: reduced absolute distance from the declared target is improved;
- values within the declared tolerance are unchanged;
- invalid or missing values are inconclusive.

For multi-objective approved bundles, the request must map every hypothesis to
its own objective and expected direction; a single shared fallback is rejected.
Each hypothesis stores its measured classification and a separate
pass/fail/inconclusive verdict. “Improved”
means only that an explicitly selected mathematical function of model-predicted
cortical responses moved in its requested direction. It is not a causal claim,
an estimate of human response, or evidence of real-world effectiveness.

## Building a request

`ResimulationRequest` is deliberately hard to satisfy: the exact objective set,
both of its hashes, a content-addressed media binding, and a per-hypothesis
objective and direction map that cannot contradict the objective it cites.
Those constraints are what stop a pass from measuring something other than what
was approved. They also make the request impossible to type by hand, and the
wrong fix would be to relax the schema.

`resimulation.request_builder.build_request` derives every one of those fields
from records Phase 6 and Phase 7 already wrote. A caller supplies only what a
person actually knows: which optimization, which approved candidate, which
media file, and what to name the pass. The source file defaults to the stimulus
the parent run was produced from, read out of that run's manifest, so the
comparison is anchored to the exact bytes behind the parent prediction rather
than to a file that merely shares its name.

Each hypothesis takes its objective and expected direction from Phase 7's own
`OptimizationHypothesis` record. A candidate may carry hypotheses about
different objectives; defaulting a missing one to the spec's target objective
would file its result under an objective it was never about, so a hypothesis
Phase 7 did not record is an error rather than a default. The single exception
is a synthesised hypothesis id, which `build_spec` emits when no approved
recommendation carried one, and which has no recorded objective to read.

## HTTP surface

Phase 8 is the one exception to the API being read-only, and the module says so
rather than leaving the claim standing. `/api/health` reports
`runs_inference: true` with `inference_scope: phase8_resimulation_only`.

| Route | Effect |
| --- | --- |
| `POST /api/experiments/{id}/resimulations` | Derive the request, bind it durably, queue the pass |
| `GET /api/resimulations` | Every recorded pass, optionally filtered by experiment |
| `GET /api/resimulations/{id}` | One pass, for polling |
| `POST /api/resimulations/{id}/resume` | Verify every stored checksum and continue |
| `POST /api/resimulations/{id}/stop` | Record a stop observed between stages |

A start returns as soon as the run is durable. The pipeline behind it runs on a
single background worker and is observed by polling, because a full pass takes
hours and a handler that waited for one would return nothing at all. Work is
serialized through one worker: inference is memory-bound before it is
compute-bound, so two concurrent passes on one machine contend for the same
weights rather than finishing sooner. The store's exclusive lock independently
refuses a duplicate start from another process.

Responses carry the durable state plus `worker_alive`, which is read from the
run's lock at request time and never persisted. A run whose worker was killed
still says `active` in its own file, because the process that would have
recorded the failure is the one that died. Without that field there is no way
to tell an interrupted run from a progressing one.

Status codes distinguish two different problems. A malformed request is 422; an
id already bound to different inputs is 409, because the request may be
perfectly well formed and the conflict is with what is already stored. The id
is how every artifact and outcome is addressed, so it is never silently
repointed.

An inference service is constructed on first use by a stage, not when a loader
is built. Binding a run, listing runs and requesting a stop all go through the
same backend without running anything, and none of them has any use for a
resolved device or a model backend.

## Command line

    blackmirror bind-resimulation loop-1 \
        --experiment exp --optimization <key> --candidate cand-1 \
        --media path/to/variant.mp4 --out request.json
    blackmirror resimulate request.json          # runs inline; takes hours
    blackmirror resume-resimulation loop-1
    blackmirror stop-resimulation loop-1
    blackmirror list-resimulations --experiment exp

## Interface

`/resimulation` binds a candidate and shows what each pass has measured. It
asks only for the fields a person knows and derives the rest, for the same
reason the builder does. The pipeline is drawn from durable state, so a filled
segment is a claim about a checksummed artifact rather than an animation.

Raw values lead and normalized scores follow, matching Phase 6, because
normalization can turn a negligible raw difference into a decisive-looking
score. Every row prints the improvement rule its objective actually declared,
so a fall under a minimize objective is not read as a regression. Measured
outcomes and hypothesis verdicts are kept visually separate: one says which way
a declared number moved, the other says whether a prediction written down
beforehand held, and collapsing them would turn "the number went up" into "the
idea was right".

Polling runs only while a pass is actually in progress. Resume is offered only
when nothing holds the run's lock, and stop only while a worker is alive.

## Persistence

State lives at `artifacts/resimulations/<resimulation_id>/state.json`. Media stays
at its source location and is never duplicated or changed. Existing Phase 1, 3,
4, 5, and 6 artifacts retain their native immutable stores; Phase 8 records
content-addressed pointers rather than copying their arrays.

## Verification boundary

Unit tests cover adapter authorization and hashes, schema lineage, monotonic
transitions, idempotent duplicate starts, concurrent locking, failure retention,
resume without repeated inference, upstream corruption rejection, atomic-temp
crash recovery, stop reasons, and outcome direction. Concrete execution delegates
to the existing public pipelines through a dependency-injected backend so tests
never require TRIBE or content models.

The request builder and HTTP surface are covered against a real Phase 6 score
and a real Phase 7 optimization written to disk, not stand-ins: reading those
records is the builder's entire job, so faking them would test nothing. Those
tests assert that hashes and per-hypothesis maps are derived rather than
supplied, that a minimize objective yields a test-for-decrease hypothesis, that
the source defaults to the parent run's stimulus, that an unapproved candidate
and identical media are refused, that a start returns at stage `BOUND` without
running the pipeline, that rebinding an id is a conflict rather than a bad
request, and that binding or stopping a run never constructs an inference
service.

### Measured end to end

The loop has been run to completion on real artifacts, not only against a fake
backend. Binding the approved candidate `cand-1` under `exp-optimize` to
`sintel_10s_vB.mp4` produced all five stage artifacts, reached `EVALUATED` on
the first attempt, and measured `roi_mean`:

| | parent | candidate | delta |
| --- | --- | --- | --- |
| raw | 0.006378 | -0.013970 | -0.020349 |

The objective declares `maximize`, so the value moved away from its declared
direction: outcome `worsened`, hypothesis verdict `fail`. That is a real
negative result and is reported as one. It is *not* a fair test of the
hypothesis, which called for raising audio energy between 6 and 8 seconds,
while the variant used was a global brightness and loudness edit built for a
different purpose. It demonstrates the machinery, not the idea.

The produced score landed under its own cache key, keyed on its own run set,
leaving the Phase 7 source score it was derived from intact — confirmed by
reading both after the pass. The produced objective definition hash matched the
request, proving the same mathematical objectives were used throughout.

An earlier attempt at the same pass failed at the scoring stage with
`variants use incompatible model fingerprints`, which was correct: the parent
had been produced with an ungated Llama mirror and the new run with the gated
repo, whose access had since been granted. That attempt is the better evidence
for the failure path — the three completed stages were retained with their
checksums, the failure was recorded with attempt, stage, type and message, and
the run remained resumable. See the pinning warning in `.env.example`.

The builder was also exercised against the repository's real artifacts: an
approved candidate under `exp-optimize` produced a valid request whose source
was resolved to the parent run's recorded stimulus and whose objective set,
both hashes and direction map came from storage.
