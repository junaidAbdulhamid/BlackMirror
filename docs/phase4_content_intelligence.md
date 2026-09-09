# Phase 4 — Multimodal Content Intelligence

Turns the stimulus itself into timestamped, typed, traceable data, and aligns it
to the predicted cortical responses from Phases 1–3.

> **Association, never causation.** Phase 4 can say *"this phrase occurred around
> the same stimulus interval as a predicted response change."* It cannot say the
> phrase produced it, and nothing in the code, the API, or the UI claims otherwise.

---

## Architecture

```
Stimulus (referenced by path + SHA-256 from Phase 1)
   │
   ├──────────────┬──────────────┬───────────────┐
   ▼              ▼              ▼               ▼
Visual         Audio          Speech          Language
 shots          RMS/ZCR        Phase 1         tone · CTA
 motion         flatness       events.parquet  embeddings
 CLIP           segments       (WhisperX)      structure
 OCR
   │              │              │               │
   └──────────────┴──────┬───────┴───────────────┘
                         ▼
              Multimodal Fusion (event-driven intervals)
                         ▼
                   ContentEvent[]  →  ContentTimeline
                         │                   │
                         ▼                   ▼
              Feature matrix X∈R^(T×F)   ContentNeuralTimeMapper
                         │                   │
                         └─────────┬─────────┘
                                   ▼
                       NeuralContentAssociation[]
                       (Phase 3 events + surrounding content)
```

Visual and audio pipelines run **concurrently** (threads — the payloads are
large arrays and the heavy libraries release the GIL).

---

## Reuse, not duplication

Phase 4 consumes earlier phases through their artifacts, never by importing
their internals — the same boundary Phase 2 keeps with Phase 1.

| Reused | From | Why |
|---|---|---|
| `StimulusInput` path + SHA-256 | Phase 1 manifest | Media is referenced, never copied |
| `ArtifactStore` run directories | Phase 1 | One storage convention |
| **`events.parquet` transcript** | Phase 1 | See below |
| `analytics/metadata.json` events | Phase 3 | Read as JSON, not imported |
| `stimulus_time_seconds` clock | Phase 1 | The single shared timeline |

### The transcript decision

TRIBE's preprocessing already transcribed the stimulus with WhisperX and stored
word-level timings. Phase 4 **reuses that transcript** rather than producing its
own.

This matters more than it first appears. Re-transcribing would risk analysing
text *the model never saw* — a slightly different transcript means the content
analysis and the neural prediction describe subtly different stimuli, which
would quietly undermine every alignment claim here and every comparison in
Phase 5. WhisperX is run directly only as a fallback, and that is recorded in
`warnings` when it happens.

---

## Shot detection

**What:** finds cuts; returns the continuous takes between them.
**Why:** a shot is the atomic unit of video. Keyframe selection, the fusion grid
and "cuts per minute" all build on it.
**How:** PySceneDetect `ContentDetector` — HSV mean-absolute-difference between
consecutive frames; a cut is a single-frame spike above threshold 27.

Measured on the Sintel trailer (52 s): **11 shots**, mean 4.75 s, median 2.17 s,
11.5 cuts/min — a slow 13.9 s open, a rapid-cut middle (0.67–3.75 s shots), a
20 s close. The mean/median gap is exactly why both are reported.

**Limitation:** hard cuts are reliable; dissolves and fades are frequently
missed. Shots under 0.25 s are merged as detector noise, so the shot list stays
a complete contiguous partition.

---

## Scene segmentation

**A scene is not a shot.** A conversation cut back and forth is many shots but
one scene. Shots come from pixels; scenes need meaning.

**Algorithm** (greedy agglomeration over consecutive shots, documented because
it is a judgement call). A shot joins the current scene when:

* CLIP label-vector cosine similarity ≥ 0.86, **or**
* an utterance spans the shot boundary (strong evidence of continuity even when
  framing changes completely),

and the scene has not exceeded 20 s. Every scene records its
`grouping_reason`.

Label vectors rather than raw embeddings keep this interpretable: two shots are
similar *because they scored alike on named labels*.

---

## Keyframes

Three interior samples per shot. **Interior, not edge-inclusive** — a frame taken
exactly on a cut often lands on a dissolve or a black frame, describing neither
shot. OCR gets its own denser 1 Hz uniform sampling because a text overlay can
appear and vanish inside a single shot.

---

## Multimodal fusion — the core

Each pipeline produces intervals on its own natural grid: shots bounded by cuts,
utterances by speech, music by acoustics, motion by a 4 Hz sampler. None agree.

**Event-driven intervals, not fixed bins.** Fixed 1-second bins are the obvious
approach and are wrong: a bin straddling a cut mixes two shots into one
description, and a 0.4 s text flash disappears entirely.

Instead the grid is the **union of real boundaries** — every cut, utterance edge,
audio-segment edge and overlay edge. Between two consecutive boundaries nothing
changes categorically, so each interval is internally homogeneous *by
construction*. Intervals are then populated by asking each modality what it held
over that span.

**Cost:** intervals are uneven in length, so anything averaging over them must
weight by duration. Handled in `features.py` and `metrics.py`.

Intervals shorter than 0.35 s are absorbed into their neighbour, keeping the
event list a gapless partition — asserted by a test.

---

## Feature matrix

Events are uneven, which is right for reading and wrong for maths. `X ∈ R^(T×F)`
resamples everything onto a uniform 4 Hz grid: continuous signals by linear
interpolation, categorical signals by nearest containing event (never
interpolated).

Both representations are kept. Events are the human-facing truth; the matrix is
the machine-facing projection, and the method is stated rather than hidden.
Column order is fixed and versioned — Phase 5 diffing two matrices is only
meaningful if column *i* means the same thing in both.

See [`content_features.md`](content_features.md).

---

## Content → neural alignment

**Two clocks.** Content events are continuous stimulus time. Neural predictions
are TRIBE's sparse per-TR grid. Phase 1 established they share an origin
(`output_is_stimulus_aligned`), but **not a grid** — and up to 96% of neural rows
can be dropped. Alignment is a real operation, done explicitly in
`ContentNeuralTimeMapper` rather than assumed anywhere else.

`covering_row()` uses half-open `[start, start + TR)` interval membership, not
proximity — the same distinction Phase 2 established. A gap returns −1.

**Context windows.** A BOLD response reflects an interval of stimulation, not an
instant, and the haemodynamic response is smeared over seconds. "What was on
screen at exactly 13.2 s" is the wrong question. The default half-width is
**±2 s** — roughly one TR either side — and is recorded on every association.

---

## Association, and why it is not causation

`NeuralContentAssociation` records the content present in a window around a
Phase 3 neural event. Three things keep the claim honest:

1. **Field names.** `visual_context`, not `visual_cause`.
2. **A carried statement.** Every record contains: *"These content features
   occurred around the same stimulus interval as a predicted cortical response
   change. This is temporal co-occurrence, not evidence that the content
   produced the response."* It is asserted by tests and rendered verbatim in
   the UI.
3. **The heading.** The UI says "Co-occurring stimulus content".

Why this restraint is correct, not pedantry: the neural response is itself a
*model prediction*, the content labels are *model outputs with confidence*, and
the window is wide enough to contain several unrelated events. Three layers of
inference separate a content label from any real viewer. Co-occurrence is
genuinely all that has been established.

### Exploratory correlations — exercised on real data

`explore_correlations` correlates content features against neural metrics across
offsets 0-5 s. Content is resampled onto the neural grid, never the reverse.

The original Phase 4 report noted this was implemented but **untested against
real data**, because the only real run had 4 neural samples against a minimum of
8. A 10-second Sintel clip was put through the full Phase 1 pipeline
(57 minutes of CPU inference) to close that gap: **10 kept segments of 100**.

Running it for real found a bug no synthetic test could. The code looked for
neural series named `global_response` while Phase 3 writes `global_mean`; a
missing key silently produced an empty dict, so correlations returned nothing
*regardless of sample count*. There is now a warning when no recognised metric
is found.

**Current measured output** (75 correlations, 150 comparisons, analysis 1.2):

| Content feature | Neural metric | r | raw p | Bonferroni | FDR q | offset | n |
|---|---|---|---|---|---|---|---|
| object_count | global_l2_magnitude | −0.893 | 0.0028 | 0.420 | 0.210 | 1 s | 8 |
| entropy | global_l2_magnitude | −0.787 | 0.0118 | 1.000 | 0.443 | 0 s | 9 |
| speech_present | change_magnitude | +0.744 | 0.0215 | 1.000 | 0.548 | 0 s | 9 |
| motion_structural | spatial_concentration | −0.743 | 0.0219 | 1.000 | 0.548 | 0 s | 9 |

A raw p of 0.0028 reads as strong until the 150 tests behind it are counted.
**Nothing survives correction**, by either criterion, and the run says so.

Two changes since the first report. `comparisons` is now the number of tests
*actually performed* rather than pairs x offsets — the earlier 360 was an
overestimate, because many pair/offset combinations are skipped for insufficient
overlap; the true count here is 150. And every result carries a
Benjamini-Hochberg `q_value_bh` alongside Bonferroni, computed over every test
performed rather than over the reported best-per-pair. Correcting only the
winners would understate multiplicity by exactly the number of offsets searched.

The two criteria answer different questions and the gap between them is real
work, not decoration: the strongest result sits at Bonferroni 0.420 and FDR
0.210. Bonferroni controls the probability of *any* false positive; FDR controls
the expected *proportion* among those called significant, and is the standard
criterion for exploratory neuroimaging. Neither clears 0.05 here.

`motion_structural` appearing among the strongest associations is worth noting:
it is a new feature that did not exist when this was first run, and it
correlates differently from plain `motion`, which is the point of separating
them.

**Offsets are not BOLD-delay estimates.** Phase 1 output is already
stimulus-aligned, so a non-zero best offset is a search artefact, not evidence of
haemodynamic timing — and with 8-9 samples per test it is noise either way.

---

## Persistence, versioning, caching

```
artifacts/runs/<run_id>/content_analysis/
    metadata.json      the full ContentAnalysisResult
    features.npz       feature matrix + every dense signal
    keyframes/         extracted representative frames (referenced by the API)

<model_cache>/content_workspace/<hash prefix>/<stimulus hash>/
    audio.wav          decoded audio  — regenerable, shared across runs
    ocr_frames/        densely sampled frames — regenerable
    speaker/           cached speaker-model weights
```

**Artifacts vs intermediates.** Phase 1's rule is that artifacts *reference*
media rather than duplicating it. Decoded audio broke that rule — ~19 MB per run
for a 10-minute video — so regenerable intermediates now live in the model cache
keyed by stimulus hash. Two runs over the same stimulus share one decode, the
artifact store holds no duplicated media, and `blackmirror clear-content-cache`
discards the lot (measured: 89 MB reclaimed, run directory 236 KB → 60 KB).

Keyframes stay in the run directory: they are referenced by
`SceneSegment.keyframe_path`, served by the API, and shown in the UI, so they are
genuine artifacts rather than intermediates.

`cache_key = hash(analysis_version, stimulus_sha256, every model id, config)`.
A stale key recomputes; a matching one returns immediately.

**Measured: 16.8 s cold → 0.42 s cached (40×).** A page load must never trigger
the pipeline, so the API 404s an un-analysed run with instructions rather than
starting a 17-second job.

---

## Failure isolation

Every stage runs inside `_Stage.run`, which catches, records a warning, and
returns a default. OCR failing must not destroy a run with good transcript and
shot data. Partial results are returned **with warnings**, never silently.

---

## API

| Endpoint | Returns |
|---|---|
| `GET /api/runs/{id}/content-analysis` | Full result |
| `.../content-analysis/events` | Fused content events |
| `.../content-analysis/transcript` | Segments + word timings |
| `.../content-analysis/scenes` | Scenes and their shots |
| `.../content-analysis/context?time=13.2&window=2` | What was happening at/around an instant |
| `.../content-analysis/associations` | Neural events + co-occurring content |
| `.../content-analysis/features` | Feature matrix, raw float32 |
| `.../content-analysis/keyframes/{name}` | One keyframe image |

Keyframe paths are resolved against the run directory and refuse traversal.

---

## UI

A **Content Intelligence** panel with five tabs: Overview (metrics + models),
Scenes (keyframe cards), Transcript (word-level seeking), Content Events
(list + inspector), Neural Associations.

Clicking any word seeks the shared timeline — verified: clicking "remember."
moved playback to **1.917 s**, exactly its Phase 1 word timing, and the video,
cortex and analytics all followed.

---

## Measured performance

Apple M5, 16 GB, CPU only.

| Stage | 4 s clip (1 shot, 3 keyframes) | 52 s trailer (11 shots, 33 keyframes) |
|---|---|---|
| ffprobe | 0.06 s | 0.06 s |
| Shot detection | 0.45 s | 0.65 s |
| Visual signals | 0.02 s | 0.40 s |
| Audio extract + signals | 0.07 s | 0.26 s |
| Keyframe extraction | 0.15 s | 2.20 s |
| CLIP semantics | 7.53 s (incl. cold model load) | 3.19 s (warm) |
| OCR | 1.05 s | 6.81 s |
| **Component total** | **9.3 s** | **13.6 s** |

Full pipeline on the real run (4 s, all modalities + fusion + alignment):
**16.8 s cold**, dominated by CLIP (8.3 s) and language/embeddings (4.4 s);
**0.42 s cached**.

OCR scales with keyframe count (~0.2 s/frame); CLIP text embeddings are cached
across a run, so per-frame cost falls after the first.

**Model/API cost: zero.** Everything runs locally.

---

## Scientific limitations

Every one of these is measured, not assumed, and each has a corresponding
behaviour in the code.

| Limitation | What the system does about it |
|---|---|
| **Object vocabularies used to be closed.** DETR called a dragon "bird" 0.97 and a fantasy interior "bed" 0.99. | Replaced with open-vocabulary OWLv2. The vocabulary is supplied, so it cannot invent "bed", and it is free to return nothing. Same frames: "a flying creature" 0.78, and nothing where DETR said "bed". The queried phrases are persisted, because an absent label means *not asked or not found*, never *not present*. |
| **Closed-set CLIP labelling.** CLIP called the Sintel title card "a store or shop" at 0.62. | Improved: the reported setting must now pass **two independent tests** — the top-two margin *and* a null-prompt gap. Ground-truthed on eight frames, this removed a confident error the margin alone missed. Three other candidate fixes were measured to fail. See below. |
| **Speaker embeddings degrade under loud music.** Measured 0.811 within-speaker vs 0.818 between-speaker on the trailer. | Diarization scores its own clustering and abstains below a separation of 0.10. Source separation was tested as a fix and did not help — see below. |
| **Simultaneous speech.** | Now **scored** as a probability on each utterance, via pyannote in an isolated environment (opt-in). Reported as a continuous score, never a verdict: the model's overlap classes rank real overlap correctly (0.090 vs 0.022-0.039) but never win a hard argmax. Runs still warn that diarization itself assigns one speaker per utterance. |
| **Audio tagging has a 10.24 s receptive field.** | The field is fixed by the model; localization was not. The hop went from 5 s to 1 s, consecutive above-threshold windows are merged into one event, and each event is timed at its probability **peak** rather than a window start, with start/end still bounding the real uncertainty. Dilution *within* the receptive field remains irreducible. |
| **Tone describes the content, never a viewer.** | `ContentTone` is documented as a property of the writing. UI strings say "content tone". Below 0.30 similarity it reports `UNKNOWN` rather than a coin flip. |
| **`structural_role` was clock position, not rhetoric.** | Derived from discourse markers, CTA presence, and role-exemplar similarity with a required margin; position is the *fallback*, and the reason is recorded. |
| **Correlations need samples, and multiplicity is severe.** | **Resolved.** A 30 s run gives 30 samples, and 17 of 80 correlations survive FDR at q<0.05 across 480 comparisons — one of them surviving Bonferroni. Every result carries `p_value`, `lags_tested`, `comparisons`, `p_value_bonferroni` and `q_value_bh`. |
| **Associations are temporal co-occurrence.** | Field names say `context`; every record carries the non-causal statement; the UI renders it verbatim. |
| **Sung vocals are not speech.** | AudioSet `Singing` is scored against `Speech`, and singing is segmented as music with a warning. The DSP fallback cannot make this distinction and now says so. |
| **Shot detection missed dissolves and fades.** | Fixed: three detectors are unioned and each boundary is typed. See below. |
| **Overlapping speakers within one utterance.** | Still unresolved. ECAPA embeds a mixture as a mixture; separating concurrent speakers needs a diarizer built for overlap. Not claimed. |
| **OCR script coverage.** | Partly fixed: Japanese is now selectable and verified. Korean, Cyrillic, Arabic, Devanagari and Hebrew are **not** available — see "OCR" below for exactly why each one. |

### Two tests, because they fail on different frames

The top-two margin asks *can the model separate its labels?* It cannot ask *does
any label fit at all?* — that requires knowing what a non-answer scores. A bank
of twelve control prompts describing nothing visual ("a quadratic equation",
"the year 1834") supplies exactly that reference, and the best real label must
now beat the best control by 0.02.

The two tests were ground-truthed against eight frames from the Sintel trailer,
inspected by eye rather than assumed:

| Frame | What it actually is | CLIP's answer | margin | null gap | Reported? |
|---|---|---|---|---|---|
| t=36 s | desert, sand dunes | a desert | 0.0472 | 0.0540 | **yes — correct** |
| t=30 s | close-up of a person and a small creature | a desert | 0.0192 | 0.0069 | no — *margin passed it; the null test caught it* |
| t=42 s | the Sintel title card | a store or shop | 0.0133 | 0.0036 | no — *margin caught it* |
| t=02 s | snowy mountainside | a desert | 0.0004 | −0.0020 | no |
| t=20 s | a narrow alley | a city street | 0.0085 | −0.0043 | no — **correct answer lost** |
| t=25 s | fortress ruins at sunset | a desert | 0.0009 | 0.0059 | no |
| t=12 s | a dark frame | a desert | 0.0003 | −0.0427 | no |
| t=48 s | — | a stage or venue | 0.0036 | −0.0222 | no |

The two rows that matter are t=30 and t=42: each is caught by one test and
missed by the other, so neither test alone is sufficient and applying both is
not redundancy. Result on this set: one report, and it is correct; zero
confident errors.

The cost is stated plainly at t=20, where "a city street" was right and is now
withheld. Given that the alternative was reporting "a desert" for a close-up of
a person, that is the trade being made deliberately.

### Three candidate fixes that were measured and rejected

Closed-set CLIP labelling is the limitation most worth fixing, so three
mechanisms were implemented and tested against real frames from the Sintel
trailer. All three failed, and the measurements are recorded so they are not
re-attempted:

| Candidate | Why it should have worked | What was measured |
|---|---|---|
| **A published 365-label scene taxonomy** (Places365) replacing 11 hand-written labels | "No label describes a title card" is a coverage problem, so cover the space | Margins *collapsed*: 0.0006-0.0154 across eight frames, against 0.0032-0.0626 for the hand-written set. With 365 fine-grained competitors nothing clears the abstain threshold, so every frame abstains. A bigger vocabulary trades "no label fits" for "all labels fit equally" |
| **Coarse-to-fine hierarchy** — ask indoor/outdoor first, descend only if confident | A two-way contrast should be far easier than an eleven-way one | Coarse margins were **not** larger: median 0.0077. The clearest case is decisive — a desert frame scored 0.0009 on "indoors vs outdoors" and 0.0626 on the fine question. CLIP margins do not scale with question coarseness |
| **Cross-model agreement** — require an independently-pretrained LAION CLIP to agree | Different pretraining data implies different biases, so agreement is evidence | The two models agreed on 6 of 8 frames **including the failure case**: both called the title card "a desert", LAION at margin 0.0487. Agreement is correlated error, not independent evidence. On the two frames where they disagreed, the margin rule already abstained |

The conclusion is not that nothing works — it is that this failure is a property
of scoring an image against a fixed list of phrases, and the defence has to be
abstention rather than a better list. That is what ships.

### Source separation for diarization: also rejected

Speaker embeddings are corrupted by loud music, so HDEMUCS vocal separation was
tested as a fix, with ground truth built by splitting one speaker's utterance in
half (same speaker) and pairing it against a second speaker.

| Condition | same-speaker | cross-speaker | gap |
|---|---|---|---|
| Raw audio | 0.715 | 1.006 | +0.291 |
| Vocals isolated (HDEMUCS) | 0.721 | 0.983 | +0.262 |

Separation made it slightly *worse*. Sweeping music level from +20 dB to -10 dB
SNR did not produce a level at which it rescued a failing case. That sweep also
exposed why: at -10 dB the raw gap was *larger* (+0.394) than the separated one,
which is only possible if the embeddings are tracking the music bed rather than
the voice — the windows compared happen to sit in different musical passages.
That confound is precisely the reason the shipped defence is abstention rather
than threshold tuning, and it is why a 300 MB model and minutes of compute per
run were not adopted on this evidence.

### Shot detection: three detectors, one timeline

Frame-difference detection is blind to fades, because two consecutive near-black
frames barely differ. Measured on the Sintel trailer, `ContentDetector` alone
found 11 boundaries and missed fades at 0.58 s, 10.83 s and 15.67 s; frame
luminance there is 0.0-7.1 out of 255 against a 46.9 clip mean, confirming real
fades rather than detector noise.

Three detectors now run and their boundaries are unioned: `ContentDetector`
(hard cuts), `AdaptiveDetector` (progressive change), `ThresholdDetector`
(absolute luminance, i.e. fades). The trailer goes from 11 to 15 shots, and each
boundary is typed by *which* detector fired rather than by ranking their
opinions — ranking mislabelled the measured 124.6-luminance hard cut at 16.50 s
as "gradual", because AdaptiveDetector fires on hard cuts too.

### Overlapping speakers — now scored, in an isolated environment

Diarization assigns exactly one speaker per utterance. A two-speaker mixture
embeds 0.232 from the dominant voice and 0.808 from the other, so it is labelled
confidently with the louder one and nothing marks it. That is now scored.

**The environments cannot be merged.** Three independent collisions, each
measured rather than assumed:

| | pyannote needs | TRIBE/transformers needs |
|---|---|---|
| torch | 4.x: `>=2.8` (resolves to 2.14) | `>=2.5.1,<2.7` |
| numpy | 3.x uses `np.NaN` | `numpy==2.2.6` (2.0 removed `np.NaN`) |
| huggingface_hub | 3.x calls `hf_hub_download(use_auth_token=)` | transformers 5.x needs `>=1.5`, which removed that argument |

Shimming the 3.x path took one monkeypatch for the hub signature and a second
for `torch.load`'s `weights_only` default — and the second did not even work,
because Lightning captures its own `torch.load` reference at import. Patching
three libraries' internals to hold a scientific pipeline together is worse than
a subprocess, so `.venv-diarization` holds pyannote 4.0.7 with its own torch
2.14, the main environment is untouched at torch 2.6.0, and only JSON crosses
the boundary (`scripts/overlap_worker.py`).

**It reports a probability, not a verdict.** `pyannote/segmentation-3.0` is a
powerset model over {silence, s1, s2, s3, s1+s2, s1+s3, s2+s3}. On a constructed
mixture of two real Sintel speakers at matched levels:

| Region | P(two-speaker classes) | Hard argmax says |
|---|---|---|
| Speaker A alone | 0.039 | 1 speaker |
| **A and B together** | **0.090** — 2.3x | 1 speaker |
| Speaker B alone | 0.022 | 1 speaker |

The model separates the two voices correctly in their solo regions, and its
overlap classes rank the true overlap correctly — but never win the argmax. A
binary label would therefore report *no overlap at all*. The continuous score is
what carries the information, so that is what is stored, on
`TranscriptSegment.overlap_probability`.

**The threshold is a prompt, not a finding.** 0.07 is calibrated against exactly
one constructed example. It flags utterances worth listening to; it does not
establish that overlap occurred, and the warning text says so. On the real 30 s
run both utterances scored 0.027 and 0.005 and neither was flagged, which is
correct — they are consecutive single-speaker lines.

**One methodological trap, recorded because it inverted the answer.** Scoring
each utterance in isolation gives different and wrong results: a 1 s span padded
to the model's 10 s window is 9 s of silence, and speaker slots are reassigned
per call. Doing that ranked the true overlap region *lowest* (0.016) and a
single-speaker region highest (0.095) — an exact inversion. The worker scores
the whole recording once and reads spans off a shared time axis.

**Off by default** (`enable_overlap_scoring=False`): it needs a second
environment and an accepted pyannote licence, so it is opt-in rather than a
silent dependency.

### OCR — what was fixed, and exactly what was not

The recognition model's character dictionary is embedded in the ONNX metadata,
so an alternative script is a model swap rather than a config file. Rendered
text was used as ground truth:

| Script | Default model | Dedicated model | Status |
|---|---|---|---|
| Latin | "HELLO WORLD" 0.98 | — | already worked |
| Japanese | **"世界" at 1.00** — every kana silently dropped | "こんにちは世界" correct | **fixed** — `ocr_script="japanese"` |
| Korean | nothing | **nothing** | not fixed: the published Korean model returned no text at all |
| Cyrillic | "PИBET MИP" 0.88 (Latin lookalikes) | — | not fixable: no published RapidOCR model |
| Arabic | "lUeJP2l" 0.58 | — | not fixable: no published RapidOCR model |

Two details worth keeping. The v1-era Japanese model takes a 32-pixel input
height where the v4 default takes 48; passing the wrong shape raises
`Got: 48 Expected: 32` rather than degrading quietly. And the script must be set
explicitly, because **automatic selection by confidence was measured to be
unreliable**: the default model rates its incomplete Japanese reading at 1.00,
exactly as high as the correct reading from the Japanese model, so a
confidence-based selector would confidently choose the wrong one.

### Correlations — resolved on a 30 s run

The sample shortfall was quantified, then closed. Sample count is one per TR,
and TR is 1.0 s, so it is one per second of stimulus. At the multiplicity these
runs carry, and noting that for the single strongest of m tests BH and
Bonferroni coincide (q = p x m / 1):

| True effect | n for raw p<0.05 | n to survive correction |
|---|---|---|
| r = 0.9 | 5 | 11 |
| r = 0.7 | 9 | 22 |
| r = 0.5 | 16 | 48 |

The 10 s run's strongest association (r = −0.893) needed **n = 11** and had
n = 8 — three samples short, not orders of magnitude. A 30 s clip was therefore
put through the full pipeline (2 h 34 m of CPU inference, run
`20260905T031228Z-e3b95870`, 30 samples, gapless, no non-finite values).

**17 of 80 correlations now survive FDR at q<0.05 across 480 comparisons.**

| Content feature | Neural metric | r | p | FDR q | Bonferroni | offset | n |
|---|---|---|---|---|---|---|---|
| motion | global_mean | +0.765 | <0.00001 | 0.0006 | **0.000** | 0 s | 29 |
| audio_energy | spatial_concentration | +0.740 | 0.00002 | 0.0033 | 0.011 | 4 s | 25 |
| saturation | global_mean | +0.713 | 0.00001 | 0.0033 | 0.007 | 0 s | 29 |
| motion_structural | global_mean | +0.681 | 0.00005 | 0.0033 | 0.023 | 0 s | 29 |
| luminance_shift | global_mean | +0.674 | 0.00006 | 0.0037 | 0.030 | 0 s | 29 |
| text_present | global_mean | −0.663 | 0.00009 | 0.0048 | 0.043 | 0 s | 29 |
| motion | spatial_concentration | +0.678 | 0.00027 | 0.0082 | 0.129 | 5 s | 24 |

Three things in this table are worth stating explicitly.

`motion` vs `global_mean` survives **Bonferroni** at 480 comparisons, which is
the strictest criterion available and required no appeal to FDR.

The last row survives FDR (q = 0.008) but not Bonferroni (0.129). That is the
entire justification for adding FDR, and it is now demonstrated on real data
rather than argued from principle.

`motion_structural` and `luminance_shift` each survive **separately**, at
+0.681 and +0.674 against the same metric. They were one column (`motion`)
before the decomposition; that they now carry distinct surviving signal is
evidence the split was real and not cosmetic.

**Still not causal.** These are associations between a content feature and a
*predicted* response, on 24-29 samples from one stimulus, with the offset
selected from six candidates. Surviving correction means unlikely to be noise —
nothing more. The lag-0 rows with n=29 are the defensible ones; the 4-5 s
offsets with n=24-25 pay for both the lag search and fewer pairs.

### A misleading statistic, corrected

Earlier reports said the real run kept "10 of 100 segments", which reads as a
90% loss and a badly gappy timeline. It is neither. The 10 kept rows are
`[0,1,2,…,9]` seconds with 1.0 s durations and 3-10 events each: a **complete,
gapless tiling of the whole stimulus**. TRIBE's `n_samples` counts TR
sub-segments enumerated across every batch its loader yields, and the same
stretch of stimulus is enumerated more than once, so the kept/total ratio is not
a coverage fraction. The field now documents this, and a run whose kept rows
tile their span without a gap says so explicitly in `temporal.notes`.

### A silent failure found while fixing the above

Audio tagging required 16 kHz. `load_waveform` returned each file's native rate,
the AST feature extractor raised on anything else, and the per-window handler
swallowed that exception at debug level. The result: on any non-16 kHz audio the
tagger returned **zero windows and no warning**, and the analysis simply had no
audio events without saying why.

The pipeline's own ffmpeg extraction writes 16 kHz, so real runs were unaffected
— but `.wav` and `.flac` are valid stimulus inputs and are commonly 44.1 kHz.
Verified on a 44.1 kHz file: 0 windows before, 5 after. Waveforms are now
resampled, and a tagging pass that produces nothing while every window failed
says so at warning level.

### Structural markers, added for Phase 7

Two derived markers now accompany the fused timeline: `HOOK` and
`PRODUCT_REVEAL`. Neither is invented.

`HOOK` is the **first measured scene**, not an arbitrary opening duration. Its
provenance states that no detector for "a hook" exists and that the label
asserts position only. `PRODUCT_REVEAL` is the first appearance of an object
from the open-vocabulary detector's product list that persists at least 0.5 s;
a stimulus without one produces a warning rather than a marker.

They live in `structural_markers`, not in `events`. `events` is validated as a
**contiguous partition** of the stimulus so that coverage arithmetic holds, and
a marker overlaps whatever interval it sits inside. Appending them to `events`
was tried first and the schema rejected it, which is the invariant working.

### Abstention as a design principle

Four separate subsystems now decline to answer rather than guess: CLIP settings,
frame kind, diarization, and correlations. That is deliberate. A tool whose
output feeds scientific comparison is more useful when it says "cannot
determine" than when it produces a plausible label that a reader has no way to
distinguish from a reliable one.

The cost is real and worth stating: a correctly-identified desert frame now
abstains because its margin was 0.0037. Some true labels are lost to remove all
confident false ones.

## Known limitations

- **No frame-level diarization.** Utterance-level only; overlapping speakers
  within one utterance are unresolved.
- **Object tracking is by label, not instance.** "person, 30.9 s" means the
  label was present, not that one person persisted. Re-identification is out of
  scope.
- **Scene grouping is greedy and thresholded**, not learned.
- **Audio events are limited to AudioSet's vocabulary** and its window length.
- **Lag analysis** searches 0–5 s at 1 s granularity, which is coarse relative
  to haemodynamic timing.

## What Phase 5 can consume

- `ContentMetrics` — the natural per-variant diff targets.
- `X ∈ R^(T×F)` with a fixed, versioned column order.
- `ContentEvent[]` + `ContentTimeline` for interval queries.
- `ContentNeuralTimeMapper` for putting two runs on one clock.
- `NeuralContentAssociation[]` — the pattern for "what content sat near this
  response feature", which Phase 5 extends to "what content differed near where
  the two responses diverged".
