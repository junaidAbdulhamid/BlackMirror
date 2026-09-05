# Content Analysis Models

Every model Phase 4 uses, what it is used for, and what it costs. Recorded in
each run's `metadata.models` so a result can always be attributed.

**Everything runs locally.** No cloud API is called, so `metadata.api_cost` is
empty on every run to date. The field exists because a hosted VLM is the most
likely future upgrade to visual semantics.

---

## Speech

| | |
|---|---|
| **Primary source** | Phase 1's `events.parquet` — WhisperX `large-v3` run by TRIBE's own preprocessing |
| **Fallback** | `uvx whisperx --model large-v3`, invoked as a subprocess |
| License | WhisperX BSD-4-Clause; Whisper weights MIT |
| Output | Sentence segments + word-level timings |

**Why the Phase 1 transcript is preferred.** TRIBE already transcribed the
stimulus to build its text stream. Re-transcribing risks analysing text *the
model never saw*, which would quietly weaken every alignment claim in Phase 4
and every comparison in Phase 5. The fallback is used only when a run has no
speech events, and it is flagged in `warnings` when it fires.

---

## Visual semantics

| | |
|---|---|
| Model | `openai/clip-vit-base-patch32` via `transformers` |
| License | MIT (weights), Apache-2.0 (transformers) |
| Size | ~600 MB fp32, ~151 M parameters |
| Output | Structured attributes with **explicit abstention** |

**How it is used.** Zero-shot classification, not captioning. Image and text are
embedded into one shared space; each label group is scored with its own softmax.

### The confident-wrong-answer problem, and what actually fixed it

The Sintel title card was originally reported as **"a store or shop" at 0.62
confidence**. Three candidate fixes were measured; only the third worked.

Three further mechanisms were built and measured against real frames — a 365-label published scene taxonomy, a coarse-to-fine hierarchy, and cross-model agreement with an independently-pretrained LAION CLIP. All three failed, for reasons worth knowing before re-attempting them; the numbers are in `docs/phase4_content_intelligence.md`. Margin abstention remains the defence.

| Attempt | Result |
|---|---|
| Threshold the raw similarity | **Fails.** The title card scored 0.229 — *higher* than a correctly-labelled dialogue frame at 0.219. No absolute threshold separates them. |
| Add labels covering title cards | **Backfires.** "end credits on a dark background" then hijacked the dark dialogue frame at 0.98, because the trailer is very dark. |
| Ask CLIP "is this a title card?" | **Fails.** Labelled a wilderness shot a title card at 0.95. |

What works is two changes, both grounded in measurement:

**1. Confidence from the top-1/top-2 raw cosine margin, not softmax.** Softmax
must sum to 1 across a closed set, so it reports high confidence for a forced
choice among equally-bad options. The margin asks the better question — did the
model actually *prefer* this label? Below `SETTING_MARGIN_ABSTAIN` (0.018) the
setting is reported as `None` with a stated reason.

Measured margins: correctly-labelled dialogue **0.0248**; forced guesses on
out-of-distribution frames **0.0026–0.0144**. The threshold sits between them.

**2. Frame kind from evidence, not from a prompt.** OCR text coverage plus image
entropy and saturation decide whether a frame is `photographic`, `graphic` or
`blank`. A graphic has no meaningful "setting", so the field is suppressed
rather than guessed.

Entropy alone is not enough — measured: title card 0.26, black frame 0.00, and a
*dark photographic* frame 0.47. Saturation and OCR text separate the last two.

**The cost, stated plainly.** Abstention is conservative: the desert frame at
t=27 s was correctly "a desert" but had a 0.0037 margin, so it now abstains.
Trading some correct-but-uncertain labels for zero confident-wrong ones is the
right trade for a tool whose output feeds scientific comparison.

**Remaining limitations.** Scores are still relative to the label set; label
groups are scored independently so confidences are not comparable across groups;
CLIP is weak at counting and fine-grained product identity.

**Pluggability.** `VisualSemanticsBackend` is a Protocol. A hosted VLM can
replace CLIP without touching fusion, storage, or the API.

## Object detection

| | |
|---|---|
| Model | `google/owlv2-base-patch16-ensemble` via `transformers` |
| License | Apache-2.0 |
| Size | ~1.4 GB, **open vocabulary** — the query list is supplied per run |
| Output | `ObjectAppearance[]` — first/last seen, screen time, detection counts |

OWLv2 is given text queries and finds those, rather than predicting from a fixed
class list. Two structural properties matter: the vocabulary is supplied, so the
detector cannot invent a label; and it is free to return nothing.

**Why it replaced DETR.** `facebook/detr-resnet-50` predicts exactly 91 COCO
classes, so anything outside them is mapped to its nearest neighbour,
confidently, and no threshold fixes it. Measured on the Sintel trailer
(animated fantasy), identical frames and sampling:

| t | Content | DETR (91 COCO classes) | OWLv2 (open vocabulary) |
|---|---|---|---|
| 33 s | dragon in flight | "bird" 0.97 | "a flying creature" 0.73 |
| 35 s | dragon in flight | "person" 0.99 | "a flying creature" 0.78 |
| 37 s | fantasy interior | **"bed" 0.99** | *(nothing)* |
| 39 s | fantasy interior | "bed" 0.78 | *(nothing)* |
| 2 s | title card | "boat" 0.58 | *(nothing above 0.25)* |

Returning nothing is as important as returning the dragon. A fixed-class
detector has no way to say "none of my classes are here", which is how a fantasy
interior became a bed at 0.99 confidence.

**The vocabulary is part of the result.** Because only queried phrases can be
found, the query list bounds what could possibly be reported. A 38-phrase base
list covers common on-screen entities, and it is extended per run with words the
transcript repeats — a fantasy film has to be *asked* about dragons. Both the
queried vocabulary and its size are recorded, and every run warns that objects
were searched for rather than enumerated: an absent label means "not asked, or
not found", never "not present".

Remaining mitigations, unchanged in intent:

* `MIN_PERSISTENCE = 2` — an appearance must be seen in more than one sampled
  frame, with a warning naming what was dropped.
* Per-label greedy NMS at IoU 0.6. OWLv2 predicts a box per patch with no
  suppression and does emit duplicates: three overlapping "a mountain" boxes
  were measured on one frame, which would otherwise treble that label's count.
* Nothing is renamed by category. A query that returns 0.7 means the model
  matched that phrase to a region; it remains a model judgement.

**Measured on the real 10 s run:** 21 detections over 38 queries produced four
persistent labels — "a person" (13 detections, 6.19 s screen time), "a door",
"a house", "a weapon" — with one singleton ("a building") dropped. None of the
DETR-era artefacts ("bed", "clock", "boat") are reachable, because nothing asked
about them.

**Tracking is by label, not instance.** "person: 30.9 s screen time" means the
label was present in those frames, not that one person persisted. Screen time is
estimated at keyframe resolution and is a lower bound.

## Audio tagging

| | |
|---|---|
| Model | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| License | BSD-3-Clause |
| Size | ~340 MB, 87 M parameters, 527 AudioSet labels |
| Output | Music/speech/silence segments **with calibrated confidence**, plus discrete audio events |

**This replaced a DSP heuristic.** Music detection was previously "audible,
non-speech, low spectral flatness" — a reasonable proxy with no calibrated
confidence, which confused tonal sound design with music and could not name
anything else.

AudioSet is **multi-label**: music and speech genuinely co-occur, so the head is
sigmoid and each probability is independent. Reading them as competing classes
would be a category error.

Validated against the transcript on the Sintel trailer — Speech probability
peaks (0.86, 0.84, 0.67) fall exactly on the dialogue windows, and the 20–31 s
gap correctly scores Speech 0.06:

| Window | Top labels |
|---|---|
| 0–10 s | Music 0.61 |
| 10–20 s | Speech 0.86, Music 0.79 |
| 20–31 s | Music 0.68, Speech 0.06 |
| 31–41 s | Music 0.87, Speech 0.84 |

The DSP heuristic remains the documented fallback when the model is unavailable,
and a warning says so.

**Sample rate is not negotiable.** AST is trained at 16 kHz and its feature
extractor *rejects* any other rate. `load_waveform` previously returned each
file's native rate, the extractor raised, and the per-window handler swallowed
that at debug level — so on 44.1 kHz audio the tagger returned **zero windows
and no warning**. Waveforms are now resampled (torchaudio's polyphase filter,
falling back to interpolation), and a pass where every window failed says so.
Verified on a 44.1 kHz file: 0 windows before the fix, 5 after.

**Segments are a partition, not a window dump.** Labels are sampled onto a
0.25 s grid and equal neighbours collapsed into runs. Emitting one segment per
window returned a pile of near-duplicate 10 s spans that overlapped by 90%
rather than a partition of the timeline.

**Singing is scored against speech.** A sung vocal is voiced, loud and tonal, so
AudioSet lights up `Speech` on it too. `Speech` must exceed 1.2x `Singing`
before a region is called speech; otherwise it is segmented as music, and the
run says how many frames that affected.

**Limitations.** The 10.24 s receptive field is fixed by the model, so sounds
shorter than a second are diluted within a window — hopping cannot fix that.
Localization *was* improvable and was improved: the hop went from 5 s to 1 s
(90% overlap), placing a sound within one second rather than five, at roughly
5x the forward passes. AudioSet's label hierarchy means broad parents ("Sound
effect") are dropped as uninformative, and a transcript still wins for speech
boundaries because ASR gives exact word timings.

## Speaker diarization

| | |
|---|---|
| Model | `speechbrain/spkrec-ecapa-voxceleb` |
| License | Apache-2.0 |
| Size | ~80 MB, 192-d embeddings |
| Scope | **Utterance-level**, not frame-level |

Each transcript segment is embedded and clustered agglomeratively on cosine
distance with a threshold rather than a fixed *k* — forcing k=2 on a monologue
would invent a second speaker.

**It abstains when the embeddings carry no speaker structure.** Measured on the
Sintel trailer, whose dialogue sits under a full orchestral score: the two male
utterances were correctly closest (0.594), but the two female utterances scored
**0.811 — indistinguishable from the 0.818 between different speakers**.
Clustering produced four "speakers" for two people.

A silhouette-style separation score now gates the result. Below `MIN_SEPARATION`
(0.10) diarization reports nothing and explains why. On the trailer it correctly
abstains with separation 0.000.

**Limitations.** Overlapping speakers within one utterance are not resolved;
utterances under 0.6 s are skipped; loud background music degrades embeddings
badly, as measured above.
## On-screen text (OCR)

| | |
|---|---|
| Model | `rapidocr-onnxruntime` (PP-OCRv4 detection + recognition) |
| License | Apache-2.0 |
| Size | ~15 MB ONNX |
| Output | `TextOverlay[]` with intervals, confidence, occurrence counts |

Chosen over EasyOCR (heavy torch models) and Tesseract (system package, weaker
on video frames) because it is small, CPU-fast, needs no system dependency, and
returns per-detection confidence.

**Script coverage, stated precisely.** The bundled recognition model is
`ch_PP-OCRv4_rec`, which covers Chinese, Latin script and digits — the earlier
"Latin-script bias" note was itself wrong about what ships. What it does *not*
cover is Arabic, Cyrillic, Devanagari, Hebrew, Japanese kana or Korean hangul:
those need a different recognition model per script, plus script detection to
choose one. That is a real feature, not a config flag, and it is not implemented.
Text in an uncovered script is missed or garbled rather than flagged, which is
the honest shape of this gap.

**Other limitations.** Stylised or moving
title text is missed; low-contrast overlays fall below the 0.5 confidence floor.

---

## Embeddings, tone, and CTA

| | |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| License | Apache-2.0 |
| Size | ~90 MB, 384-dim |
| Output | Tone labels, CTA scores, segment embeddings |

Used together with a regex lexicon: patterns are precise and auditable,
embeddings catch paraphrase. Each `LanguageAnalysis` records which mechanism
fired via `provenance.source` (`lexicon_rule` vs `sentence_embedding`).

**Limitations.** Tone exemplars are hand-written and English-only. Tone below
0.30 similarity is reported as `UNKNOWN` rather than a coin flip. `structural_role`
is **positional**, not rhetorical — it splits the timeline into thirds and says
so; it is not a discourse model.

---

## Shot detection

| | |
|---|---|
| Library | PySceneDetect 0.7 — `ContentDetector` + `AdaptiveDetector` + `ThresholdDetector` |
| License | BSD-3-Clause |
| Method | Three detectors run independently; boundaries within 0.40 s are unioned |

**Why three.** `ContentDetector` measures frame-to-frame HSV change and is
excellent on hard cuts, but structurally blind to fades: two consecutive
near-black frames barely differ, so the signal it keys on never appears.
`ThresholdDetector` tracks absolute luminance and fires exactly there.
`AdaptiveDetector` compares against a rolling window and catches progressive
change a fixed threshold rides over.

Measured on the Sintel trailer, `ContentDetector` alone found 11 boundaries and
missed fades at 0.58 s, 10.83 s and 15.67 s. Luminance at those points is
0.0-7.1 out of 255 against a 46.9 clip mean — real fades, not detector noise.
The union finds 15.

**Transitions are typed by which detector fired**, not by ranking their
opinions. Luminance evidence means a fade; a content spike means a cut;
`AdaptiveDetector` alone means gradual. Ranking instead mislabelled the measured
124.6-luminance hard cut at 16.50 s as "gradual", because AdaptiveDetector fires
on hard cuts too. Merged boundaries keep the earliest time, since a fade begins
before content change becomes measurable.

**Limitations.** Wipes and cross-dissolves between two similarly-lit shots
remain the hard case: neither luminance nor a single-frame spike marks them.
Shots under 0.25 s are merged into their neighbour as detector noise, keeping
the shot list a complete partition.

---

## Audio signals

| | |
|---|---|
| Implementation | numpy + scipy, written out in `content/audio/signals.py` |
| Method | 25 ms frames / 10 ms hop: RMS, zero-crossing rate, spectral centroid, spectral flatness |

No librosa: these are short-time statistics that numpy computes in a few lines,
and writing them out makes the definitions auditable rather than hidden behind a
library default. It also avoids a numba dependency.

**The speech/music/silence split is a documented DSP heuristic, not a trained
classifier.** When a transcript exists its intervals are authoritative for
speech — a real ASR beats any heuristic. Music is inferred from tonality
(low spectral flatness) among audible non-speech audio, and will confuse sung
vocals with speech and tonal sound design with music. No confidence is reported
because a heuristic has none to give.
