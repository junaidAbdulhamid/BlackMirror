# Content Features

Every temporal feature Phase 4 produces. Columns of the feature matrix
`X ∈ R^(T×F)` are listed in the fixed order Phase 5 will rely on when diffing
two runs.

Stored in `artifacts/runs/<run_id>/content_analysis/features.npz` as `features`
(the matrix) and `feature_times` (row centre times, seconds).

**Grid:** uniform 4 Hz. Chosen to match the visual sampler and to be comfortably
finer than the 1 s TR of the neural predictions, so alignment never upsamples
the content side.

---

## Continuous visual features

### `motion`
- **Source:** frame differencing (`content/visual/motion.py`)
- **Type / range:** continuous, `[0, 1]`
- **Resolution:** 4 Hz
- **Meaning:** mean absolute difference between consecutive grayscale frames — relative magnitude of visual change.
- **Limitations:** it measures *change*, not object movement, and **does not measure viewer attention**. Frame differencing was chosen over optical flow because nothing downstream uses direction and flow costs far more. A global brightness change still lands here — which is why `motion` is now accompanied by the two features below that separate it out.

### `motion_structural`
- **Source:** mean absolute difference between frames standardised to zero mean and unit contrast · **Range:** `[0, ~2]` · 4 Hz
- **Meaning:** visual change with the global luminance change removed — a subject moving, not the lights changing.
- **Why standardised, not merely centred:** a fade to black is *multiplicative* (`f2 = a·f1`), so subtracting each frame's mean leaves a residual of `(a−1) × contrast`. Dividing by the frame's own standard deviation removes that too. Verified: a synthetic fade at ×0.5, ×0.2 and ×0.05, and an additive +0.2 flash, all score exactly `0.000000`, while plain differencing scores 0.26–0.49.
- **Limitations:** undefined when either frame has no contrast to standardise (below 0.025), where it returns 0. Standardising a near-blank frame amplifies noise without limit — measured, that turned a real motion of 0.0009 into an apparent 0.0366 — and comparing a structured frame against a flat one manufactures a large change at exactly the fade boundary it should ignore. Such transitions are luminance events and are reported as such.

### `luminance_shift`
- **Source:** `|mean(frame) − mean(previous)|` · **Range:** `[0, 1]` · 4 Hz
- **Meaning:** the global brightness change alone — fades, flashes, cuts to or from black.
- **Measured** on the Sintel trailer: fades at 10.83 s and 15.67 s score `motion_structural` 0.0000 with the change appearing entirely in `luminance_shift`; the lit-to-lit cut at 13.92 s inverts this exactly (structural 0.687, luminance 0.006). Correlation between `motion` and `motion_structural` across the clip is 0.62, so they are not redundant columns.

### `brightness`
- **Source:** mean grayscale intensity · **Range:** `[0, 1]` · 4 Hz
- **Limitations:** a global average; a bright subject on a dark ground reads as dark. Pair it with `brightness_p90`, which is what that case moves.

### `brightness_p90`
- **Source:** 90th-percentile grayscale intensity · **Range:** `[0, 1]` · 4 Hz
- **Meaning:** how bright the bright part of the frame is. A dark frame containing a lit subject has a low `brightness` and a high `brightness_p90`; a uniformly dim frame has both low. Measured across the Sintel trailer: mean `brightness` 0.183 against mean `brightness_p90` 0.337.

### `contrast`
- **Source:** grayscale standard deviation · **Range:** `[0, ~0.5]` · 4 Hz

### `saturation`
- **Source:** mean HSV S channel · **Range:** `[0, 1]` · 4 Hz

### `entropy`
- **Source:** Shannon entropy of a 64-bin intensity histogram · **Range:** `[0, 6]` · 4 Hz
- **Meaning:** proxy for visual complexity. Flat colour ≈ 0; dense texture high.

---

## Continuous audio features

### `audio_energy`
- **Source:** short-time RMS (25 ms / 10 ms hop), resampled to 4 Hz
- **Range:** `[0, 1]` for normalised PCM
- **Meaning:** signal power. **Not** "intensity", "excitement", or any affective quantity.

### `spectral_centroid`
- **Source:** magnitude-weighted mean frequency · **Range:** `[0, sr/2]` Hz
- **Meaning:** spectral "brightness". Higher for sibilance and cymbals, lower for bass.

### `spectral_flatness`
- **Source:** geometric ÷ arithmetic mean of the spectrum · **Range:** `[0, 1]`
- **Meaning:** near 1 for noise, near 0 for strongly tonal/harmonic audio. The main discriminator behind music detection.

### `speech_activity`
- **Source:** weighted DSP heuristic over energy, flatness and ZCR · **Range:** `[0, 1]`
- **Limitations:** classical VAD heuristic. It keys on loudness, tonality and zero-crossing rate, on all three of which sung vocals score exactly like speech, so it cannot tell singing from speech; it may also miss whispering. It is now only the last-resort fallback: a transcript is authoritative where one exists, and otherwise AudioSet tagging decides, scoring `Singing` against `Speech` so that song lyrics are segmented as music rather than dialogue. When the DSP path is used, the run warns that this distinction was unavailable.

---

## Categorical features

Never interpolated — a half-present CTA is not a meaningful quantity. Each is
`0.0` or `1.0`, resolved from the fused event containing that instant.

| Feature | Source | Meaning |
|---|---|---|
| `speech_present` | transcript intervals | An utterance covers this instant |
| `music_present` | AudioSet tagger (AST) | Dominant audio is music, with calibrated confidence |
| `text_present` | OCR overlays | On-screen text is visible |
| `cta_present` | language analysis | The covering event is a call to action |
| `shot_change` | shot detection | **Impulse** at a cut, not a continuous value |
| `object_count` | CLIP object labels | Number of object labels above threshold (integer, not binary) |

---

## Aggregate metrics

Single numbers per stimulus (`ContentMetrics`), the natural Phase 5 diff targets.

| Metric | Notes |
|---|---|
| `shot_count`, `mean_shot_duration`, `median_shot_duration` | Median is reported because one long establishing shot skews the mean badly on short content |
| `cuts_per_minute` | **Cuts, not shots**: N shots is N−1 transitions |
| `speech_fraction`, `silence_fraction`, `music_fraction` | Overlapping spans are **merged, not summed** — two overlapping speech segments cannot exceed the stimulus length |
| `word_count`, `words_per_minute` | From word timings where available |
| `text_overlay_count`, `cta_count`, `first_cta_time` | After temporal deduplication |
| `mean_motion`, `mean_audio_energy`, `mean_brightness` | Means of the dense signals |

**A metric that cannot be derived is `None`, never estimated.**

---

## What none of these measure

They describe **the content**, not any viewer. Rapid editing is not "excitement",
high audio energy is not "impact", and a detected CTA is not "persuasion". The
only bridge to predicted neural responses is temporal co-occurrence, and it is
labelled as such everywhere it appears.


---

## Availability masks

Three columns record whether a modality produced data at all, so a downstream
model can distinguish "measured zero" from "not measured":

| Feature | Meaning |
|---|---|
| `visual_available` | Visual signals were computed for this instant |
| `audio_available` | Audio signals were computed for this instant |
| `content_event_available` | A fused content event covers this instant |

Without these, a stimulus with no audio track and one with silent audio would
look identical in the matrix.

---

## Abstention in feature terms

Several fields can legitimately be absent, and absence is information:

| Field | Absent means |
|---|---|
| `VisualAttributes.setting` | The classifier had no confident preference, or the frame is not photographic |
| `TranscriptSegment.speaker` | Diarization could not separate speakers reliably |
| `ContentEvent.visual_description` | No confident visual label for that interval |
| `FeatureCorrelation` (empty) | Too few neural samples to correlate |

None of these should be filled with a default. A `setting` of `None` is a
different claim from a low-confidence guess, and Phase 5 will need that
distinction when deciding whether two variants genuinely differ.
