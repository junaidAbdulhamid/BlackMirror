# Scientific Limitations

**Read this before interpreting any BlackMirror output.**

BlackMirror produces **predicted cortical responses** from a machine-learning
model. This document states plainly what those predictions are, and — more
importantly — what they are not. Every claim the product makes downstream must
stay inside these boundaries.

---

## 1. Predictions are model output, not measurement

TRIBE v2 is an *encoding model*: it was trained to predict fMRI BOLD responses
from stimulus features. When you run a video through BlackMirror, no brain is
involved. The output is the model's estimate of what a cortical response *might*
look like, carrying every error the model has.

- It is **not** a recording from you, your audience, or any specific person.
- It is **not** validated for your content domain unless it resembles the
  naturalistic movie/audio stimuli in the training data.
- Prediction accuracy varies by cortical region. Sensory areas (early visual,
  auditory) are predicted far better than association cortex. A number in a
  poorly-predicted region is much weaker evidence than the same number in a
  well-predicted one, and the raw matrix does not tell you which is which.

## 2. Predictions describe an *average* subject

`TribeModel.from_pretrained` forces `average_subjects=True`. The output is
therefore a prediction for an averaged subject representation, not for any
individual and not for any demographic group.

Do **not** claim BlackMirror simulates a particular person, age group, gender,
culture, or market segment. The released checkpoint provides no basis for that.

## 3. BOLD is not neural firing

fMRI measures the **blood-oxygen-level-dependent (BOLD)** signal — a slow,
indirect, metabolic correlate of neural population activity.

- BOLD lags neural activity by roughly 4–6 seconds and is smeared over several
  seconds. It cannot resolve fast, millisecond-scale neural events.
- Each fsaverage5 vertex summarises the activity of a very large number of
  neurons. Excitation and inhibition are not distinguishable in it.
- "High response" does not mean "more thinking", "more processing power", or
  "better".

## 4. Activation does not imply a mental state

This is the single most common misreading of neuroimaging, and the one this
project must never commit. A predicted response in a region does **not**
establish that a viewer:

- felt an **emotion** — regions activate in many contexts; the inference from
  activation to a specific feeling is a *reverse inference* and is not valid
  without a supporting inferential framework;
- **remembered** the content, or will remember it later;
- was **paying attention**;
- **preferred**, liked, or trusted the content;
- **intends to purchase** anything.

BlackMirror does not output emotion scores, memory scores, engagement scores or
purchase-intent scores, and Phase 1 deliberately stores no such derived
quantity. Any future scoring must be presented as *a defined function of
predicted responses*, with its assumptions stated — never as a measured
psychological fact.

## 5. Regional interpretation requires care

Mapping a vertex to a named region (via an atlas) tells you *where*, not *what*.
Region names are anatomical labels, not functions. Most regions participate in
many tasks, so "region X responded" rarely licenses "the viewer did Y".

The model also emits values for **medial wall** vertices, where there is no
meaningful cortical signal. These must be masked before any regional summary;
`blackmirror export-mesh` writes `medial_wall_mask.npy` for exactly this.

## 6. Comparison is more defensible than absolute values

The predicted values have no published physical unit. An isolated number is
close to meaningless. What is more defensible is a **controlled comparison**:
the same model, same configuration, same preprocessing, two variants that differ
only in the content being tested.

That is why Phase 1 records a full provenance record for every run. Before
comparing two runs, check that their `model_fingerprint`, `preprocessing_config`
and `applied_compat_patches` match. A comparison across differing configurations
is not a controlled comparison.

Even then, a difference between variants is a difference in *predicted cortical
response*. Whether it corresponds to any real-world outcome is an empirical
question this system does not answer.

## 7. Not for medical or diagnostic use

BlackMirror must not be used to diagnose, screen for, or make any inference
about a medical or psychiatric condition in any individual. It is not a medical
device and has undergone no clinical validation.

## 8. Known technical caveats in this pipeline

| Caveat | Consequence |
|---|---|
| The 5 s hemodynamic offset's sign is established from the model's training code, not from prose (`hemodynamic_offset_verified: true`) | Model output is already stimulus-aligned; `stimulus_time_seconds` is the content clock. Still an *average* HRF assumption — real lag varies by region and person, so attribution is approximate at the 1-2 s scale |
| Transcription drives the text stream, and ASR is imperfect | Misheard or missing words change the text features the model sees. Check `events.parquet` before trusting a text-heavy comparison |
| `BLACKMIRROR_TEXT_ENCODER_ID` can repoint the frozen text encoder | Only ever set it to the same model (Llama-3.2-3B). Runs with different encoders are non-comparable, and the fingerprint enforces that |
| `remove_empty_segments=True` (upstream default) drops event-free segments | The prediction time axis can be **non-uniform**; never assume row `i` is time `i × TR` |
| Non-CUDA runs apply compatibility patches to the transcription step | Word timings may differ slightly from a stock CUDA run; compare only equally-patched runs |
| Transcription can be disabled (`enable_transcription=false`) | Produces a **reduced-modality** run; TRIBE is trimodal, so such runs are not comparable with full-modality runs |
| Synthetic (mock backend) runs exist for development | Always check `model.is_synthetic` before interpreting anything |

## 9. Licensing

TRIBE v2 is released under **CC-BY-NC-4.0 — non-commercial use only**. This
constrains any commercial deployment of a product built on it. Confirm licensing
before any commercial use.

---

## Language rules for this codebase

Use: *predicted cortical response*, *predicted fMRI/BOLD response*, *model
prediction*, *predicted response difference between variants*.

Avoid: *brain scan*, *reads minds*, *measures emotion*, *proves*, *the viewer
felt/thought/wanted*, *digital twin of your customer*, *neural engagement score*.
