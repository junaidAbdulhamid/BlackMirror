# TRIBE v2 Integration

How BlackMirror uses Meta FAIR's TRIBE v2, verified against the installed
package rather than documentation alone.

## What TRIBE v2 is

**TRIBE** (TRImodal Brain Encoder) is a deep multimodal brain-encoding model
that predicts fMRI BOLD responses to naturalistic stimuli. TRIBE v2 builds on
the Algonauts 2025 award-winning architecture, trained on ~1,000 hours of fMRI
from 700+ subjects.

It combines three frozen foundation encoders — video, audio and text — into a
transformer that maps multimodal representations onto the cortical surface.

- Repository: <https://github.com/facebookresearch/tribev2>
- Weights: <https://huggingface.co/facebook/tribev2>
- Paper: <https://arxiv.org/abs/2507.22229> (TRIBE), see repo for the v2 paper
- Demo: <https://aidemos.atmeta.com/tribev2/>
- **License: CC-BY-NC-4.0 — non-commercial use only.**

## Version pinned by this project

| Item | Value |
|---|---|
| Package | `tribev2` 0.1.0, git commit `af58661791a351a448a489042a28f6c37e1c14b7` |
| Checkpoint | `facebook/tribev2` → `best.ckpt` (709 MB) |
| Snapshot | `f894e783020944dcd96e5568550afe2aa9743f9f` |

## The upstream API BlackMirror calls

Exactly three upstream entry points, all in `tribev2/demo_utils.py`:

```python
from tribev2.demo_utils import TribeModel

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    checkpoint_name="best.ckpt",
    cache_folder="./cache",
    device="cpu",                      # "auto" -> "cuda" if available else "cpu"
    config_update={"remove_empty_segments": True},
)
events = model.get_events_dataframe(video_path="clip.mp4")   # xor audio_path xor text_path
preds, segments = model.predict(events=events, verbose=False)
```

`preds` is `np.ndarray` of shape `(n_kept_segments, n_vertices)`; `segments` is a
list of `neuralset.segments.Segment`.

## Model configuration, read from the checkpoint

Values below are read at runtime from the loaded config, not hard-coded:

| Property | Value | Source |
|---|---|---|
| Trainable parameters | 177,205,397 | `sum(p.numel() for p in model._model.parameters())` |
| Surface mesh | `fsaverage5` | `data.neuro.projection.mesh` |
| Neuro frequency | 1.0 Hz → **TR = 1.0 s** | `data.neuro.frequency`; `Data.TR = 1 / neuro.frequency` |
| Subject handling | `average_subjects = True` | forced by `from_pretrained` |
| Vertices | 10,242 per hemisphere → 20,484 total | `tribev2/utils_fmri.py`: `FSAVERAGE_5 = ("fsaverage5", (10242,))` |

### Frozen feature extractors

| Modality | Encoder |
|---|---|
| Text | `meta-llama/Llama-3.2-3B` **(gated on HuggingFace)** |
| Audio | `facebook/w2v-bert-2.0` |
| Video (motion) | `facebook/vjepa2-vitg-fpc64-256` |
| Video (appearance) | `facebook/dinov2-large` |

Downloaded on first use and cached under `BLACKMIRROR_MODEL_CACHE_DIR`; well over
10 GB in total. `blackmirror verify-env` checks Llama access explicitly.

#### The gated text encoder

Any stimulus containing **speech** produces `Word` events, which routes features
through Llama-3.2-3B. Without access, feature extraction fails with
`OSError: You are trying to access a gated repo` — after preprocessing has
already run. Content with no speech skips the text extractor entirely
(TRIBE logs `Removing extractor text as there are no corresponding events`),
which is why silent clips run without a token.

The supported fix is to accept the license at
<https://huggingface.co/meta-llama/Llama-3.2-3B> and set `HF_TOKEN`.

Where that is not possible, `BLACKMIRROR_TEXT_ENCODER_ID` repoints the frozen
text encoder at a repo you can read (your own mirror, a local path, or an
ungated mirror of the same weights):

```bash
BLACKMIRROR_TEXT_ENCODER_ID=unsloth/Llama-3.2-3B blackmirror predict clip.mp4
```

Constraints, in order of importance:

1. **It must be the same model.** The trained head expects Llama-3.2-3B's hidden
   dimensions; a different architecture will fail to load or produce meaningless
   numbers.
2. **It is recorded in the model fingerprint**, so runs using different text
   encoders are correctly treated as non-comparable and never share a cache entry.
3. **It does not grant you a license.** Llama 3.2 remains under Meta's community
   license regardless of which repo the weights come from. Accepting it is the
   correct path for anything beyond local verification.

## Supported stimulus formats

From `tribev2.demo_utils.VALID_SUFFIXES`:

| Modality | Extensions |
|---|---|
| Video | `.mp4 .avi .mkv .mov .webm` |
| Audio | `.wav .mp3 .flac .ogg` |
| Text | `.txt` |

Exactly one input path may be supplied. A text stimulus is synthesised to speech
with gTTS, then transcribed back to recover word-level timings.

## Preprocessing, and how BlackMirror wraps it

`get_events_dataframe` runs the upstream transform chain: extract audio from
video → chunk into 30–60 s pieces → **WhisperX word-level transcription** →
attach sentences and context → drop missing.

BlackMirror does not reimplement any of this. It wraps the call
(`inference/preprocessing.py`), then adds a modality-neutral `events_summary`
and persists the event table as `events.parquet` — information upstream does not
save, and which Phase 3+ needs to correlate content events with response
changes.

### The transcription subprocess

`ExtractWordsFromAudio._get_transcript_from_audio` does **not** import WhisperX.
It shells out:

```
uvx whisperx <wav> --model large-v3 --language en --device <cpu|cuda> \
    --compute_type float16 --batch_size 16 \
    --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --output_format json
```

`uvx` provisions its own ephemeral environment, so:

- `uv` must be on `PATH` (`brew install uv`);
- the `whisperx` package need **not** be installed in this project's venv;
- patching an in-process `whisperx` import has no effect.

## Prediction output

```
preds.shape == (n_kept_segments, 20484)
```

| Axis | Meaning |
|---|---|
| 0 | Time. One row per segment of length TR (1.0 s). |
| 1 | Cortical vertex. `[0:10242]` = **left** hemisphere, `[10242:20484]` = **right**, in fsaverage5 mesh order. |

The hemisphere split is asserted by upstream's own plotting code
(`tribev2/plotting/cortical.py` slices `[:FSAVERAGE_SIZES[mesh]]` and
`[FSAVERAGE_SIZES[mesh]:]`), not inferred by us.

No physical unit is published upstream, so BlackMirror records `units: null`
rather than asserting one.

### The time axis is not necessarily uniform

This is the most consequential detail in the whole integration.
`TribeModel.predict` contains:

```python
if self.remove_empty_segments:                    # default True
    keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
...
y_pred = rearrange(y_pred, "b d t -> (b t) d")[keep]
```

Segments with no events are **dropped from the output**. Therefore:

> **Row `i` of the prediction matrix is not time `i × TR`.**

The only reliable alignment is `Segment.start`. BlackMirror always persists
these as `segment_start_seconds` in `temporal.npz`, and sets
`temporal.timeline_is_contiguous` so a consumer can tell at a glance. Set
`BLACKMIRROR_REMOVE_EMPTY_SEGMENTS=false` for a contiguous timeline.

### The 5-second hemodynamic offset — resolved

The upstream README says predictions "are offset by 5 seconds in the past, in
order to compensate for the hemodynamic lag." That prose is ambiguous about
direction, so the convention was established from the code instead.

**Evidence.** The value lives in the checkpoint as `data.neuro.offset = 5.0` on a
`neuralset.extractors.neuro.FmriExtractor`, documented there as *"Seconds to
shift TRs forward to align delayed BOLD response"*. During training it is applied
to the **fMRI recording**:

```python
# neuralset/extractors/neuro.py
yield TimedArray(data=data, frequency=ta.frequency,
                 start=ta.start - self.offset, duration=ta.duration)
```

A BOLD sample physically recorded at time `T` is therefore placed at timeline
position `T - 5`. A training segment at timeline time `t` consequently pairs
**stimulus content at `t`** with **BOLD measured at `t + 5`**.

**Conclusion.** At inference, a prediction row whose segment starts at `t` is the
predicted response *to the content at time `t`*. The model already performed the
lag compensation, so:

> **Segment start time IS stimulus time. Do not shift it again.**

BlackMirror records `hemodynamic_offset_verified: true`,
`output_is_stimulus_aligned: true`, and the derivation in
`hemodynamic_offset_source`. The offset value is read from the checkpoint, never
hard-coded. Two derived columns are provided in `temporal.npz`:

| Column | Value | Use |
|---|---|---|
| `stimulus_time_seconds` | `= segment_start_seconds` | **Sync content playback on this** |
| `bold_acquisition_time_seconds` | `= stimulus_time + 5.0` | When BOLD would be measured; not a playback clock |

> **Correction.** An earlier build of BlackMirror computed
> `stimulus_time = segment_start - 5.0`, producing non-physical times like
> `[-5, -4, -3]` for a stimulus starting at 0. That sign was wrong and is fixed.
> A guard now flags any pre-stimulus content time, and the raw matrix was never
> shifted in either version, so no stored prediction data was affected.

## Cortical mapping

fsaverage5 is a standard FreeSurfer template surface, not a TRIBE artifact.
BlackMirror obtains it from nilearn (`load_fsaverage("fsaverage5")`), so mesh
export requires neither the checkpoint nor GPU:

```
blackmirror export-mesh --space fsaverage5
```

See `docs/data_contracts.md` for the exported files.

Upstream also ships HCP ROI helpers (`tribev2.utils.get_hcp_labels`,
`get_hcp_vertex_labels`, `summarize_by_roi`, `get_topk_rois`). Phase 1 does not
use them — ROI interpretation is Phase 3 — but they are the natural basis for it.

## Compute

- `device="auto"` resolves to `"cuda" if torch.cuda.is_available() else "cpu"`.
  **There is no MPS code path**, in the head or the feature extractors. On Apple
  Silicon everything runs on CPU; BlackMirror refuses an explicit `--device mps`
  for this backend rather than producing a half-migrated model.
- Dependency pins that matter: **Python ≥3.11**, **`torch>=2.5.1,<2.7`**,
  `numpy==2.2.6`.

## macOS compatibility patches

Two upstream assumptions break on Apple Silicon. Both are patched narrowly at
runtime (`inference/compat_macos.py`), never by editing site-packages, and both
are recorded in each run's `provenance.applied_compat_patches`.

| Patch | Problem | Fix |
|---|---|---|
| `whisperx.compute_type:float16->int8_on_cpu` | `--compute_type float16` is hard-coded; CTranslate2 on Apple Silicon CPU supports only `{float32, int8, int8_float32, int16}` and raises `ValueError` before inference starts | Rewrite the argument to `int8` for non-CUDA devices |
| `whisperx.subprocess:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | The `uvx` environment resolves torch ≥2.6, where `torch.load` defaults to `weights_only=True`; WhisperX's pyannote VAD checkpoint then fails with `Unsupported global: omegaconf.listconfig.ListConfig` | Set PyTorch's documented override for that subprocess only |

Both alter the **transcription** step, which can shift word timings. Runs being
compared must therefore have identical `applied_compat_patches`. Disable with
`BLACKMIRROR_ENABLE_MACOS_COMPAT=false`.

## Dependency conflict worth knowing

`whisperx` (recent versions) requires `torch~=2.8.0`, which is incompatible with
TRIBE's `torch<2.7`. This does **not** matter in practice, because TRIBE runs
WhisperX through `uvx` in a separate environment. Do not install `whisperx` into
this project's venv: it will drag torch out of TRIBE's supported range.

## Differences from the prompt's assumed API

| Assumed | Actual |
|---|---|
| `get_events_dataframe` produces events | Correct, but it is a method on the loaded model, not a free function |
| Output `[T, V]` where T is temporal samples | Correct in shape, but T is *kept segments* and may be non-contiguous |
| WhisperX is an importable dependency | It is invoked as a `uvx` subprocess and is not declared in any dependency metadata |
| `device="auto"` picks the best device | It only ever picks CUDA or CPU — and it moves only the head, not the frozen extractors, which ship set to `"cuda"` |
| Predictions are "offset 5 s in the past", direction unclear | The offset is applied to the fMRI during training; model output is already stimulus-aligned |

## Segment totals

`TribeModel.predict` filters event-free segments but returns only the survivors.
The pre-filter total exists solely inside this log call:

```python
logger.info("Predicted %d / %d segments (%.1f%% kept)", n_kept, n_samples, pct)
```

BlackMirror attaches a `logging.Handler` to `tribev2.demo_utils` for the duration
of the call and reads `record.args` — the structured integers — rather than
formatting and re-parsing the message. The tally is validated (both integers,
`total >= kept >= 0`, and `kept` must equal the number of returned segments) and
degrades to `null` on any mismatch, so `n_segments_total` is either correct or
absent, never wrong. The handler also lowers the logger level for the duration
(suppressing propagation so visible output is unchanged), because a user running
at `WARNING` would otherwise lose the count silently.
