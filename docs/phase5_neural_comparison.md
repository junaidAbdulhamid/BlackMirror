# Phase 5 — controlled neural A/B/N comparison

Phase 5 compares two or more completed neural simulations without changing any
source artifact. One run is the declared reference; every other run is compared
to it using the signed convention `candidate - reference`.

## Comparability gate

A comparison is refused unless both runs share model fingerprint, preprocessing
configuration, compatibility patches, synthetic/real status, cortical surface,
vertex and hemisphere ordering, subcortex inclusion, medial-wall handling,
prediction semantics/units/**normalization**, sampling period (`tr_seconds`),
the **hemodynamic alignment convention** (`output_is_stimulus_aligned`, the
offset, and whether it was applied to the raw matrix), analytics schema and
version, aggregation method, atlas identity, and ordered ROI identities — 26
checks in total on a real pair.

The last two groups are the ones that decide whether a subtraction means
anything. Subtracting a normalised response from an unnormalised one produces a
number with no units, and nothing downstream could detect it. Two runs with
different sampling periods or different hemodynamic conventions attach different
stimulus moments to the same timestamp, so aligning on time pairs rows that
describe different content and every difference below becomes an artefact of
that. A structural test asserts that no field of the prediction schema is left
ungated without being explicitly listed as irrelevant, because a hand-written
gate silently goes stale as schemas grow.
Run IDs and stimulus hashes must differ. These checks establish matching model
conditions; they do not turn an observational model difference into a causal
effect.

## Timeline alignment

The default is the exact intersection of **observed stimulus timestamps** within
an explicit tolerance (`1e-6` seconds). Inputs must be finite, strictly
increasing, and duplicate-free. Matching is one-to-one. If a tolerance permits
multiple partners, alignment is ambiguous and fails. Zero overlap fails.

No interpolation is used by default. An explicit `linear_candidate_to_reference`
mode evaluates only observed reference timestamps within the candidate domain,
never extrapolates, and refuses candidate brackets wider than the configured
maximum gap. Persisted left/right candidate indices and weights make every
interpolated value reproducible. Forward-fill and `index * TR` remain forbidden.

## Outputs and definitions

- Signed cortical/ROI delta: `candidate - reference`.
- Absolute delta: magnitude of signed delta, with direction removed.
- Cortical L2 difference: `||candidate_t - reference_t||_2` over jointly finite,
  non-medial-wall vertices.
- Cortical cosine similarity: pattern-direction similarity at an aligned time.
- Region summaries: mean signed delta, mean absolute delta, RMS delta, and peak
  absolute delta with its sign and timestamp.
- Divergence events: nonzero aligned times ranked by the **RMS** cortical
  difference — the L2 divided by the square root of the number of jointly finite
  vertices in that row — with the largest finite absolute ROI deltas attached.
  Raw L2 is reported but is deliberately **not** the ranking key: it grows with
  how many vertices happened to be usable, so two rows with an identical 1.0
  difference at every comparable vertex score 2.0 and 1.414 when 4 and 2
  vertices are finite. Ranking on that would order timepoints by data
  availability rather than by divergence, and Phase 1 preserves non-finite
  predictions rather than dropping them, so counts genuinely can differ. Ties
  break by ascending sample index under a stable sort, so a rerun ranks
  identically. A deterministic minimum sample separation prevents overlapping
  neighborhoods from being reported as duplicate peaks. “Event” means numerical
  divergence, never emotion, preference, memory, attention, or intent.
- Divergence windows: observed-sample neighborhoods around ranked events,
  reranked by mean RMS difference for the same reason, with mean and peak L2
  also reported. Their start/end are real matched timestamps; gaps inside a
  window are not filled.

Nonfinite values remain nonfinite in signed/absolute delta artifacts. Medial-wall
vertices are also stored as nonfinite and excluded from every cortical scalar.
Scalar metrics use jointly finite cortical values; a row with no finite overlap
is rejected.

## Persistence and downstream boundary

Comparison and schema version 1.1 is stored atomically and immutably under:

```
artifacts/comparisons/<comparison_id>/
  metadata.json
  pairs/<candidate_run_id>.npz
```

Metadata records input run IDs, checks, alignment coverage, parameters,
summaries, and array descriptions. Dense difference arrays stay in NPZ.
For interpolated comparisons, candidate coverage means the fraction of observed
candidate rows that contributed either directly or as a bracketing endpoint.

Functional-network outputs use the official native-fsaverage5 Yeo 2011
seven-network annotation. The loader pins the trusted manifest and every source
file checksum, requires exact project/atlas pial coordinates and medial-wall
identity, and aggregates directly from vertices. Source:
<https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011>;
paper DOI `10.1152/jn.00338.2011`. Nilearn's official dataset documentation
identifies the Yeo 2011 atlas license as MIT and links the CBIG MIT license;
both provenance URLs are pinned in the atlas manifest.

When both corrected Phase 4 dense feature artifacts exist, their schema,
analysis version, model identifiers, configuration and ordered feature names
must match. Deltas are emitted only where both modality/event availability masks
are true; otherwise values stay unavailable.

That gate is load-bearing rather than decorative, because an unanalysed stream
is not represented as NaN: the feature matrix starts from zeros, so a run with
no video reports `motion = 0.0`, which is finite. Without a gate the comparison
subtracts 0.0 from 0.0 and reports a real, measured difference of zero for
content that was never looked at. The mapping from feature to availability mask
is hand-written and had already gone stale — three Phase 4 features
(`motion_structural`, `luminance_shift`, `brightness_p90`) were added upstream
and were being compared ungated. Comparison now refuses to run when any feature
lacks a gate, so the next feature added cannot repeat it silently. One incompatible content pair is
reported as such without invalidating its neural comparison. The shared content
analysis version, model identifiers, and configuration are persisted alongside
each comparable pair.

Only signed/absolute/RMS/L2/cosine descriptive metrics are allowlisted.
Psychological, causal, and goal-conditioned requests raise a typed unsupported
metric error; those belong neither to this phase nor to honest interpretation.
