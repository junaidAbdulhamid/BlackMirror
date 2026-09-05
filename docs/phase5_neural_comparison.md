# Phase 5 — controlled neural A/B/N comparison

Phase 5 compares two or more completed neural simulations without changing any
source artifact. One run is the declared reference; every other run is compared
to it using the signed convention `candidate - reference`.

## Comparability gate

A comparison is refused unless both runs share model fingerprint, preprocessing
configuration, compatibility patches, synthetic/real status, cortical surface,
vertex and hemisphere ordering, prediction semantics/units, analytics schema
and version, aggregation method, atlas identity, and ordered ROI identities.
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
- Divergence events: nonzero aligned times ranked by cortical L2 difference,
  with the largest finite absolute ROI deltas attached. A deterministic minimum
  sample separation prevents overlapping neighborhoods from being reported as
  duplicate peaks. “Event” means numerical divergence,
  never emotion, preference, memory, attention, or intent.
- Divergence windows: observed-sample neighborhoods around ranked events,
  reranked by mean L2 difference. Their start/end are real matched timestamps;
  gaps inside a window are not filled.

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
are true; otherwise values stay unavailable. One incompatible content pair is
reported as such without invalidating its neural comparison. The shared content
analysis version, model identifiers, and configuration are persisted alongside
each comparable pair.

Only signed/absolute/RMS/L2/cosine descriptive metrics are allowlisted.
Psychological, causal, and goal-conditioned requests raise a typed unsupported
metric error; those belong neither to this phase nor to honest interpretation.
