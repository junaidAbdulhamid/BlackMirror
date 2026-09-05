# Neural metric definitions

All inputs are predicted cortical response values, not measurements of a person.
None of these metrics establishes a psychological state.

## ROI response

- Formula: `A[t,r] = sum(R[t,v]) / n`, over finite, non-medial-wall vertices
  assigned to region `r`.
- Input/output: `T x V` response plus vertex mapping → `T x 148` response.
- Interpretation: average predicted response within an anatomical parcel.
- Limitation: equal vertex weighting; anatomical labels do not imply function.

## Hemisphere response and difference

- Formula: finite-value mean within each hemisphere after excluding medial-wall
  vertices; `left - right`.
- Output: three length-`T` series.
- Interpretation: mathematical hemispheric comparison.
- Limitation: it does not support “analytical left brain/creative right brain” claims.

## Global mean, median, standard deviation, and spatial variance

- Formula: ordinary finite-value distribution statistics across non-medial-wall
  vertices per row.
- Output: one length-`T` series each.
- Interpretation: center and spread of each predicted cortical pattern.
- Limitation: signed responses can cancel in the mean.

## Response magnitude

- Formula: `L2[t] = sqrt(sum_v R[t,v]^2)` over non-medial-wall vertices; RMS
  divides squared sum by finite count.
- Output: nonnegative length-`T` series.
- Interpretation: size of the predicted response vector; RMS is less sensitive
  to differing finite vertex counts.
- Limitation: neither quantity is “engagement.” L2 also scales with vertex count.

## Neural change magnitude

- Formula: `D[t] = ||R[t] - R[t-1]||_2` over jointly finite,
  non-medial-wall vertices, with `D[0] = 0`.
- Output: nonnegative length-`T` series.
- Interpretation: amount of pattern change between consecutive retained samples.
- Limitation: sparse TRIBE timelines may have gaps. The actual timestamp must be
  consulted; this is not normalized per elapsed second.

## Spatial concentration

- Formula: normalized Herfindahl index of absolute vertex response shares,
  `(sum(p_v^2) - 1/n) / (1 - 1/n)`.
- Output: length-`T`, from 0 (equal magnitude) to 1 (one-vertex concentration).
- Limitation: ignores sign and spatial adjacency; it is not “focus.”

## Cortical pattern similarity

- Formula: cosine similarity `a dot b / (||a|| ||b||)` on jointly finite values.
- Output: scalar in `[-1,1]`, or undefined for zero/no-overlap vectors.
- Interpretation: similarity in direction of two predicted cortical patterns.
- Limitation: insensitive to overall scale and does not explain the cause.

## ROI response correlation

- Formula: pairwise-finite Pearson correlation among ROI time-series; fewer than
  two shared samples or a constant series produces an undefined (`NaN`) entry.
- Output: `148 x 148` symmetric matrix.
- Interpretation: co-variation among predicted regional series.
- Limitation: **not functional connectivity**, causality, or anatomical connectivity.

## Response events

- Formula: strict local maxima above `mean + z_threshold * std`, followed by
  minimum sample-distance suppression. Defaults: z=1.5, distance=2, max=20.
- Types: whole-cortex response-magnitude peak and large-transition peak.
- Score: z-score above the corresponding series mean.
- Limitation: events depend on explicit parameters and retained sample cadence;
  they are not emotion, memory, attention, or intent events.
