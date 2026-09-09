/**
 * Mapping raw predicted responses onto a visual range.
 *
 * SCIENTIFIC RULE: this layer never modifies stored predictions. It computes a
 * display transform only; the raw Float32Array stays exactly as Phase 1 wrote
 * it, and the active mode is always shown in the UI.
 *
 * Roughly half of TRIBE's predicted values are negative (47.6% in the reference
 * run), so the default scale is **zero-centred and symmetric**. Sign is
 * meaningful — it is the direction of the predicted response relative to
 * baseline — and a sequential ramp would hide it.
 */

import type { NormalizationMode, ResponseScale } from "@/types/neural";

/** A concrete display range. `center` is the value that maps to 0.5. */
export interface NormalizationRange {
  low: number;
  high: number;
  center: number;
  mode: NormalizationMode;
  symmetric: boolean;
  /** Human-readable description shown in the legend. */
  label: string;
}

/**
 * Build the display range for a mode.
 *
 * - `robust_global` (default): symmetric around zero at ±max(|p01|,|p99|).
 *   Percentiles keep a handful of extreme vertices from flattening everything
 *   else, and a fixed range keeps colours comparable across time — which is
 *   what makes watching the response evolve meaningful.
 * - `global`: symmetric around zero at ±max(|min|,|max|). Shows true extremes,
 *   at the cost of contrast.
 * - `per_frame`: rescales to the current timestep. Maximises contrast within a
 *   frame but colours are NOT comparable between timesteps.
 */
export function buildNormalizationRange(
  mode: NormalizationMode,
  scale: ResponseScale,
  frameStats?: { min: number; max: number },
): NormalizationRange {
  const diverging = scale.diverging_recommended;

  if (mode === "per_frame" && frameStats) {
    const extent = Math.max(Math.abs(frameStats.min), Math.abs(frameStats.max)) || 1;
    return diverging
      ? {
          low: -extent,
          high: extent,
          center: 0,
          mode,
          symmetric: true,
          label: `Per-frame · ±${extent.toFixed(3)}`,
        }
      : {
          low: frameStats.min,
          high: frameStats.max,
          center: (frameStats.min + frameStats.max) / 2,
          mode,
          symmetric: false,
          label: `Per-frame · ${frameStats.min.toFixed(3)} to ${frameStats.max.toFixed(3)}`,
        };
  }

  if (mode === "global") {
    const extent = Math.max(Math.abs(scale.min), Math.abs(scale.max)) || 1;
    return diverging
      ? {
          low: -extent,
          high: extent,
          center: 0,
          mode,
          symmetric: true,
          label: `Global min/max · ±${extent.toFixed(3)}`,
        }
      : {
          low: scale.min,
          high: scale.max,
          center: (scale.min + scale.max) / 2,
          mode,
          symmetric: false,
          label: `Global min/max · ${scale.min.toFixed(3)} to ${scale.max.toFixed(3)}`,
        };
  }

  const extent = scale.abs_max || 1;
  return diverging
    ? {
        low: -extent,
        high: extent,
        center: 0,
        mode: "robust_global",
        symmetric: true,
        label: `Robust global (1st–99th pct) · ±${extent.toFixed(3)}`,
      }
    : {
        low: scale.p01,
        high: scale.p99,
        center: (scale.p01 + scale.p99) / 2,
        mode: "robust_global",
        symmetric: false,
        label: `Robust global (1st–99th pct) · ${scale.p01.toFixed(3)} to ${scale.p99.toFixed(3)}`,
      };
}

/**
 * Map a raw value to [0, 1] for colour lookup.
 *
 * Values beyond the range are clamped, not discarded — clamping is a display
 * decision and the underlying number remains available on hover.
 */
export function normalizeValue(value: number, range: NormalizationRange): number {
  // Callers must screen non-finite values BEFORE colouring — see
  // INVALID_VALUE_COLOR. Returning the midpoint here is a last-resort guard so
  // a stray NaN cannot corrupt a LUT index; it is never a legitimate colour.
  if (!Number.isFinite(value)) return 0.5;
  const span = range.high - range.low;
  if (span <= 0) return 0.5;
  return clamp01((value - range.low) / span);
}

/** Signed position in [-1, 1] relative to `center`. Used by diverging maps. */
export function signedNormalize(value: number, range: NormalizationRange): number {
  if (!Number.isFinite(value)) return 0;
  const half = Math.max(range.high - range.center, range.center - range.low);
  if (half <= 0) return 0;
  return Math.max(-1, Math.min(1, (value - range.center) / half));
}

export function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/** min/max over one timestep, ignoring non-finite values. */
export function frameStatistics(
  predictions: Float32Array,
  index: number,
  vertexCount: number,
): { min: number; max: number; mean: number } {
  const start = index * vertexCount;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let count = 0;

  for (let i = 0; i < vertexCount; i += 1) {
    const value = predictions[start + i] as number;
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    count += 1;
  }
  if (count === 0) return { min: 0, max: 0, mean: 0 };
  return { min, max, mean: sum / count };
}

/**
 * Mean predicted response per timestep, for the aggregate temporal graph.
 *
 * Optionally excludes medial-wall vertices, where the model still emits values
 * but fMRI signal is not meaningful. Including them would bias the aggregate
 * toward a region that means nothing.
 */
export function meanResponseOverTime(
  predictions: Float32Array,
  timePoints: number,
  vertexCount: number,
  medialWall?: Uint8Array | null,
): Float32Array {
  const series = new Float32Array(timePoints);
  for (let t = 0; t < timePoints; t += 1) {
    const start = t * vertexCount;
    let sum = 0;
    let count = 0;
    for (let v = 0; v < vertexCount; v += 1) {
      if (medialWall && medialWall[v] === 1) continue;
      const value = predictions[start + v] as number;
      if (!Number.isFinite(value)) continue;
      sum += value;
      count += 1;
    }
    series[t] = count > 0 ? sum / count : 0;
  }
  return series;
}

/**
 * Timesteps where the aggregate response changes most between rows.
 *
 * Purely descriptive: these are *response changes*, not "emotional moments".
 * Phase 3 owns interpretation.
 */
export function detectResponseChanges(series: Float32Array, maxMarkers = 3): number[] {
  if (series.length < 3) return [];
  const deltas: { index: number; magnitude: number }[] = [];
  for (let i = 1; i < series.length; i += 1) {
    deltas.push({
      index: i,
      magnitude: Math.abs((series[i] as number) - (series[i - 1] as number)),
    });
  }
  const magnitudes = deltas.map((d) => d.magnitude);
  const mean = magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length;
  const spread = Math.sqrt(
    magnitudes.reduce((a, b) => a + (b - mean) ** 2, 0) / magnitudes.length,
  );
  // Only flag changes that stand out from the run's own variability.
  const threshold = mean + spread * 0.5;
  return deltas
    .filter((d) => d.magnitude >= threshold && d.magnitude > 0)
    .sort((a, b) => b.magnitude - a.magnitude)
    .slice(0, maxMarkers)
    .map((d) => d.index)
    .sort((a, b) => a - b);
}
