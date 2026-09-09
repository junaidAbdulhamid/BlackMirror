/**
 * Scalar response -> colour.
 *
 * All colour logic lives here. Components ask for colours; they never invent
 * them, so the legend and the surface can never disagree.
 *
 * Choice of scale: TRIBE's predicted responses straddle zero, so the default is
 * a **perceptually balanced diverging map** (cool = below baseline, warm =
 * above, near-neutral at zero). Both arms have comparable lightness ramps, so
 * neither sign looks "stronger" than the other — important, because negative
 * does not mean "bad" and this project must not imply otherwise.
 *
 * The sequential map is kept for data that is genuinely one-sided.
 */

import { clamp01 } from "@/lib/normalization";

export type ColorScheme = "diverging" | "sequential";

interface Stop {
  t: number;
  color: [number, number, number];
}

/**
 * Cool-to-warm diverging ramp, zero at the desaturated midpoint.
 * Derived from the Blue-White-Red family used in neuroimaging, darkened
 * slightly at the extremes to sit well on a dark scientific background.
 */
const DIVERGING: Stop[] = [
  { t: 0.0, color: [0.19, 0.35, 0.68] },
  { t: 0.25, color: [0.34, 0.6, 0.84] },
  { t: 0.5, color: [0.86, 0.87, 0.89] },
  { t: 0.75, color: [0.9, 0.51, 0.33] },
  { t: 1.0, color: [0.7, 0.16, 0.16] },
];

/** Perceptually-ordered sequential ramp (dark blue -> yellow), viridis-like. */
const SEQUENTIAL: Stop[] = [
  { t: 0.0, color: [0.267, 0.005, 0.329] },
  { t: 0.25, color: [0.229, 0.322, 0.545] },
  { t: 0.5, color: [0.128, 0.567, 0.551] },
  { t: 0.75, color: [0.369, 0.789, 0.383] },
  { t: 1.0, color: [0.993, 0.906, 0.144] },
];

function ramp(scheme: ColorScheme): Stop[] {
  return scheme === "diverging" ? DIVERGING : SEQUENTIAL;
}

/** Sample a ramp at t in [0,1]. Returns linear RGB in [0,1]. */
export function sampleColor(t: number, scheme: ColorScheme = "diverging"): [number, number, number] {
  const stops = ramp(scheme);
  const x = clamp01(t);

  let lower = stops[0] as Stop;
  let upper = stops[stops.length - 1] as Stop;
  for (let i = 0; i < stops.length - 1; i += 1) {
    const a = stops[i] as Stop;
    const b = stops[i + 1] as Stop;
    if (x >= a.t && x <= b.t) {
      lower = a;
      upper = b;
      break;
    }
  }
  const span = upper.t - lower.t;
  const f = span <= 0 ? 0 : (x - lower.t) / span;
  return [
    lower.color[0] + (upper.color[0] - lower.color[0]) * f,
    lower.color[1] + (upper.color[1] - lower.color[1]) * f,
    lower.color[2] + (upper.color[2] - lower.color[2]) * f,
  ];
}

/** CSS `rgb(...)` for a normalized value. Used by the legend and tooltips. */
export function colorToCss(t: number, scheme: ColorScheme = "diverging"): string {
  const [r, g, b] = sampleColor(t, scheme);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

/** A CSS gradient matching the ramp, so the legend cannot drift from the mesh. */
export function gradientCss(scheme: ColorScheme = "diverging"): string {
  const steps = ramp(scheme)
    .map((stop) => `${colorToCss(stop.t, scheme)} ${(stop.t * 100).toFixed(0)}%`)
    .join(", ");
  return `linear-gradient(to right, ${steps})`;
}

/**
 * A 256-entry lookup table as a flat Float32Array of RGB triples.
 *
 * Precomputing once turns per-vertex colouring into an array index instead of
 * interpolation, which matters when repainting 20,484 vertices per timestep.
 */
export function buildColorLut(scheme: ColorScheme = "diverging", size = 256): Float32Array {
  const lut = new Float32Array(size * 3);
  for (let i = 0; i < size; i += 1) {
    const [r, g, b] = sampleColor(i / (size - 1), scheme);
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}

/** Medial-wall vertices: inert grey, clearly outside the response scale. */
export const MEDIAL_WALL_COLOR: [number, number, number] = [0.22, 0.23, 0.25];

/**
 * Vertices whose predicted value is NaN or infinite.
 *
 * These MUST NOT be drawn with a scale colour. A non-finite prediction rendered
 * at the neutral midpoint is indistinguishable from a genuine zero response —
 * it would look like real, unremarkable data. Magenta sits outside the entire
 * cool-to-warm ramp, so it can only mean "no valid value here".
 */
export const INVALID_VALUE_COLOR: [number, number, number] = [0.85, 0.1, 0.6];

/**
 * The whole surface when no prediction row covers the current moment.
 *
 * Distinct from both of the above: the data is fine, there simply is no
 * prediction for this instant.
 */
export const NO_COVERAGE_COLOR: [number, number, number] = [0.16, 0.17, 0.2];

/**
 * Fill a Three.js colour buffer for one timestep.
 *
 * Hot path — called on every timestep change for every hemisphere. It writes
 * into a preallocated Float32Array and does no allocation, so playback does not
 * generate garbage.
 *
 * Medial-wall vertices are painted inert grey rather than a scale colour:
 * the model emits values there, but fMRI signal is not meaningful, so showing
 * them as "response" would be misleading.
 */
export interface FillResult {
  /** Vertices skipped because their predicted value was NaN or infinite. */
  invalidCount: number;
}

export function fillVertexColors(
  target: Float32Array,
  predictions: Float32Array,
  frameOffset: number,
  predictionOffset: number,
  vertexCount: number,
  normalize: (value: number) => number,
  lut: Float32Array,
  medialWall: Uint8Array | null,
  options: { covered?: boolean; lutSize?: number } = {},
): FillResult {
  const { covered = true, lutSize = 256 } = options;
  const maxIndex = lutSize - 1;
  let invalidCount = 0;

  for (let i = 0; i < vertexCount; i += 1) {
    const globalVertex = predictionOffset + i;
    const out = i * 3;

    // No prediction row covers this moment: the surface is drawn inert rather
    // than showing a nearby row's values as if they applied here.
    if (!covered) {
      target[out] = NO_COVERAGE_COLOR[0];
      target[out + 1] = NO_COVERAGE_COLOR[1];
      target[out + 2] = NO_COVERAGE_COLOR[2];
      continue;
    }

    if (medialWall && medialWall[globalVertex] === 1) {
      target[out] = MEDIAL_WALL_COLOR[0];
      target[out + 1] = MEDIAL_WALL_COLOR[1];
      target[out + 2] = MEDIAL_WALL_COLOR[2];
      continue;
    }

    const value = predictions[frameOffset + globalVertex] as number;

    // A non-finite prediction must never be painted at the neutral midpoint,
    // where it would be indistinguishable from a genuine zero response.
    if (!Number.isFinite(value)) {
      invalidCount += 1;
      target[out] = INVALID_VALUE_COLOR[0];
      target[out + 1] = INVALID_VALUE_COLOR[1];
      target[out + 2] = INVALID_VALUE_COLOR[2];
      continue;
    }

    const lutIndex = Math.round(clamp01(normalize(value)) * maxIndex) * 3;
    target[out] = lut[lutIndex] as number;
    target[out + 1] = lut[lutIndex + 1] as number;
    target[out + 2] = lut[lutIndex + 2] as number;
  }

  return { invalidCount };
}
