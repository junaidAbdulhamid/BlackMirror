/**
 * Display normalization.
 *
 * Guards the scientific rules: the raw array is never mutated, the default
 * scale is zero-centred because ~48% of TRIBE's values are negative, and
 * clamping is a display decision that must not lose the sign.
 */

import { describe, expect, it } from "vitest";
import {
  buildNormalizationRange,
  clamp01,
  detectResponseChanges,
  frameStatistics,
  meanResponseOverTime,
  normalizeValue,
  signedNormalize,
} from "@/lib/normalization";
import type { ResponseScale } from "@/types/neural";

// Modelled on the real reference run.
const SCALE: ResponseScale = {
  min: -1.2361,
  max: 0.668,
  mean: -0.0161,
  std: 0.2133,
  p01: -0.6012,
  p50: 0.0105,
  p99: 0.4032,
  abs_max: 0.6012,
  negative_fraction: 0.476,
  diverging_recommended: true,
};

describe("buildNormalizationRange", () => {
  it("defaults to a symmetric robust range centred on zero", () => {
    const range = buildNormalizationRange("robust_global", SCALE);
    expect(range.symmetric).toBe(true);
    expect(range.center).toBe(0);
    expect(range.low).toBeCloseTo(-SCALE.abs_max);
    expect(range.high).toBeCloseTo(SCALE.abs_max);
  });

  it("uses percentiles, not extremes, so outliers cannot flatten the map", () => {
    const robust = buildNormalizationRange("robust_global", SCALE);
    const global = buildNormalizationRange("global", SCALE);
    expect(robust.high).toBeLessThan(global.high);
  });

  it("global mode spans the true extremes symmetrically", () => {
    const range = buildNormalizationRange("global", SCALE);
    expect(range.high).toBeCloseTo(Math.max(Math.abs(SCALE.min), Math.abs(SCALE.max)));
  });

  it("per-frame mode uses the supplied frame statistics", () => {
    const range = buildNormalizationRange("per_frame", SCALE, { min: -0.2, max: 0.4 });
    expect(range.high).toBeCloseTo(0.4);
    expect(range.low).toBeCloseTo(-0.4);
  });

  it("falls back to a global range when frame stats are absent", () => {
    expect(buildNormalizationRange("per_frame", SCALE).mode).toBe("robust_global");
  });

  it("uses a sequential range for one-sided data", () => {
    const sequential: ResponseScale = {
      ...SCALE,
      min: 0.1,
      p01: 0.2,
      negative_fraction: 0,
      diverging_recommended: false,
    };
    const range = buildNormalizationRange("robust_global", sequential);
    expect(range.symmetric).toBe(false);
    expect(range.low).toBeCloseTo(0.2);
  });

  it("labels every mode so the legend can state what is applied", () => {
    for (const mode of ["robust_global", "global", "per_frame"] as const) {
      expect(buildNormalizationRange(mode, SCALE).label.length).toBeGreaterThan(0);
    }
  });
});

describe("normalizeValue", () => {
  const range = buildNormalizationRange("robust_global", SCALE);

  it("maps zero to the middle of a symmetric scale", () => {
    expect(normalizeValue(0, range)).toBeCloseTo(0.5);
  });

  it("maps the endpoints to 0 and 1", () => {
    expect(normalizeValue(range.low, range)).toBeCloseTo(0);
    expect(normalizeValue(range.high, range)).toBeCloseTo(1);
  });

  it("clamps beyond the range rather than producing out-of-gamut values", () => {
    expect(normalizeValue(-99, range)).toBe(0);
    expect(normalizeValue(99, range)).toBe(1);
  });

  it("preserves order — a larger value never maps lower", () => {
    const values = [-0.6, -0.2, 0, 0.2, 0.6];
    const mapped = values.map((v) => normalizeValue(v, range));
    for (let i = 1; i < mapped.length; i += 1) {
      expect(mapped[i] as number).toBeGreaterThanOrEqual(mapped[i - 1] as number);
    }
  });

  it("sends non-finite values to the neutral midpoint", () => {
    expect(normalizeValue(Number.NaN, range)).toBe(0.5);
  });

  it("never mutates the input array", () => {
    const original = new Float32Array([-0.5, 0, 0.5]);
    const copy = Float32Array.from(original);
    original.forEach((v) => normalizeValue(v, range));
    expect(Array.from(original)).toEqual(Array.from(copy));
  });
});

describe("signedNormalize", () => {
  it("preserves sign around the centre", () => {
    const range = buildNormalizationRange("robust_global", SCALE);
    expect(signedNormalize(-0.6, range)).toBeLessThan(0);
    expect(signedNormalize(0, range)).toBe(0);
    expect(signedNormalize(0.6, range)).toBeGreaterThan(0);
  });

  it("clamps to [-1, 1]", () => {
    const range = buildNormalizationRange("robust_global", SCALE);
    expect(signedNormalize(-99, range)).toBe(-1);
    expect(signedNormalize(99, range)).toBe(1);
  });
});

describe("clamp01", () => {
  it("clamps", () => {
    expect(clamp01(-1)).toBe(0);
    expect(clamp01(0.4)).toBe(0.4);
    expect(clamp01(5)).toBe(1);
  });
});

describe("frameStatistics", () => {
  const predictions = new Float32Array([1, 2, 3, /* t=1 */ -4, 0, 4]);

  it("reads only the requested timestep", () => {
    expect(frameStatistics(predictions, 0, 3)).toEqual({ min: 1, max: 3, mean: 2 });
    expect(frameStatistics(predictions, 1, 3)).toEqual({ min: -4, max: 4, mean: 0 });
  });

  it("ignores non-finite values", () => {
    const withNaN = new Float32Array([1, Number.NaN, 3]);
    expect(frameStatistics(withNaN, 0, 3)).toEqual({ min: 1, max: 3, mean: 2 });
  });
});

describe("meanResponseOverTime", () => {
  it("computes one mean per timestep", () => {
    const predictions = new Float32Array([1, 3, /* t=1 */ 0, 10]);
    expect(Array.from(meanResponseOverTime(predictions, 2, 2))).toEqual([2, 5]);
  });

  it("excludes medial-wall vertices, which carry no meaningful signal", () => {
    const predictions = new Float32Array([1, 100]);
    const wall = new Uint8Array([0, 1]);
    expect(Array.from(meanResponseOverTime(predictions, 1, 2, wall))).toEqual([1]);
  });
});

describe("detectResponseChanges", () => {
  it("finds the largest step changes", () => {
    const series = new Float32Array([0, 0.01, 0.02, 5, 5.01, 5.02]);
    expect(detectResponseChanges(series)).toContain(3);
  });

  it("returns nothing for a flat series", () => {
    expect(detectResponseChanges(new Float32Array([1, 1, 1, 1]))).toEqual([]);
  });

  it("returns nothing for a series too short to have changes", () => {
    expect(detectResponseChanges(new Float32Array([1, 2]))).toEqual([]);
  });

  it("returns sorted, bounded markers", () => {
    const series = new Float32Array([0, 5, 0, 6, 0, 7, 0, 8]);
    const markers = detectResponseChanges(series, 3);
    expect(markers.length).toBeLessThanOrEqual(3);
    expect([...markers]).toEqual([...markers].sort((a, b) => a - b));
  });
});
