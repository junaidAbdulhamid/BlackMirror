/**
 * Colour mapping and the vertex-colour hot path.
 *
 * The buffer-fill test also pins the prediction->vertex indexing that the whole
 * visualization depends on.
 */

import { describe, expect, it } from "vitest";
import {
  buildColorLut,
  colorToCss,
  fillVertexColors,
  gradientCss,
  INVALID_VALUE_COLOR,
  MEDIAL_WALL_COLOR,
  NO_COVERAGE_COLOR,
  sampleColor,
} from "@/lib/colorMapping";

describe("sampleColor", () => {
  it("returns RGB inside [0,1]", () => {
    for (let i = 0; i <= 20; i += 1) {
      for (const channel of sampleColor(i / 20)) {
        expect(channel).toBeGreaterThanOrEqual(0);
        expect(channel).toBeLessThanOrEqual(1);
      }
    }
  });

  it("clamps out-of-range inputs", () => {
    expect(sampleColor(-5)).toEqual(sampleColor(0));
    expect(sampleColor(5)).toEqual(sampleColor(1));
  });

  it("gives the diverging scale a near-neutral midpoint", () => {
    const [r, g, b] = sampleColor(0.5, "diverging");
    expect(Math.max(Math.abs(r - g), Math.abs(g - b))).toBeLessThan(0.1);
  });

  it("keeps the diverging arms distinct so sign is readable", () => {
    const low = sampleColor(0, "diverging");
    const high = sampleColor(1, "diverging");
    // Cool end is blue-dominant, warm end is red-dominant.
    expect(low[2]).toBeGreaterThan(low[0]);
    expect(high[0]).toBeGreaterThan(high[2]);
  });

  it("is monotonic in lightness for the sequential ramp", () => {
    let previous = -1;
    for (let i = 0; i <= 10; i += 1) {
      const [r, g, b] = sampleColor(i / 10, "sequential");
      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      expect(luma).toBeGreaterThan(previous);
      previous = luma;
    }
  });
});

describe("css helpers", () => {
  it("emits valid rgb()", () => {
    expect(colorToCss(0.5)).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
  });

  it("emits a gradient with stops", () => {
    expect(gradientCss("diverging")).toContain("linear-gradient");
  });
});

describe("buildColorLut", () => {
  it("has three floats per entry", () => {
    expect(buildColorLut("diverging", 256).length).toBe(256 * 3);
  });

  it("matches direct sampling at the endpoints", () => {
    const lut = buildColorLut("diverging", 256);
    const [r, g, b] = sampleColor(0, "diverging");
    expect(lut[0]).toBeCloseTo(r, 5);
    expect(lut[1]).toBeCloseTo(g, 5);
    expect(lut[2]).toBeCloseTo(b, 5);
  });
});

describe("fillVertexColors", () => {
  const lut = buildColorLut("diverging", 256);
  const identity = (v: number) => Math.min(Math.max(v, 0), 1);

  it("colours only the requested hemisphere's slice of the prediction", () => {
    // 2 timesteps x 4 vertices; right hemisphere starts at index 2.
    const predictions = new Float32Array([0, 0, 1, 1, /* t=1 */ 0, 0, 0, 0]);
    const target = new Float32Array(2 * 3);

    fillVertexColors(target, predictions, 0, 2, 2, identity, lut, null);
    const expected = sampleColor(1, "diverging");
    expect(target[0]).toBeCloseTo(expected[0], 5);

    // t=1 has zeros in the same slice, so the colour must change.
    const second = new Float32Array(2 * 3);
    fillVertexColors(second, predictions, 4, 2, 2, identity, lut, null);
    expect(second[0] as number).not.toBeCloseTo(target[0] as number, 5);
  });

  it("uses the frame offset to select the timestep", () => {
    const predictions = new Float32Array([0, 1]); // t=0 -> 0, t=1 -> 1
    const a = new Float32Array(3);
    const b = new Float32Array(3);
    fillVertexColors(a, predictions, 0, 0, 1, identity, lut, null);
    fillVertexColors(b, predictions, 1, 0, 1, identity, lut, null);
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  it("paints medial-wall vertices inert grey, not a scale colour", () => {
    const predictions = new Float32Array([1, 1]);
    const wall = new Uint8Array([0, 1]);
    const target = new Float32Array(2 * 3);

    fillVertexColors(target, predictions, 0, 0, 2, identity, lut, wall);

    expect(target[3]).toBeCloseTo(MEDIAL_WALL_COLOR[0], 5);
    expect(target[4]).toBeCloseTo(MEDIAL_WALL_COLOR[1], 5);
    expect(target[5]).toBeCloseTo(MEDIAL_WALL_COLOR[2], 5);
    expect(target[0]).not.toBeCloseTo(MEDIAL_WALL_COLOR[0], 5);
  });

  it("writes in place without allocating", () => {
    const target = new Float32Array(3 * 3);
    const before = target.buffer;
    fillVertexColors(target, new Float32Array([0, 0.5, 1]), 0, 0, 3, identity, lut, null);
    expect(target.buffer).toBe(before);
  });

  it("fills every vertex slot", () => {
    const vertexCount = 64;
    const target = new Float32Array(vertexCount * 3).fill(-1);
    const predictions = new Float32Array(vertexCount).fill(0.5);
    fillVertexColors(target, predictions, 0, 0, vertexCount, identity, lut, null);
    expect(Array.from(target).every((v) => v >= 0)).toBe(true);
  });
});


describe("fillVertexColors — invalid and uncovered data", () => {
  const lut = buildColorLut("diverging", 256);
  const identity = (v: number) => Math.min(Math.max(v, 0), 1);

  it("paints NaN and Infinity distinctly, never at the neutral midpoint", () => {
    // The bug this prevents: a non-finite prediction normalised to 0.5 renders
    // exactly like a genuine zero response, so broken data looks like real data.
    const predictions = new Float32Array([Number.NaN, Infinity, -Infinity, 0]);
    const target = new Float32Array(4 * 3);

    const { invalidCount } = fillVertexColors(
      target, predictions, 0, 0, 4, identity, lut, null,
    );

    expect(invalidCount).toBe(3);
    for (let v = 0; v < 3; v += 1) {
      expect(target[v * 3] as number).toBeCloseTo(INVALID_VALUE_COLOR[0], 5);
      expect(target[v * 3 + 1] as number).toBeCloseTo(INVALID_VALUE_COLOR[1], 5);
      expect(target[v * 3 + 2] as number).toBeCloseTo(INVALID_VALUE_COLOR[2], 5);
    }

    // The genuine zero must NOT share the invalid colour.
    const neutral = sampleColor(identity(0), "diverging");
    expect(target[9] as number).toBeCloseTo(neutral[0], 5);
    expect(target[9] as number).not.toBeCloseTo(INVALID_VALUE_COLOR[0], 5);
  });

  it("reports zero invalid vertices for clean data", () => {
    const result = fillVertexColors(
      new Float32Array(3 * 3), new Float32Array([0, 0.5, 1]), 0, 0, 3,
      identity, lut, null,
    );
    expect(result.invalidCount).toBe(0);
  });

  it("draws the whole surface inert when no row covers the moment", () => {
    const predictions = new Float32Array([1, -1]);
    const target = new Float32Array(2 * 3);

    fillVertexColors(target, predictions, 0, 0, 2, identity, lut, null, {
      covered: false,
    });

    for (let v = 0; v < 2; v += 1) {
      expect(target[v * 3] as number).toBeCloseTo(NO_COVERAGE_COLOR[0], 5);
      expect(target[v * 3 + 1] as number).toBeCloseTo(NO_COVERAGE_COLOR[1], 5);
      expect(target[v * 3 + 2] as number).toBeCloseTo(NO_COVERAGE_COLOR[2], 5);
    }
  });

  it("keeps the sentinel colours mutually distinguishable", () => {
    const distance = (a: readonly number[], b: readonly number[]) =>
      Math.hypot(
        (a[0] ?? 0) - (b[0] ?? 0),
        (a[1] ?? 0) - (b[1] ?? 0),
        (a[2] ?? 0) - (b[2] ?? 0),
      );

    expect(distance(INVALID_VALUE_COLOR, MEDIAL_WALL_COLOR)).toBeGreaterThan(0.3);
    expect(distance(INVALID_VALUE_COLOR, NO_COVERAGE_COLOR)).toBeGreaterThan(0.3);

    // The invalid colour must not collide with anything on the ramp either.
    for (let i = 0; i <= 32; i += 1) {
      expect(
        distance(INVALID_VALUE_COLOR, sampleColor(i / 32, "diverging")),
      ).toBeGreaterThan(0.15);
    }
  });
});
