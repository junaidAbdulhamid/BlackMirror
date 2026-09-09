import { describe, expect, it } from "vitest";
import { rankRegionDifferences, sparklinePoints } from "../comparison";
import type { RegionDifferenceSummary } from "@/types/neural";

function region(id: number, signed: number, absolute = Math.abs(signed)): RegionDifferenceSummary {
  return {
    region_id: id,
    name: `ROI ${id}`,
    hemisphere: "left",
    mean_signed_delta: signed,
    mean_absolute_delta: absolute,
    rms_delta: absolute,
    peak_absolute_delta: absolute,
    peak_signed_delta: signed,
    peak_time_seconds: 0,
  };
}

describe("rankRegionDifferences", () => {
  it("ranks magnitude without discarding signed direction", () => {
    const ranked = rankRegionDifferences([region(1, 2), region(2, -5), region(3, 3)], 2);
    expect(ranked.map((item) => item.region_id)).toEqual([2, 3]);
    expect(ranked[0]?.mean_signed_delta).toBe(-5);
  });

  it("breaks equal-magnitude ties deterministically by region id", () => {
    expect(rankRegionDifferences([region(8, -2), region(3, 2)], 2).map((item) => item.region_id)).toEqual([3, 8]);
  });

  it("omits nonfinite summaries and validates limit", () => {
    expect(rankRegionDifferences([region(1, 1), region(2, Number.NaN)], 5)).toHaveLength(1);
    expect(() => rankRegionDifferences([], -1)).toThrow("nonnegative integer");
  });
});

describe("sparklinePoints", () => {
  it("preserves irregular observed-time gaps", () => {
    const points = sparklinePoints(
      new Float64Array([0, 1, 9, 10]),
      new Float64Array([1, 1, 1, 1]),
    );
    expect(points.split(" ").map((point) => Number(point.split(",")[0]))).toEqual([0, 80, 720, 800]);
  });

  it("rejects misaligned arrays", () => {
    expect(() => sparklinePoints(new Float64Array([0]), new Float64Array())).toThrow("align");
  });
});

describe("sparklinePoints robustness", () => {
  it("keeps every point inside the viewBox when timestamps are unsorted", () => {
    // Deriving extremes from the first/last entries would place points at
    // negative x for input that does not arrive sorted.
    const times = new Float64Array([5, 1, 9, 3]);
    const values = new Float64Array([1, 2, 3, 4]);
    const points = sparklinePoints(times, values);
    for (const pair of points.split(" ")) {
      const [x, y] = pair.split(",").map(Number);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(800);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(100);
    }
  });

  it("clamps negative values into the plot area rather than drawing outside it", () => {
    const points = sparklinePoints(new Float64Array([0, 1]), new Float64Array([-5, 10]));
    for (const pair of points.split(" ")) {
      const y = Number(pair.split(",")[1]);
      expect(y).toBeLessThanOrEqual(100);
      expect(y).toBeGreaterThanOrEqual(0);
    }
  });

  it("handles a long series without a call-stack overflow", () => {
    // Math.max(...array) throws for large inputs; scanning does not.
    const size = 200_000;
    const times = new Float64Array(size);
    const values = new Float64Array(size);
    for (let i = 0; i < size; i += 1) {
      times[i] = i;
      values[i] = i % 17;
    }
    expect(() => sparklinePoints(times, values)).not.toThrow();
  });

  it("still drops non-finite samples so gaps stay visible", () => {
    const points = sparklinePoints(
      new Float64Array([0, 1, 2]),
      new Float64Array([1, Number.NaN, 3]),
    );
    expect(points.split(" ")).toHaveLength(2);
  });
});
