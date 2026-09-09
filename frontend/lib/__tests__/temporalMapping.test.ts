/**
 * Time -> prediction index.
 *
 * The sparse-timeline cases matter most: TRIBE kept 4 of 100 segments in the
 * reference run, so any assumption that index == time / TR is wrong.
 */

import { describe, expect, it } from "vitest";
import {
  coveringPredictionIndex,
  formatTimecode,
  isWithinCoverage,
  mapTimeToPredictionIndex,
  predictionIndexToTime,
  temporalGap,
  timelineBounds,
} from "@/lib/temporalMapping";

const CONTIGUOUS = new Float32Array([0, 1, 2, 3]);
// A real gappy timeline: rows exist at 0-2s and 7-8s, nothing between.
const SPARSE = new Float32Array([0, 1, 2, 7, 8]);
const ONE_SECOND = new Float32Array([1, 1, 1, 1, 1]);

describe("mapTimeToPredictionIndex", () => {
  it("maps exact sample times to their own index", () => {
    for (let i = 0; i < CONTIGUOUS.length; i += 1) {
      expect(mapTimeToPredictionIndex(CONTIGUOUS[i] as number, CONTIGUOUS)).toBe(i);
    }
  });

  it("picks the nearest sample, not the preceding one", () => {
    expect(mapTimeToPredictionIndex(1.4, CONTIGUOUS)).toBe(1);
    expect(mapTimeToPredictionIndex(1.6, CONTIGUOUS)).toBe(2);
  });

  it("clamps below the first sample", () => {
    expect(mapTimeToPredictionIndex(-10, CONTIGUOUS)).toBe(0);
    expect(mapTimeToPredictionIndex(0, CONTIGUOUS)).toBe(0);
  });

  it("clamps above the last sample — never returns T", () => {
    expect(mapTimeToPredictionIndex(999, CONTIGUOUS)).toBe(CONTIGUOUS.length - 1);
  });

  it("never returns -1 for a non-empty timeline", () => {
    for (const t of [-5, 0, 0.5, 3, 100, Number.NaN]) {
      expect(mapTimeToPredictionIndex(t, CONTIGUOUS)).toBeGreaterThanOrEqual(0);
    }
  });

  it("handles a gappy timeline without assuming uniform spacing", () => {
    // The regression this whole design prevents: index != time / TR.
    expect(mapTimeToPredictionIndex(3, SPARSE)).toBe(2); // nearest is t=2
    expect(mapTimeToPredictionIndex(6.6, SPARSE)).toBe(3); // nearest is t=7
    expect(mapTimeToPredictionIndex(7.4, SPARSE)).toBe(3);
    expect(mapTimeToPredictionIndex(4.5, SPARSE)).toBe(2); // tie-ish, lower wins
  });

  it("returns 0 for a single-sample run and -1 for an empty one", () => {
    expect(mapTimeToPredictionIndex(42, new Float32Array([5]))).toBe(0);
    expect(mapTimeToPredictionIndex(0, new Float32Array())).toBe(-1);
  });

  it("agrees with a linear scan on random queries", () => {
    const timeline = new Float32Array([0, 1, 2, 7, 8, 13, 21, 34]);
    for (let i = 0; i < 500; i += 1) {
      const t = Math.random() * 40 - 5;
      let best = 0;
      for (let k = 1; k < timeline.length; k += 1) {
        if (
          Math.abs((timeline[k] as number) - t) <
          Math.abs((timeline[best] as number) - t)
        ) {
          best = k;
        }
      }
      expect(mapTimeToPredictionIndex(t, timeline)).toBe(best);
    }
  });
});

describe("predictionIndexToTime", () => {
  it("round-trips with the mapping", () => {
    for (let i = 0; i < SPARSE.length; i += 1) {
      const t = predictionIndexToTime(i, SPARSE);
      expect(mapTimeToPredictionIndex(t, SPARSE)).toBe(i);
    }
  });

  it("clamps out-of-range indices", () => {
    expect(predictionIndexToTime(-3, SPARSE)).toBe(0);
    expect(predictionIndexToTime(99, SPARSE)).toBe(8);
  });
});

describe("coverage", () => {
  it("reports the distance to the nearest sample", () => {
    expect(temporalGap(4.5, SPARSE)).toBeCloseTo(2.5);
    expect(temporalGap(7, SPARSE)).toBe(0);
  });

  it("knows when playback has drifted outside any sample", () => {
    // 4.5s sits in the gap; the nearest row is 2.5s away, far beyond one TR.
    expect(isWithinCoverage(4.5, SPARSE, ONE_SECOND)).toBe(false);
    expect(isWithinCoverage(1.2, SPARSE, ONE_SECOND)).toBe(true);
  });

  it("does not mistake proximity before a future row for coverage", () => {
    expect(coveringPredictionIndex(6.5, SPARSE, ONE_SECOND)).toBe(-1);
    expect(isWithinCoverage(6.5, SPARSE, ONE_SECOND)).toBe(false);
  });

  it("uses half-open row intervals at exact boundaries", () => {
    expect(coveringPredictionIndex(1, SPARSE, ONE_SECOND)).toBe(1);
    expect(coveringPredictionIndex(3, SPARSE, ONE_SECOND)).toBe(-1);
  });
});

describe("timelineBounds", () => {
  it("extends the end by one TR so the last row is reachable", () => {
    expect(timelineBounds(SPARSE, 1)).toEqual({ start: 0, end: 9 });
  });

  it("handles an empty timeline", () => {
    expect(timelineBounds(new Float32Array(), 1)).toEqual({ start: 0, end: 0 });
  });
});

describe("formatTimecode", () => {
  it("formats mm:ss.s", () => {
    expect(formatTimecode(0)).toBe("00:00.0");
    expect(formatTimecode(13.24)).toBe("00:13.2");
    expect(formatTimecode(75.5)).toBe("01:15.5");
  });

  it("never renders negative or NaN time", () => {
    expect(formatTimecode(-5)).toBe("00:00.0");
    expect(formatTimecode(Number.NaN)).toBe("00:00.0");
  });
});
