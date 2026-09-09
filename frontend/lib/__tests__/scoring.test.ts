import { describe, expect, it } from "vitest";

import {
  contributionPoints,
  formatDelta,
  formatPercent,
  formatRaw,
  hasObjectiveConflict,
  objectiveMatrix,
  rankedVariants,
  weightDrift,
} from "@/lib/scoring";
import type { ExperimentScoreResult, NeuralObjective, VariantScore } from "@/types/scoring";

function objective(id: string, weight: number): NeuralObjective {
  return {
    objective_id: id,
    name: id,
    metric: "MEAN_RESPONSE",
    target: { type: "roi", region_id: 0 },
    temporal_scope: { type: "full_stimulus" },
    direction: "maximize",
    normalization: { strategy: "none" },
    weight,
  };
}

function variant(id: string, total: number, scores: Record<string, number>): VariantScore {
  return {
    variant_id: id,
    total_score: total,
    objective_scores: Object.entries(scores).map(([oid, score]) => ({
      objective_id: oid,
      variant_id: id,
      raw_value: score,
      normalized_value: score,
      score,
      window: {
        start_seconds: 0,
        end_seconds: 1,
        sample_indices: [0],
        sample_count: 1,
        source: "full_stimulus",
      },
      target: { type: "roi", label: "r0", vertex_count: 0 },
      statistics: {},
      temporal_contribution: null,
      valid: true,
      warnings: [],
    })),
    contributions: [],
    baseline_delta: null,
    baseline_relative_delta: null,
    rank: null,
    valid: true,
    warnings: [],
  };
}

function result(overrides: Partial<ExperimentScoreResult> = {}): ExperimentScoreResult {
  return {
    schema_version: "1.0",
    experiment_id: "e1",
    objectives: [objective("o1", 0.5), objective("o2", 0.5)],
    baseline_variant_id: null,
    variant_scores: [
      { ...variant("A", 0.7, { o1: 0.9, o2: 0.4 }), rank: 1 },
      { ...variant("B", 0.6, { o1: 0.4, o2: 0.9 }), rank: 2 },
    ],
    ranking: ["A", "B"],
    ranking_margin: 0.1,
    pareto: null,
    sensitivity: [],
    normalization_detail: {},
    effective_weights: { o1: 0.5, o2: 0.5 },
    metadata: {
      scoring_version: "1.0",
      created_at: "2026-01-01T00:00:00Z",
      objective_set_hash: "abc",
      objective_definition_hash: "definition",
      input_artifact_sha256: {},
      atlas_name: null,
      metric_descriptions: {},
      interpretation_notice: "notice",
    },
    warnings: [],
    ...overrides,
  };
}

describe("number formatting", () => {
  it("never prints a non-finite value as a number", () => {
    expect(formatRaw(null)).toBe("—");
    expect(formatRaw(Number.NaN)).toBe("—");
    expect(formatDelta(undefined)).toBe("—");
    expect(formatPercent(null)).toBe("—");
  });

  it("signs deltas so direction is visible", () => {
    expect(formatDelta(0.038)).toBe("+0.0380");
    expect(formatDelta(-0.038)).toBe("-0.0380");
    expect(formatPercent(0.226)).toBe("+22.6%");
  });

  it("keeps four decimals, because raw values are small", () => {
    expect(formatRaw(0.0296)).toBe("0.0296");
  });
});

describe("weightDrift", () => {
  it("reports when an objective did not act at its stated weight", () => {
    // The measured failure: a declared 50/50 that behaved as 5/95.
    const drift = weightDrift(result({ effective_weights: { o1: 0.05, o2: 0.95 } }));
    const o2 = drift.find((d) => d.objectiveId === "o2")!;
    expect(o2.stated).toBeCloseTo(0.5);
    expect(o2.effective).toBeCloseTo(0.95);
    expect(o2.drift).toBeGreaterThan(0.15);
  });

  it("normalises stated weights that do not sum to one", () => {
    const drift = weightDrift(
      result({ objectives: [objective("o1", 7), objective("o2", 3)] }),
    );
    expect(drift.find((d) => d.objectiveId === "o1")!.stated).toBeCloseTo(0.7);
  });
});

describe("objectiveMatrix", () => {
  it("identifies the leader per objective", () => {
    const rows = objectiveMatrix(result());
    expect(rows.find((r) => r.objective.objective_id === "o1")!.leaderId).toBe("A");
    expect(rows.find((r) => r.objective.objective_id === "o2")!.leaderId).toBe("B");
  });

  it("detects a conflict when objectives disagree", () => {
    expect(hasObjectiveConflict(result())).toBe(true);
  });

  it("reports no conflict when one variant leads everything", () => {
    const dominant = result({
      variant_scores: [
        { ...variant("A", 0.9, { o1: 0.9, o2: 0.9 }), rank: 1 },
        { ...variant("B", 0.1, { o1: 0.1, o2: 0.1 }), rank: 2 },
      ],
    });
    expect(hasObjectiveConflict(dominant)).toBe(false);
  });
});

describe("rankedVariants", () => {
  it("orders by rank and pushes unranked variants last", () => {
    const mixed = result({
      variant_scores: [
        { ...variant("B", 0.6, { o1: 0.4, o2: 0.9 }), rank: 2 },
        { ...variant("C", 0, { o1: 0, o2: 0 }), total_score: null, rank: null },
        { ...variant("A", 0.7, { o1: 0.9, o2: 0.4 }), rank: 1 },
      ],
    });
    expect(rankedVariants(mixed).map((v) => v.variant_id)).toEqual(["A", "B", "C"]);
  });
});

describe("contributionPoints", () => {
  it("places the zero line inside the box so negatives stay visible", () => {
    const { zeroY } = contributionPoints([0, 1, 2], [-1, 0, 1], 100, 100);
    expect(zeroY).toBeGreaterThan(0);
    expect(zeroY).toBeLessThan(100);
  });

  it("returns nothing for mismatched inputs rather than drawing garbage", () => {
    expect(contributionPoints([0, 1], [1]).points).toBe("");
    expect(contributionPoints([], []).points).toBe("");
  });

  it("keeps every point inside the viewBox", () => {
    const { points } = contributionPoints([0, 5, 10], [0.2, -0.4, 0.9], 640, 96);
    for (const pair of points.split(" ")) {
      const [x, y] = pair.split(",").map(Number);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(640);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(96);
    }
  });

  it("survives a flat series without dividing by zero", () => {
    const { points } = contributionPoints([0, 1, 2], [0, 0, 0]);
    expect(points).not.toContain("NaN");
  });
});
