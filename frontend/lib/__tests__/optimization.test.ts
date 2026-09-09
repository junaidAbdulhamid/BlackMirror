import { describe, expect, it } from "vitest";

import {
  directionLabel,
  evidenceFor,
  formatInterval,
  hasApproval,
  trackDuration,
  trackSegments,
  withReviewState,
} from "@/lib/optimization";
import type {
  Evidence,
  OptimizationInterval,
  OptimizationRecommendation,
  OptimizationResult,
  RecommendationReview,
} from "@/types/optimization";

function interval(
  id: string,
  kind: "weak" | "strong",
  start: number,
  end: number,
): OptimizationInterval {
  return {
    interval_id: id,
    kind,
    objective_id: "o1",
    start_seconds: start,
    end_seconds: end,
    sample_count: 2,
    contribution_share: 0.02,
    contribution: 0.02,
    reference_contribution_share: null,
    reason: "r",
  };
}

function evidence(id: string): Evidence {
  return {
    evidence_id: id,
    kind: "content_feature_delta",
    description: `evidence ${id}`,
    feature: "motion",
    source_value: 0.2,
    reference_value: 0.5,
    delta: 0.3,
    interval_start_seconds: 0,
    interval_end_seconds: 5,
    reference_variant_ids: ["B"],
    consistency: 1,
    reference_count: 2,
    sample_count: 5,
    association: "temporal_comparison",
  };
}

function recommendation(id: string, evidenceIds: string[]): OptimizationRecommendation {
  return {
    recommendation_id: id,
    title: "Increase visual movement",
    objective_id: "o1",
    source_variant_id: "A",
    interventions: [],
    target_interval_start: 0,
    target_interval_end: 5,
    rationale: "measured difference",
    evidence_ids: evidenceIds,
    reference_variant_ids: ["B"],
    evidence_confidence: {
      value: 0.8,
      reference_consistency: 1,
      feature_difference_strength: 0.6,
      objective_gap_strength: 0.5,
      evidence_count: 2,
      formula: "f",
      caveat: "not a success probability",
    },
    expected_direction: "test_for_increase",
    risk: "low",
    edit_cost: "low",
    priority: 0.8,
    is_bundle: false,
    hypothesis_id: "HYP-1",
    warnings: [],
  };
}

describe("directionLabel", () => {
  it("never renders an outcome claim", () => {
    for (const direction of [
      "test_for_increase",
      "test_for_decrease",
      "test_for_target_proximity",
    ]) {
      expect(directionLabel(direction).toLowerCase()).toContain("test for");
    }
  });
});

describe("trackSegments", () => {
  it("keeps every segment inside the track", () => {
    const segments = trackSegments(
      [interval("a", "weak", 0, 5), interval("b", "strong", 25, 30)],
      30,
    );
    for (const segment of segments) {
      expect(segment.leftPercent).toBeGreaterThanOrEqual(0);
      expect(segment.leftPercent + segment.widthPercent).toBeLessThanOrEqual(100.01);
    }
  });

  it("clamps an interval that runs past the track", () => {
    const [segment] = trackSegments([interval("a", "weak", 25, 60)], 30);
    expect(segment!.leftPercent + segment!.widthPercent).toBeLessThanOrEqual(100.01);
  });

  it("gives a zero-length interval a visible minimum width", () => {
    const [segment] = trackSegments([interval("a", "weak", 10, 10)], 30);
    expect(segment!.widthPercent).toBeGreaterThan(0);
  });

  it("returns nothing for a zero-length track rather than dividing by zero", () => {
    expect(trackSegments([interval("a", "weak", 0, 5)], 0)).toEqual([]);
  });
});

describe("trackDuration", () => {
  it("spans the furthest interval of either kind", () => {
    const result = {
      weak_intervals: [interval("a", "weak", 0, 5)],
      strong_intervals: [interval("b", "strong", 20, 28)],
    } as OptimizationResult;
    expect(trackDuration(result)).toBe(28);
  });

  it("is zero when there are no intervals", () => {
    expect(
      trackDuration({ weak_intervals: [], strong_intervals: [] } as unknown as OptimizationResult),
    ).toBe(0);
  });
});

describe("evidenceFor", () => {
  it("resolves cited ids and silently drops unknown ones", () => {
    const resolved = evidenceFor(recommendation("R1", ["E1", "MISSING"]), [evidence("E1")]);
    expect(resolved.map((item) => item.evidence_id)).toEqual(["E1"]);
  });
});

describe("withReviewState", () => {
  it("defaults to pending when a recommendation has no decision", () => {
    const rows = withReviewState([recommendation("R1", ["E1"])], []);
    expect(rows[0]!.state).toBe("pending");
  });

  it("reflects a recorded decision", () => {
    const reviews: RecommendationReview[] = [
      { recommendation_id: "R1", state: "rejected", reason: "brand" },
    ];
    expect(withReviewState([recommendation("R1", ["E1"])], reviews)[0]!.state).toBe("rejected");
  });
});

describe("hasApproval", () => {
  it("is false until something is approved or modified", () => {
    expect(hasApproval([])).toBe(false);
    expect(hasApproval([{ recommendation_id: "R1", state: "rejected" }])).toBe(false);
    expect(hasApproval([{ recommendation_id: "R1", state: "approved" }])).toBe(true);
    expect(hasApproval([{ recommendation_id: "R1", state: "modified" }])).toBe(true);
  });
});

describe("formatInterval", () => {
  it("renders two decimals so short windows stay distinguishable", () => {
    expect(formatInterval(6, 8)).toBe("6.00–8.00s");
  });
});

describe("evidence consistency reporting", () => {
  it("distinguishes an uncorroborated single reference from a pattern", () => {
    // The flaw this guards: one reference and three both reporting 1.0.
    const single = { ...evidence("E1"), consistency: null, reference_count: 1 };
    const pattern = { ...evidence("E2"), consistency: 1, reference_count: 3 };
    expect(single.consistency).toBeNull();
    expect(pattern.consistency).toBe(1);
    expect(single.reference_count).toBeLessThan(pattern.reference_count);
  });
});
