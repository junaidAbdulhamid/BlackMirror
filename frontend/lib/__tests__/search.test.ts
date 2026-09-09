import { describe, expect, it } from "vitest";

import {
  budgetFraction,
  byGeneration,
  candidateStanding,
  eliminationFor,
  formatDuration,
  formatFitness,
  formatSigned,
  improvementVerdict,
  trajectoryPath,
} from "@/lib/search";
import type { CandidateResult, Population, TrajectoryPoint } from "@/types/search";

function result(overrides: Partial<CandidateResult> = {}): CandidateResult {
  return {
    candidate_id: "c1",
    status: "evaluated",
    scalar_fitness: 0.5,
    raw_objectives: { o1: 0.5 },
    candidate_run_id: "run-c1",
    variant_path: "/v.mp4",
    reason: "",
    duration_change_seconds: 0,
    compute: { wall_seconds: 10, tribe_runs: 1, served_from_cache: false },
    feasibility: { feasible: true, violations: [], unchecked: [] },
    genome: {
      genome_id: "c1",
      values: { gain: 1 },
      origin: "random_sample",
      generation: 0,
      parent_genome_ids: [],
      proposed_by: "random_search",
    },
    ...overrides,
  };
}

function point(n: number, best: number, fitness = best): TrajectoryPoint {
  return {
    evaluation_number: n,
    generation: 0,
    candidate_id: `c${n}`,
    fitness,
    best_fitness: best,
    best_candidate_id: `c${n}`,
    wall_seconds: n * 100,
    tribe_runs: n,
    is_new_best: false,
  };
}

describe("formatting", () => {
  it("shows an absent value as an em dash rather than zero", () => {
    expect(formatFitness(null)).toBe("—");
    expect(formatFitness(undefined)).toBe("—");
    expect(formatFitness(0)).toBe("0.0000");
  });

  it("signs an improvement so its direction is readable", () => {
    expect(formatSigned(0.05)).toBe("+0.0500");
    expect(formatSigned(-0.05)).toBe("-0.0500");
    expect(formatSigned(null)).toBe("—");
  });

  it("scales duration to a readable unit", () => {
    expect(formatDuration(45)).toBe("45s");
    expect(formatDuration(600)).toBe("10.0m");
    expect(formatDuration(7200)).toBe("2.0h");
    expect(formatDuration(null)).toBe("—");
  });
});

describe("budget", () => {
  it("returns null for an unlimited dimension", () => {
    // A bar implies a ceiling; there isn't one, so there is no bar.
    expect(budgetFraction(5, null)).toBeNull();
  });

  it("clamps to the range even when a limit is overshot", () => {
    expect(budgetFraction(12, 10)).toBe(1);
    expect(budgetFraction(5, 10)).toBe(0.5);
  });
});

describe("improvement verdict", () => {
  it("says plainly when a gain is inside the noise floor", () => {
    // The single most misleading thing this screen could omit.
    const verdict = improvementVerdict({
      absolute_improvement: 0.005,
      improvement_is_resolvable: false,
    });

    expect(verdict.resolvable).toBe(false);
    expect(verdict.label).toContain("noise floor");
  });

  it("reports a resolvable gain as above the floor", () => {
    const verdict = improvementVerdict({
      absolute_improvement: 0.05,
      improvement_is_resolvable: true,
    });

    expect(verdict.resolvable).toBe(true);
    expect(verdict.label).toContain("Above");
  });

  it("does not claim anything when nothing was measured", () => {
    expect(improvementVerdict({}).resolvable).toBeNull();
  });
});

describe("candidate standing", () => {
  it("treats a guardrail breach as ineligible, not merely low-scoring", () => {
    const blocked = result({
      scalar_fitness: 0.99,
      feasibility: {
        feasible: false,
        violations: [{ guardrail_id: "g1", message: "stability too low" }],
        unchecked: [],
      },
    });

    const standing = candidateStanding(blocked);

    expect(standing.eligible).toBe(false);
    expect(standing.label).toContain("guardrail");
  });

  it("distinguishes a failure from a rejection", () => {
    expect(candidateStanding(result({ status: "rejected" })).label).toContain(
      "before evaluation",
    );
    expect(candidateStanding(result({ status: "evaluation_failed" })).label).toContain(
      "failed",
    );
  });

  it("marks a measured, feasible candidate eligible", () => {
    expect(candidateStanding(result()).eligible).toBe(true);
  });
});

describe("grouping", () => {
  it("groups by generation and orders each best first", () => {
    const results = [
      result({ candidate_id: "a", scalar_fitness: 0.1 }),
      result({
        candidate_id: "b",
        scalar_fitness: 0.9,
        genome: { ...result().genome, genome_id: "b", generation: 0 },
      }),
      result({
        candidate_id: "c",
        scalar_fitness: 0.5,
        genome: { ...result().genome, genome_id: "c", generation: 1 },
      }),
    ];

    const grouped = byGeneration(results);

    expect(grouped.get(0)?.map((item) => item.candidate_id)).toEqual(["b", "a"]);
    expect(grouped.get(1)?.map((item) => item.candidate_id)).toEqual(["c"]);
  });

  it("finds why a candidate was eliminated", () => {
    const populations: Population[] = [
      {
        generation: 0,
        candidate_ids: ["a", "b"],
        survivors: ["a"],
        eliminated: [
          { candidate_id: "b", reason: "outside_beam", detail: "rank 2 of 2", rank: 2, of: 2 },
        ],
        diversity: 0.4,
        best_candidate_id: "a",
        best_fitness: 0.9,
      },
    ];

    expect(eliminationFor(populations, "b")?.detail).toBe("rank 2 of 2");
    expect(eliminationFor(populations, "a")).toBeNull();
  });
});

describe("trajectory path", () => {
  it("is empty with no points rather than drawing a stray line", () => {
    expect(trajectoryPath([], 100, 50)).toBe("");
  });

  it("starts with a move and continues with lines", () => {
    const path = trajectoryPath([point(1, 0.1), point(2, 0.5)], 100, 50);

    expect(path.startsWith("M")).toBe(true);
    expect(path).toContain("L");
  });

  it("does not divide by zero when every value is identical", () => {
    const path = trajectoryPath([point(1, 0.5), point(1, 0.5)], 100, 50);

    expect(path).not.toContain("NaN");
  });
});
