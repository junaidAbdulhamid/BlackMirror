import { describe, expect, it } from "vitest";

import {
  describeOutcome,
  directionRule,
  formatSigned,
  formatValue,
  headline,
  isResumable,
  isRunning,
  latestFailure,
  outcomeCounts,
  stageLabel,
  stageProgress,
  verdictCounts,
} from "@/lib/resimulation";
import type {
  HypothesisOutcome,
  ObjectiveDelta,
  ResimulationResult,
  ResimulationView,
} from "@/types/resimulation";

function delta(overrides: Partial<ObjectiveDelta> = {}): ObjectiveDelta {
  return {
    objective_id: "o1",
    direction: "maximize",
    parent_raw_value: 1,
    candidate_raw_value: 2,
    raw_delta: 1,
    parent_score: 0.2,
    candidate_score: 0.8,
    score_delta: 0.6,
    outcome: "improved",
    ...overrides,
  };
}

function hypothesis(overrides: Partial<HypothesisOutcome> = {}): HypothesisOutcome {
  return {
    hypothesis_id: "H1",
    expected_direction: "test_for_increase",
    verdict: "pass",
    measured_outcome: "improved",
    objective_id: "o1",
    note: "n",
    ...overrides,
  };
}

function state(overrides: Partial<ResimulationResult> = {}): ResimulationResult {
  return {
    schema_version: "1.0",
    resimulation_version: "1.0",
    cache_key: "a".repeat(64),
    status: "active",
    current_stage: 0,
    attempt_count: 0,
    candidate_run_id: null,
    artifacts: [],
    failures: [],
    objective_deltas: [],
    hypothesis_outcomes: [],
    stop_reason: null,
    requested_stop_reason: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    interpretation_notice: "notice",
    request: {
      resimulation_id: "loop-1",
      experiment_id: "exp",
      optimization_request_key: "abc12345",
      parent_run_id: "A",
      proposed_variant: {
        proposed_variant_id: "cand1",
        parent_variant_id: "A",
        hypothesis_ids: ["H1"],
        target_objective_id: "o1",
        expected_direction: "test_for_increase",
      },
      binding: {
        adapter_id: "user-supplied",
        adapter_version: "1.0",
        method: "user_supplied_media",
        tool: null,
        model: null,
        parameters: {},
        parameters_sha256: "b".repeat(64),
        source_path: "/s.mp4",
        source_sha256: "c".repeat(64),
        variant_path: "/v.mp4",
        variant_sha256: "d".repeat(64),
      },
      iteration_index: 1,
      max_iterations: 1,
      max_attempts: 3,
      outcome_tolerance: 1e-9,
    },
    ...overrides,
  };
}

function view(overrides: Partial<ResimulationView> = {}): ResimulationView {
  return { state: state(), worker_alive: false, worker_error: null, ...overrides };
}

describe("stage reporting", () => {
  it("names each stage of the durable pipeline", () => {
    expect(stageLabel(0)).toBe("Bound");
    expect(stageLabel(6)).toBe("Evaluated");
  });

  it("does not invent a name for a stage it does not know", () => {
    // A newer server could add a stage. Rendering "Unknown" is honest; picking
    // the nearest known name would mislabel real work.
    expect(stageLabel(99)).toBe("Unknown");
  });

  it("reports progress as the fraction of stages durably completed", () => {
    expect(stageProgress(state({ current_stage: 0 }))).toBe(0);
    expect(stageProgress(state({ current_stage: 3 }))).toBeCloseTo(0.5);
    expect(stageProgress(state({ current_stage: 6 }))).toBe(1);
  });
});

describe("worker liveness", () => {
  it("offers a resume when a run is active but nothing holds its lock", () => {
    // This is what an interrupted worker looks like: the state file still says
    // active because the process that would have recorded the failure died.
    expect(isResumable(view({ worker_alive: false }))).toBe(true);
    expect(isRunning(view({ worker_alive: false }))).toBe(false);
  });

  it("does not offer a resume while a worker is alive", () => {
    const running = view({ worker_alive: true });

    expect(isResumable(running)).toBe(false);
    expect(isRunning(running)).toBe(true);
  });

  it("never offers a resume for a completed run", () => {
    expect(
      isResumable(view({ state: state({ status: "completed" }), worker_alive: false })),
    ).toBe(false);
  });
});

describe("measured values", () => {
  it("shows an absent value as an em dash rather than zero", () => {
    expect(formatValue(null)).toBe("—");
    expect(formatSigned(null)).toBe("—");
    expect(formatValue(0)).toBe("0.0000");
  });

  it("signs a delta so direction is readable without comparing columns", () => {
    expect(formatSigned(0.5)).toBe("+0.5000");
    expect(formatSigned(-0.5)).toBe("-0.5000");
  });
});

describe("direction-aware reporting", () => {
  it("states the improvement rule the declared objective actually uses", () => {
    expect(directionRule(delta({ direction: "maximize" }))).toContain("higher");
    expect(directionRule(delta({ direction: "minimize" }))).toContain("lower");
    expect(directionRule(delta({ direction: "target" }))).toContain("closer");
  });

  it("describes an outcome without claiming a cause", () => {
    const text = describeOutcome(delta());

    expect(text).toContain("model-predicted");
    expect(text.toLowerCase()).not.toContain("because");
    expect(text.toLowerCase()).not.toContain("caused");
  });

  it("says plainly when nothing could be measured", () => {
    expect(describeOutcome(delta({ outcome: "inconclusive" }))).toContain("Not measurable");
    expect(describeOutcome(delta({ outcome: "unchanged" }))).toContain("tolerance");
  });
});

describe("summaries", () => {
  it("counts outcomes and verdicts separately", () => {
    const measured = state({
      objective_deltas: [delta(), delta({ objective_id: "o2", outcome: "worsened" })],
      hypothesis_outcomes: [hypothesis(), hypothesis({ hypothesis_id: "H2", verdict: "fail" })],
    });

    expect(outcomeCounts(measured)).toEqual({
      improved: 1,
      worsened: 1,
      unchanged: 0,
      inconclusive: 0,
    });
    expect(verdictCounts(measured)).toEqual({ pass: 1, fail: 1, inconclusive: 0 });
  });

  it("has no headline before anything is measured", () => {
    expect(headline(state())).toBeNull();
  });

  it("never summarises a result as a bare improvement", () => {
    // "1 improved" would read as a claim about the content. The wording ties
    // the movement to the direction that was declared in advance.
    const text = headline(state({ objective_deltas: [delta()] })) ?? "";

    expect(text).toBe("1 moved toward the declared direction");
  });

  it("reports the last failure with the stage it happened in", () => {
    const failed = state({
      status: "failed",
      failures: [
        {
          attempt: 1,
          stage: 1,
          error_type: "RuntimeError",
          message: "inference failed",
          occurred_at: "2026-01-01T00:00:00Z",
        },
      ],
    });

    expect(latestFailure(failed)).toBe("Inference: RuntimeError — inference failed");
  });

  it("has no failure line when nothing failed", () => {
    expect(latestFailure(state())).toBeNull();
  });
});
