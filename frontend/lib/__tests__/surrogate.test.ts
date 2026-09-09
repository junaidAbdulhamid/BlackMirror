import { describe, expect, it } from "vitest";

import {
  errorOverTime,
  estimateLabel,
  formatEstimate,
  formatSigned,
  gatingMetrics,
  modeCounts,
  predictedVersusActual,
  trustSummary,
} from "@/lib/surrogate";
import type {
  AcquisitionRound,
  SurrogateDiagnostics,
  SurrogateMetrics,
  SurrogateSummary,
} from "@/types/surrogate";

function metrics(overrides: Partial<SurrogateMetrics> = {}): SurrogateMetrics {
  return {
    n: 10, mae: 0.02, rmse: 0.03, r2: 0.8, spearman: 0.75, kendall: 0.6,
    top_k_recall: 0.66, top_k: 3, scheme: "prefix", ...overrides,
  };
}

function diagnostics(overrides: Partial<SurrogateDiagnostics> = {}): SurrogateDiagnostics {
  return {
    state: "ready", trusted: true, trust_reason: "rank correlation 0.75",
    target_spread: 0.4,
    cross_validation: metrics({ scheme: "5_fold", spearman: 0.9 }),
    prefix_validation: metrics(),
    uncertainty: {
      n: 10, error_correlation: 0.4, mean_uncertainty: 0.05,
      mean_absolute_error: 0.02, coverage_68: 0.7,
      kind: "posterior_standard_deviation",
    },
    model: {
      model_type: "gaussian_process", model_version: 3,
      training_dataset_hash: "abc", training_dataset_size: 10,
      feature_names: ["gain"], hyperparameters: {}, training_seed: 0,
      supports_uncertainty: true, uncertainty_kind: "posterior_standard_deviation",
      trained_at: "2026-01-01T00:00:00Z",
    },
    ...overrides,
  };
}

function round(overrides: Partial<AcquisitionRound> = {}): AcquisitionRound {
  return {
    round_number: 1, mode: "surrogate", candidate_id: "c1", genome: { gain: 1 },
    predicted_score: 0.5, predicted_uncertainty: 0.05, ood_score: 0.1,
    acquisition: "expected_improvement", acquisition_value: 0.02,
    incumbent_before: 0.4, actual_score: 0.55, prediction_error: 0.05,
    pool_size: 2000, surrogate_state: "ready", surrogate_trusted: true,
    selection_reason: "predicted 0.5", ...overrides,
  };
}

function summary(overrides: Partial<SurrogateSummary> = {}): SurrogateSummary {
  return {
    search_id: "s1", training_samples: 12, target_spread: 0.4, excluded_records: 0,
    state: "ready", trusted: true, trust_reason: "ok", model: null,
    model_versions: 3, ...overrides,
  };
}

describe("formatting", () => {
  it("shows an absent estimate as an em dash, never as zero", () => {
    expect(formatEstimate(null)).toBe("—");
    expect(formatEstimate(0)).toBe("0.0000");
    expect(formatSigned(-0.01)).toBe("-0.0100");
  });

  it("always calls a prediction an estimate", () => {
    // The word is load-bearing: a bare number gets quoted as an outcome.
    expect(estimateLabel(0.5, 0.05)).toBe("estimate 0.5000 ± 0.0500");
    expect(estimateLabel(0.5, null)).toBe("estimate 0.5000");
    expect(estimateLabel(null, null)).toBe("no estimate");
  });
});

describe("trust", () => {
  it("says plainly when the surrogate is choosing candidates", () => {
    expect(trustSummary(summary()).trusted).toBe(true);
    expect(trustSummary(summary()).label).toContain("choosing");
  });

  it("distinguishes bootstrapping from having failed the gate", () => {
    expect(trustSummary(summary({ trusted: false, state: "bootstrapping" })).label).toContain(
      "Bootstrapping",
    );
    expect(trustSummary(summary({ trusted: false, state: "degraded" })).label).toContain(
      "falling back",
    );
  });
});

describe("which validation number to believe", () => {
  it("prefers the order-aware score when there is one", () => {
    const chosen = gatingMetrics(diagnostics());

    expect(chosen.metrics.scheme).toBe("prefix");
    expect(chosen.note).toContain("order-aware");
  });

  it("falls back to random folds and warns that they read high", () => {
    const chosen = gatingMetrics(
      diagnostics({ prefix_validation: metrics({ n: 0, spearman: null, scheme: "insufficient_data" }) }),
    );

    expect(chosen.metrics.scheme).toBe("5_fold");
    expect(chosen.note).toContain("reads higher");
  });
});

describe("diagnostics", () => {
  it("pairs predictions with measurements only where both exist", () => {
    const points = predictedVersusActual([
      round(),
      round({ round_number: 2, predicted_score: null, prediction_error: null }),
    ]);

    expect(points).toHaveLength(1);
    expect(points[0]!.candidateId).toBe("c1");
  });

  it("tracks running error so improvement is visible", () => {
    const series = errorOverTime([
      round({ round_number: 1, prediction_error: 0.1 }),
      round({ round_number: 2, prediction_error: -0.3 }),
    ]);

    expect(series).toEqual([
      { round: 1, mae: 0.1 },
      { round: 2, mae: 0.2 },
    ]);
  });

  it("counts how each round chose its candidate", () => {
    const counts = modeCounts([
      round({ mode: "bootstrap" }),
      round({ round_number: 2, mode: "bootstrap" }),
      round({ round_number: 3, mode: "fallback" }),
    ]);

    expect(counts).toEqual({ bootstrap: 2, fallback: 1 });
  });
});
