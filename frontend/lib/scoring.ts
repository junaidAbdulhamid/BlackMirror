/** Phase 6 API client and the pure helpers the scorecard renders from. */

import { ApiError } from "@/lib/api";
import type {
  ExperimentScoreResult,
  MetricCatalog,
  NeuralObjective,
  ObjectiveEvaluation,
  ScoreExplanation,
  VariantScore,
} from "@/types/scoring";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail: string | undefined;
    try {
      detail = ((await response.json()) as { detail?: string }).detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(`Request failed (${response.status})`, response.status, detail);
  }
  return (await response.json()) as T;
}

export function fetchMetricCatalog(signal?: AbortSignal): Promise<MetricCatalog> {
  return request<MetricCatalog>(`/api/objectives/metrics`, { signal });
}

export function fetchExperiments(signal?: AbortSignal): Promise<string[]> {
  return request<string[]>(`/api/experiments`, { signal });
}

export function fetchScores(
  experimentId: string,
  signal?: AbortSignal,
): Promise<ExperimentScoreResult[]> {
  return request<ExperimentScoreResult[]>(
    `/api/experiments/${encodeURIComponent(experimentId)}/scores`,
    { signal },
  );
}

export function fetchExplanation(
  experimentId: string,
  objectiveSetHash: string,
  signal?: AbortSignal,
): Promise<ScoreExplanation> {
  return request<ScoreExplanation>(
    `/api/experiments/${encodeURIComponent(experimentId)}/scores/` +
      `${encodeURIComponent(objectiveSetHash)}/explanation`,
    { signal },
  );
}

export function createScore(
  experimentId: string,
  body: {
    run_ids: string[];
    objectives: NeuralObjective[];
    baseline_run_id?: string | null;
    with_networks?: boolean;
    reuse_cache?: boolean;
  },
  signal?: AbortSignal,
): Promise<ExperimentScoreResult> {
  return request<ExperimentScoreResult>(
    `/api/experiments/${encodeURIComponent(experimentId)}/score`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal,
    },
  );
}

/* -------------------------------------------------------------------------- */
/* Pure helpers                                                                */
/* -------------------------------------------------------------------------- */

/** Format a model-unit value. Raw values are small, so precision matters. */
export function formatRaw(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

/** Signed formatting, for differences where direction is the point. */
export function formatDelta(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}

/**
 * How far an objective's realised influence drifted from its stated weight.
 *
 * A weighted sum only honours its weights when the objectives share a scale.
 * Measured on this codebase, a declared 50/50 split behaved as 5/95, so the UI
 * surfaces the gap rather than printing the stated weight as though it applied.
 */
export function weightDrift(
  result: ExperimentScoreResult,
): { objectiveId: string; stated: number; effective: number; drift: number }[] {
  const total = result.objectives.reduce((sum, o) => sum + o.weight, 0) || 1;
  return result.objectives.map((objective) => {
    const stated = objective.weight / total;
    const effective = result.effective_weights[objective.objective_id] ?? 0;
    return {
      objectiveId: objective.objective_id,
      stated,
      effective,
      drift: effective - stated,
    };
  });
}

/** Per-objective rows for the comparison matrix, variants as columns. */
export function objectiveMatrix(result: ExperimentScoreResult): {
  objective: NeuralObjective;
  cells: { variantId: string; evaluation: ObjectiveEvaluation | null }[];
  leaderId: string | null;
}[] {
  return result.objectives.map((objective) => {
    const cells = result.variant_scores.map((variant) => ({
      variantId: variant.variant_id,
      evaluation:
        variant.objective_scores.find((e) => e.objective_id === objective.objective_id) ?? null,
    }));
    let leaderId: string | null = null;
    let best = -Infinity;
    for (const cell of cells) {
      const score = cell.evaluation?.score;
      if (score !== null && score !== undefined && score > best) {
        best = score;
        leaderId = cell.variantId;
      }
    }
    return { objective, cells, leaderId };
  });
}

/** True when different variants lead different objectives. */
export function hasObjectiveConflict(result: ExperimentScoreResult): boolean {
  const leaders = new Set(
    objectiveMatrix(result)
      .map((row) => row.leaderId)
      .filter((id): id is string => id !== null),
  );
  return leaders.size > 1;
}

export function rankedVariants(result: ExperimentScoreResult): VariantScore[] {
  return [...result.variant_scores].sort((a, b) => {
    if (a.rank === null) return 1;
    if (b.rank === null) return -1;
    return a.rank - b.rank;
  });
}

/**
 * Polyline points for a temporal-contribution sparkline.
 *
 * Contributions can be negative, so the baseline sits where zero falls rather
 * than at the bottom of the box; drawing them clamped would hide the sign.
 */
export function contributionPoints(
  times: number[],
  contributions: number[],
  width = 640,
  height = 96,
): { points: string; zeroY: number } {
  if (!times.length || times.length !== contributions.length) {
    return { points: "", zeroY: height / 2 };
  }
  let min = Math.min(...contributions, 0);
  let max = Math.max(...contributions, 0);
  if (max === min) {
    max = min + 1;
  }
  const spanT = times[times.length - 1]! - times[0]! || 1;
  const y = (value: number) => height - ((value - min) / (max - min)) * height;
  const points = times
    .map((t, i) => `${((t - times[0]!) / spanT) * width},${y(contributions[i]!)}`)
    .join(" ");
  return { points, zeroY: y(0) };
}
