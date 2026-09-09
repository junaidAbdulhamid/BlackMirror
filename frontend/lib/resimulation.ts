/** Phase 8 API client and the pure helpers the re-simulation UI renders from. */

import { ApiError } from "@/lib/api";
import type {
  CreateResimulationBody,
  HypothesisVerdict,
  ObjectiveDelta,
  Outcome,
  ResimulationResult,
  ResimulationView,
  StageName,
} from "@/types/resimulation";
import { STAGES } from "@/types/resimulation";

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

export function createResimulation(
  experimentId: string,
  body: CreateResimulationBody,
  signal?: AbortSignal,
): Promise<ResimulationView> {
  return request<ResimulationView>(
    `/api/experiments/${encodeURIComponent(experimentId)}/resimulations`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal,
    },
  );
}

export function fetchResimulations(
  experimentId?: string,
  signal?: AbortSignal,
): Promise<ResimulationView[]> {
  const query = experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : "";
  return request<ResimulationView[]>(`/api/resimulations${query}`, { signal });
}

export function fetchResimulation(id: string, signal?: AbortSignal): Promise<ResimulationView> {
  return request<ResimulationView>(`/api/resimulations/${encodeURIComponent(id)}`, { signal });
}

export function resumeResimulation(id: string): Promise<ResimulationView> {
  return request<ResimulationView>(`/api/resimulations/${encodeURIComponent(id)}/resume`, {
    method: "POST",
  });
}

export function stopResimulation(
  id: string,
  reason: "cancelled" | "timeout" | "resource_limit" = "cancelled",
): Promise<ResimulationView> {
  return request<ResimulationView>(`/api/resimulations/${encodeURIComponent(id)}/stop`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ reason }),
  });
}

// --- pure helpers ---------------------------------------------------------

export function stageLabel(index: number): StageName | "Unknown" {
  return STAGES[index] ?? "Unknown";
}

/** Fraction of the pipeline that has durably completed. */
export function stageProgress(state: ResimulationResult): number {
  return Math.max(0, Math.min(1, state.current_stage / (STAGES.length - 1)));
}

/**
 * Whether the user should be offered a resume.
 *
 * A run that is `active` with no live worker is not progressing: its process
 * died without getting the chance to record that it had. That state looks
 * identical to a healthy run in the state file alone, so liveness decides.
 */
export function isResumable(view: ResimulationView): boolean {
  if (view.state.status === "completed") return false;
  return !view.worker_alive;
}

export function isRunning(view: ResimulationView): boolean {
  return view.worker_alive && view.state.status !== "completed";
}

/** A number as measured, or an em dash. Never a zero standing in for absent. */
export function formatValue(value: number | null, digits = 4): string {
  return value === null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

export function formatSigned(value: number | null, digits = 4): string {
  if (value === null || Number.isNaN(value)) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

/**
 * Why a delta counts as improved, in the terms the objective declared.
 *
 * A raw delta alone is ambiguous: under a minimize objective a fall is the
 * improvement, and under a target objective neither direction is, only
 * closeness. Stating the rule beside the number is what stops a reader
 * applying "bigger is better" to an objective that never said so.
 */
export function directionRule(delta: ObjectiveDelta): string {
  switch (delta.direction) {
    case "maximize":
      return "improved when the candidate's raw value is higher";
    case "minimize":
      return "improved when the candidate's raw value is lower";
    default:
      return "improved when the candidate's raw value is closer to the declared target";
  }
}

/** A one-line, non-causal statement of what was measured. */
export function describeOutcome(delta: ObjectiveDelta): string {
  if (delta.outcome === "inconclusive") {
    return "Not measurable: at least one side had no valid value for this objective.";
  }
  if (delta.outcome === "unchanged") {
    return "Within the declared tolerance, so no movement is claimed.";
  }
  const moved = delta.outcome === "improved" ? "toward" : "away from";
  return `The declared objective moved ${moved} its stated direction on model-predicted responses.`;
}

export function outcomeCounts(state: ResimulationResult): Record<Outcome, number> {
  const counts: Record<Outcome, number> = {
    improved: 0,
    worsened: 0,
    unchanged: 0,
    inconclusive: 0,
  };
  for (const delta of state.objective_deltas) counts[delta.outcome] += 1;
  return counts;
}

export function verdictCounts(state: ResimulationResult): Record<HypothesisVerdict, number> {
  const counts: Record<HypothesisVerdict, number> = { pass: 0, fail: 0, inconclusive: 0 };
  for (const outcome of state.hypothesis_outcomes) counts[outcome.verdict] += 1;
  return counts;
}

/**
 * The headline for a finished run, or null while nothing has been measured.
 *
 * Deliberately never says "improved" on its own: the objective id is carried
 * with it, because the sentence "it improved" is exactly the claim Phase 8 is
 * built not to make.
 */
export function headline(state: ResimulationResult): string | null {
  if (!state.objective_deltas.length) return null;
  const counts = outcomeCounts(state);
  const parts: string[] = [];
  if (counts.improved) parts.push(`${counts.improved} moved toward the declared direction`);
  if (counts.worsened) parts.push(`${counts.worsened} moved away`);
  if (counts.unchanged) parts.push(`${counts.unchanged} within tolerance`);
  if (counts.inconclusive) parts.push(`${counts.inconclusive} not measurable`);
  return parts.join(" · ");
}

/** The stage a failure occurred in, most recent first. */
export function latestFailure(state: ResimulationResult): string | null {
  const failure = state.failures[state.failures.length - 1];
  if (!failure) return null;
  return `${stageLabel(failure.stage)}: ${failure.error_type} — ${failure.message}`;
}

export function shortHash(value: string, length = 12): string {
  return value.slice(0, length);
}
