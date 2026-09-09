/** Phase 9 API client and the pure helpers the search UI renders from. */

import { ApiError } from "@/lib/api";
import type {
  BudgetView,
  CandidateResult,
  LineageStep,
  ParetoFront,
  Population,
  SearchEvent,
  SearchMetrics,
  SearchReport,
  SearchSummary,
  TrajectoryPoint,
} from "@/types/search";

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

const base = (id: string) => `/api/search/${encodeURIComponent(id)}`;

export const fetchSearches = (signal?: AbortSignal) =>
  request<SearchSummary[]>("/api/search", { signal });
export const fetchSearch = (id: string, signal?: AbortSignal) =>
  request<SearchSummary>(base(id), { signal });
export const fetchTrajectory = (id: string, signal?: AbortSignal) =>
  request<TrajectoryPoint[]>(`${base(id)}/trajectory`, { signal });
export const fetchBudget = (id: string, signal?: AbortSignal) =>
  request<BudgetView>(`${base(id)}/budget`, { signal });
export const fetchPopulation = (id: string, signal?: AbortSignal) =>
  request<Population[]>(`${base(id)}/population`, { signal });
export const fetchResults = (id: string, signal?: AbortSignal) =>
  request<CandidateResult[]>(`${base(id)}/results`, { signal });
export const fetchEvents = (id: string, signal?: AbortSignal) =>
  request<SearchEvent[]>(`${base(id)}/events`, { signal });
export const fetchReport = (id: string, signal?: AbortSignal) =>
  request<SearchReport>(`${base(id)}/report`, { signal });
export const fetchPareto = (id: string, signal?: AbortSignal) =>
  request<ParetoFront>(`${base(id)}/pareto`, { signal });

const post = <T>(url: string, body?: unknown) =>
  request<T>(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const pauseSearch = (id: string) => post<SearchSummary>(`${base(id)}/pause`);
export const resumeSearch = (id: string) => post<SearchSummary>(`${base(id)}/resume`);
export const stopSearch = (id: string) => post<SearchSummary>(`${base(id)}/stop`);
export const pinCandidate = (id: string, candidateId: string) =>
  post<{ pinned: string[] }>(`${base(id)}/pin`, { candidate_id: candidateId });
export const eliminateCandidate = (id: string, candidateId: string) =>
  post<{ eliminated: string[] }>(`${base(id)}/eliminate`, { candidate_id: candidateId });

// --- pure helpers ---------------------------------------------------------

/** A measured number, or an em dash. Never a zero standing in for absent. */
export function formatFitness(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : value.toFixed(digits);
}

export function formatSigned(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 90) return `${seconds.toFixed(0)}s`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

/** Fraction of a budget dimension consumed, or null when it is unlimited. */
export function budgetFraction(used: number, limit: number | null): number | null {
  if (limit === null || limit <= 0) return null;
  return Math.max(0, Math.min(1, used / limit));
}

/**
 * The headline claim, stated at the strength the evidence supports.
 *
 * A search that improved on the root by less than the pipeline's own
 * sensitivity has not been shown to have improved anything, so the wording
 * changes rather than the number being presented the same way regardless.
 */
export function improvementVerdict(metrics: SearchMetrics): {
  label: string;
  resolvable: boolean | null;
} {
  const gain = metrics.absolute_improvement;
  if (gain === null || gain === undefined) {
    return { label: "No improvement measured against the root", resolvable: null };
  }
  if (metrics.improvement_is_resolvable === false) {
    return {
      label: "Inside the noise floor — not distinguishable from an implementation detail",
      resolvable: false,
    };
  }
  return {
    label: gain > 0 ? "Above the measured noise floor" : "No gain over the root",
    resolvable: metrics.improvement_is_resolvable ?? null,
  };
}

/** Points for the best-so-far plot, scaled into a unit box. */
export function trajectoryPath(
  points: TrajectoryPoint[],
  width: number,
  height: number,
): string {
  if (points.length === 0) return "";
  const xs = points.map((p) => p.evaluation_number);
  const ys = points.map((p) => p.best_fitness);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  return points
    .map((point, index) => {
      const x = ((point.evaluation_number - minX) / spanX) * width;
      const y = height - ((point.best_fitness - minY) / spanY) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

/** Candidates grouped by generation, for the tree and the population view. */
export function byGeneration(results: CandidateResult[]): Map<number, CandidateResult[]> {
  const out = new Map<number, CandidateResult[]>();
  for (const result of results) {
    const generation = result.genome.generation;
    const bucket = out.get(generation) ?? [];
    bucket.push(result);
    out.set(generation, bucket);
  }
  for (const bucket of out.values()) {
    bucket.sort((a, b) => (b.scalar_fitness ?? -Infinity) - (a.scalar_fitness ?? -Infinity));
  }
  return out;
}

/** Why a candidate is or is not eligible to be the answer. */
export function candidateStanding(result: CandidateResult): {
  label: string;
  eligible: boolean;
} {
  if (result.status === "rejected") return { label: "Rejected before evaluation", eligible: false };
  if (result.status === "evaluation_failed") return { label: "Evaluation failed", eligible: false };
  if (result.status === "not_measurable") return { label: "Not measurable", eligible: false };
  if (!result.feasibility.feasible) {
    return { label: "Disqualified by a guardrail", eligible: false };
  }
  return { label: "Eligible", eligible: true };
}

export function eliminationFor(
  populations: Population[],
  candidateId: string,
): Elimination | null {
  for (const population of populations) {
    const found = population.eliminated.find((item) => item.candidate_id === candidateId);
    if (found) return found;
  }
  return null;
}

type Elimination = Population["eliminated"][number];

export function lineageLabel(step: LineageStep): string {
  const values = Object.entries(step.genome)
    .map(([name, value]) => `${name}=${value}`)
    .join(", ");
  return `${step.candidate_id} · ${values}`;
}
