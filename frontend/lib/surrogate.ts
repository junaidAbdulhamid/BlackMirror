/** Phase 10 API client and the pure helpers the surrogate panel renders from. */

import { ApiError } from "@/lib/api";
import type {
  AcquisitionRound,
  SurrogateDiagnostics,
  SurrogateMetrics,
  SurrogateModelMetadata,
  SurrogateSummary,
} from "@/types/surrogate";

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

export const fetchSurrogate = (id: string, signal?: AbortSignal) =>
  request<SurrogateSummary>(`${base(id)}/surrogate`, { signal });
export const fetchSurrogateMetrics = (id: string, signal?: AbortSignal) =>
  request<SurrogateDiagnostics>(`${base(id)}/surrogate/metrics`, { signal });
export const fetchAcquisitionRounds = (id: string, signal?: AbortSignal) =>
  request<AcquisitionRound[]>(`${base(id)}/acquisition`, { signal });
export const fetchSurrogateModels = (id: string, signal?: AbortSignal) =>
  request<SurrogateModelMetadata[]>(`${base(id)}/surrogate/models`, { signal });

// --- pure helpers ---------------------------------------------------------

export function formatEstimate(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : value.toFixed(digits);
}

export function formatSigned(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

/**
 * Wording for a predicted score, which must never read as a measurement.
 *
 * The word "estimate" is load-bearing. A number shown without it will be
 * quoted as an outcome by someone who did not read the surrounding text.
 */
export function estimateLabel(
  predicted: number | null,
  uncertainty: number | null,
): string {
  if (predicted === null) return "no estimate";
  const band = uncertainty === null ? "" : ` ± ${uncertainty.toFixed(4)}`;
  return `estimate ${predicted.toFixed(4)}${band}`;
}

/** Which validation number a reader should judge the model by. */
export function gatingMetrics(diagnostics: SurrogateDiagnostics): {
  metrics: SurrogateMetrics;
  note: string;
} {
  const prefix = diagnostics.prefix_validation;
  if (prefix.n >= 3 && prefix.spearman !== null) {
    return {
      metrics: prefix,
      note: "order-aware: each point predicted from the ones before it",
    };
  }
  return {
    metrics: diagnostics.cross_validation,
    note: "random folds: reads higher than online use, because it can see later points",
  };
}

/** A plain sentence about whether the surrogate may choose candidates. */
export function trustSummary(summary: SurrogateSummary): {
  label: string;
  trusted: boolean;
} {
  if (summary.trusted === true) {
    return { label: "Trusted — choosing candidates", trusted: true };
  }
  if (summary.state === "bootstrapping") {
    return { label: "Bootstrapping — not yet choosing", trusted: false };
  }
  return { label: "Not trusted — falling back to search", trusted: false };
}

/** Points for the predicted-versus-actual plot, only where both exist. */
export function predictedVersusActual(
  rounds: AcquisitionRound[],
): Array<{ predicted: number; actual: number; candidateId: string }> {
  return rounds
    .filter((row) => row.predicted_score !== null && row.actual_score !== null)
    .map((row) => ({
      predicted: row.predicted_score as number,
      actual: row.actual_score as number,
      candidateId: row.candidate_id,
    }));
}

/** Running mean absolute error, to show whether the model improves with data. */
export function errorOverTime(
  rounds: AcquisitionRound[],
): Array<{ round: number; mae: number }> {
  const errors: number[] = [];
  const out: Array<{ round: number; mae: number }> = [];
  for (const row of rounds) {
    if (row.prediction_error === null) continue;
    errors.push(Math.abs(row.prediction_error));
    out.push({
      round: row.round_number,
      mae: errors.reduce((a, b) => a + b, 0) / errors.length,
    });
  }
  return out;
}

export function modeCounts(rounds: AcquisitionRound[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const row of rounds) counts[row.mode] = (counts[row.mode] ?? 0) + 1;
  return counts;
}
