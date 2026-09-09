/** Phase 7 API client and the pure helpers the optimization UI renders from. */

import { ApiError } from "@/lib/api";
import type {
  Evidence,
  OptimizationInterval,
  OptimizationRecommendation,
  OptimizationRequest,
  OptimizationResult,
  ProposedVariantSpec,
  RecommendationReview,
} from "@/types/optimization";

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

const base = (experimentId: string) =>
  `/api/experiments/${encodeURIComponent(experimentId)}/optimization`;

export function createOptimization(
  experimentId: string,
  body: { request: OptimizationRequest; variant_runs?: Record<string, string> | null },
  signal?: AbortSignal,
): Promise<OptimizationResult> {
  return request<OptimizationResult>(
    `/api/experiments/${encodeURIComponent(experimentId)}/optimize`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal,
    },
  );
}

export function fetchOptimizationKeys(
  experimentId: string,
  signal?: AbortSignal,
): Promise<string[]> {
  return request<string[]>(base(experimentId), { signal });
}

export function fetchOptimization(
  experimentId: string,
  key: string,
  signal?: AbortSignal,
): Promise<OptimizationResult> {
  return request<OptimizationResult>(`${base(experimentId)}/${encodeURIComponent(key)}`, {
    signal,
  });
}

export function submitReview(
  experimentId: string,
  key: string,
  review: RecommendationReview,
): Promise<RecommendationReview[]> {
  return request<RecommendationReview[]>(
    `${base(experimentId)}/${encodeURIComponent(key)}/reviews`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ review }),
    },
  );
}

export function fetchReviews(
  experimentId: string,
  key: string,
  signal?: AbortSignal,
): Promise<RecommendationReview[]> {
  return request<RecommendationReview[]>(
    `${base(experimentId)}/${encodeURIComponent(key)}/reviews`,
    { signal },
  );
}

export function createCandidate(
  experimentId: string,
  key: string,
  proposedVariantId: string,
): Promise<ProposedVariantSpec> {
  return request<ProposedVariantSpec>(
    `${base(experimentId)}/${encodeURIComponent(key)}/candidate-specs`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ proposed_variant_id: proposedVariantId }),
    },
  );
}

/** Candidate specifications already approved for this optimization. */
export function fetchCandidates(
  experimentId: string,
  key: string,
  signal?: AbortSignal,
): Promise<ProposedVariantSpec[]> {
  return request<ProposedVariantSpec[]>(
    `${base(experimentId)}/${encodeURIComponent(key)}/candidate-specs`,
    { signal },
  );
}

/* -------------------------------------------------------------------------- */
/* Pure helpers                                                                */
/* -------------------------------------------------------------------------- */

/** Human wording for the only outcome claim Phase 7 may make. */
export function directionLabel(direction: string): string {
  switch (direction) {
    case "test_for_increase":
      return "Test for increase";
    case "test_for_decrease":
      return "Test for decrease";
    case "test_for_target_proximity":
      return "Test for target proximity";
    default:
      return direction;
  }
}

export function formatSeconds(value: number): string {
  return `${value.toFixed(2)}s`;
}

export function formatInterval(start: number, end: number): string {
  return `${start.toFixed(2)}–${end.toFixed(2)}s`;
}

/** Evidence cited by one recommendation, resolved against the run's evidence. */
export function evidenceFor(
  recommendation: OptimizationRecommendation,
  evidence: Evidence[],
): Evidence[] {
  const byId = new Map(evidence.map((item) => [item.evidence_id, item]));
  return recommendation.evidence_ids
    .map((id) => byId.get(id))
    .filter((item): item is Evidence => item !== undefined);
}

/**
 * Lay intervals out on a 0..duration track.
 *
 * Returns percentages so the timeline scales with its container rather than
 * assuming a pixel width.
 */
export function trackSegments(
  intervals: OptimizationInterval[],
  duration: number,
): { id: string; kind: "weak" | "strong"; leftPercent: number; widthPercent: number }[] {
  if (!(duration > 0)) return [];
  return intervals.map((interval) => {
    const left = Math.max(0, Math.min(1, interval.start_seconds / duration));
    const right = Math.max(0, Math.min(1, interval.end_seconds / duration));
    return {
      id: interval.interval_id,
      kind: interval.kind,
      leftPercent: left * 100,
      widthPercent: Math.max(0.5, (right - left) * 100),
    };
  });
}

/** The furthest point any interval reaches, used as the track length. */
export function trackDuration(result: OptimizationResult): number {
  const ends = [...result.weak_intervals, ...result.strong_intervals].map(
    (interval) => interval.end_seconds,
  );
  return ends.length ? Math.max(...ends) : 0;
}

/** Recommendations grouped by their review state, pending first. */
export function withReviewState(
  recommendations: OptimizationRecommendation[],
  reviews: RecommendationReview[],
): { recommendation: OptimizationRecommendation; state: RecommendationReview["state"] }[] {
  const byId = new Map(reviews.map((review) => [review.recommendation_id, review.state]));
  return recommendations.map((recommendation) => ({
    recommendation,
    state: byId.get(recommendation.recommendation_id) ?? "pending",
  }));
}

/** True when at least one recommendation has been approved or modified. */
export function hasApproval(reviews: RecommendationReview[]): boolean {
  return reviews.some((review) => review.state === "approved" || review.state === "modified");
}
