"use client";

/**
 * Phase 7 — neural content optimization agent.
 *
 * Produces candidate interventions to test. Nothing here edits media, runs
 * TRIBE, or claims an improvement: a recommendation becomes a candidate only
 * after an explicit human decision, and is validated only by Phase 8.
 */

import { useCallback, useEffect, useState } from "react";

import { GapTimeline } from "@/components/optimization/GapTimeline";
import { RecommendationCard } from "@/components/optimization/RecommendationCard";
import { Label, Notice, Panel } from "@/components/scoring/Primitives";
import {
  createCandidate,
  createOptimization,
  fetchReviews,
  hasApproval,
  submitReview,
  withReviewState,
} from "@/lib/optimization";
import { fetchScores } from "@/lib/scoring";
import type {
  ApprovalState,
  OptimizationResult,
  ProposedVariantSpec,
  RecommendationReview,
  ReferenceStrategy,
} from "@/types/optimization";
import type { ExperimentScoreResult } from "@/types/scoring";

export default function OptimizePage() {
  const [experimentId, setExperimentId] = useState("");
  const [scores, setScores] = useState<ExperimentScoreResult[]>([]);
  const [selectedHash, setSelectedHash] = useState("");
  const [sourceVariant, setSourceVariant] = useState("");
  const [strategy, setStrategy] = useState<ReferenceStrategy>("best_score");
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [optimizationKey, setOptimizationKey] = useState("");
  const [reviews, setReviews] = useState<RecommendationReview[]>([]);
  const [spec, setSpec] = useState<ProposedVariantSpec | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadScores = useCallback(async () => {
    if (!experimentId.trim()) return;
    setError(null);
    try {
      const found = await fetchScores(experimentId.trim());
      setScores(found);
      if (found.length) {
        setSelectedHash(found[0]!.metadata.objective_set_hash);
        setSourceVariant(found[0]!.ranking.at(-1) ?? found[0]!.variant_scores[0]!.variant_id);
      }
    } catch (caught) {
      setError((caught as { detail?: string }).detail ?? (caught as Error).message);
    }
  }, [experimentId]);

  const selected = scores.find((s) => s.metadata.objective_set_hash === selectedHash);

  async function runOptimization() {
    if (!selected) return;
    setBusy(true);
    setError(null);
    setSpec(null);
    try {
      const optimization = await createOptimization(experimentId.trim(), {
        request: {
          experiment_id: experimentId.trim(),
          source_variant_id: sourceVariant,
          objective_set_hash: selected.metadata.objective_set_hash,
          reference_strategy: strategy,
          strategy: "minimal_edit",
          max_recommendations: 5,
        },
      });
      setResult(optimization);
      const keys = await fetch(
        `/api/experiments/${encodeURIComponent(experimentId.trim())}/optimization`,
      ).then((r) => r.json() as Promise<string[]>);
      const key = keys[0] ?? "";
      setOptimizationKey(key);
      setReviews(key ? await fetchReviews(experimentId.trim(), key) : []);
    } catch (caught) {
      setError((caught as { detail?: string }).detail ?? (caught as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  async function review(recommendationId: string, state: ApprovalState, reason?: string) {
    if (!optimizationKey) return;
    try {
      setReviews(
        await submitReview(experimentId.trim(), optimizationKey, {
          recommendation_id: recommendationId,
          state,
          reason: reason ?? null,
        }),
      );
    } catch (caught) {
      setError((caught as { detail?: string }).detail ?? (caught as Error).message);
    }
  }

  async function buildCandidate() {
    if (!optimizationKey) return;
    setError(null);
    try {
      setSpec(
        await createCandidate(
          experimentId.trim(),
          optimizationKey,
          `cand-${Date.now().toString(36)}`,
        ),
      );
    } catch (caught) {
      setError((caught as { detail?: string }).detail ?? (caught as Error).message);
    }
  }

  useEffect(() => {
    setResult(null);
    setSpec(null);
  }, [experimentId]);

  const rows = result ? withReviewState(result.recommendations, reviews) : [];

  return (
    <main className="min-h-screen bg-black">
      <header className="border-b border-[var(--line)] px-8 py-10">
        <div className="mx-auto max-w-[1440px]">
          <Label>NeuroSplit · Phase 7</Label>
          <h1 className="tv-display mt-3 text-[2.5rem] leading-[1.05] sm:text-[3rem]">
            Neural Content Optimization Agent
          </h1>
          <p className="mt-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
            Locates where a variant underperforms its objective, measures how higher-scoring
            variants differ there, and proposes candidate interventions to test. Every output
            is a hypothesis — nothing is validated until a candidate is built and re-simulated.
          </p>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1440px] gap-4 px-8 py-8 lg:grid-cols-[340px_1fr]">
        <div className="space-y-4">
          <Panel title="Experiment">
            <div className="space-y-3">
              <input
                className="tv-input w-full px-2.5 py-2"
                placeholder="experiment id"
                value={experimentId}
                onChange={(event) => setExperimentId(event.target.value)}
              />
              <button className="tv-btn w-full px-4 py-2" onClick={loadScores}>
                Load scores
              </button>

              {scores.length ? (
                <>
                  <div className="space-y-1.5 pt-2">
                    <Label>Objective set</Label>
                    <select
                      className="tv-input w-full px-2.5 py-2"
                      value={selectedHash}
                      onChange={(event) => setSelectedHash(event.target.value)}
                    >
                      {scores.map((score) => (
                        <option
                          key={score.metadata.objective_set_hash}
                          value={score.metadata.objective_set_hash}
                        >
                          {score.objectives.map((o) => o.name).join(", ").slice(0, 44)}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1.5">
                    <Label>Variant to optimize</Label>
                    <select
                      className="tv-input w-full px-2.5 py-2"
                      value={sourceVariant}
                      onChange={(event) => setSourceVariant(event.target.value)}
                    >
                      {selected?.variant_scores.map((variant) => (
                        <option key={variant.variant_id} value={variant.variant_id}>
                          {variant.variant_id}
                          {variant.rank ? ` · rank ${variant.rank}` : ""}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1.5">
                    <Label>Learn from</Label>
                    <select
                      className="tv-input w-full px-2.5 py-2"
                      value={strategy}
                      onChange={(event) =>
                        setStrategy(event.target.value as ReferenceStrategy)
                      }
                    >
                      <option value="best_score">Higher-scoring variants</option>
                      <option value="baseline">Baseline</option>
                      <option value="pareto_front">Pareto front</option>
                    </select>
                  </div>

                  <button
                    className="tv-btn tv-btn--primary w-full px-4 py-2.5"
                    disabled={busy || !sourceVariant}
                    onClick={runOptimization}
                  >
                    {busy ? "Analysing…" : "Find candidate tests"}
                  </button>
                </>
              ) : null}
            </div>
          </Panel>

          {result ? (
            <Panel title="Objective gap">
              <div className="space-y-3">
                {result.gaps.map((gap) => (
                  <div key={gap.objective_id}>
                    <Label>{gap.objective_id}</Label>
                    <div className="tv-num mt-1.5 flex items-baseline gap-3 text-[12px]">
                      <span className="text-[var(--muted-strong)]">
                        {gap.source_value?.toFixed(4) ?? "—"}
                      </span>
                      <span className="text-[var(--muted)]">vs</span>
                      <span className="text-[var(--foreground)]">
                        {gap.reference_value?.toFixed(4) ?? "—"}
                      </span>
                    </div>
                    <div className="tv-label mt-1">
                      gap {gap.gap?.toFixed(4) ?? "—"} · {gap.formula}
                    </div>
                    {gap.note ? (
                      <p className="mt-1.5 text-[10.5px] text-[var(--muted)]">{gap.note}</p>
                    ) : null}
                  </div>
                ))}
                <div className="border-t border-[var(--line)] pt-3">
                  <Label>Reference variants</Label>
                  <div className="mt-1.5 text-[11px] text-[var(--muted-strong)]">
                    {result.reference_variant_ids.join(", ") || "none"}
                  </div>
                </div>
              </div>
            </Panel>
          ) : null}

          {result && hasApproval(reviews) ? (
            <Panel title="Candidate specification">
              <button className="tv-btn tv-btn--primary w-full px-4 py-2.5" onClick={buildCandidate}>
                Create candidate spec
              </button>
              {spec ? (
                <div className="mt-4 space-y-2">
                  <Label>{spec.proposed_variant_id}</Label>
                  <div className="text-[11px] text-[var(--muted-strong)]">
                    from {spec.parent_variant_id} · {spec.hypothesis_ids.length} hypothesis
                    {spec.hypothesis_ids.length === 1 ? "" : "es"}
                  </div>
                  <Notice>{spec.note}</Notice>
                </div>
              ) : null}
            </Panel>
          ) : null}

          {error ? (
            <Panel title="Request failed">
              <p className="text-[11px] leading-relaxed text-[var(--muted-strong)]">{error}</p>
            </Panel>
          ) : null}
        </div>

        <div className="space-y-4">
          {result ? (
            <>
              <Panel title="Where this variant underperforms">
                <GapTimeline result={result} />
              </Panel>

              <div className="space-y-4">
                {rows.map(({ recommendation, state }, index) => (
                  <RecommendationCard
                    key={recommendation.recommendation_id}
                    recommendation={recommendation}
                    evidence={result.evidence}
                    state={state}
                    rank={index + 1}
                    onReview={(next, reason) =>
                      review(recommendation.recommendation_id, next, reason)
                    }
                  />
                ))}
              </div>

              {result.rejected.length ? (
                <Panel title={`Rejected by validation · ${result.rejected.length}`}>
                  <div className="space-y-2">
                    {result.rejected.map((item) => (
                      <div key={item.recommendation_id} className="text-[11px]">
                        <code className="font-mono text-[10px] text-[var(--muted)]">
                          {item.recommendation_id}
                        </code>
                        <span className="ml-2 text-[var(--muted-strong)]">{item.reason}</span>
                      </div>
                    ))}
                  </div>
                </Panel>
              ) : null}

              <Panel title="What this does not establish">
                <div className="space-y-3">
                  <Notice>{result.metadata.interpretation_notice}</Notice>
                  {result.warnings.map((warning) => (
                    <Notice key={warning}>{warning}</Notice>
                  ))}
                </div>
              </Panel>
            </>
          ) : (
            <Panel>
              <div className="py-20 text-center">
                <Label>No optimization yet</Label>
                <p className="mx-auto mt-3 max-w-md text-[12px] leading-relaxed text-[var(--muted)]">
                  Load a scored experiment and choose a variant. Optimization reads saved
                  scores and content analyses; it never runs inference or edits media.
                </p>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </main>
  );
}
