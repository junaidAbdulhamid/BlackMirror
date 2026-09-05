"use client";

/**
 * The Phase 6 scorecard.
 *
 * Rendering rules that are not negotiable:
 *   - a raw value appears next to every score, because normalization can make a
 *     0.25% difference look decisive;
 *   - per-objective values stay visible even when a composite exists, so a
 *     conflict between objectives cannot be hidden behind one number;
 *   - stated weights are shown against the weights that actually applied;
 *   - ranking language says "ranked highest for this objective", never "better".
 */

import { Bar, Label, Metric, Notice, Panel } from "@/components/scoring/Primitives";
import {
  contributionPoints,
  formatDelta,
  formatPercent,
  formatRaw,
  hasObjectiveConflict,
  objectiveMatrix,
  rankedVariants,
  weightDrift,
} from "@/lib/scoring";
import type { ExperimentScoreResult } from "@/types/scoring";

export function Scorecard({ result }: { result: ExperimentScoreResult }) {
  const ranked = rankedVariants(result);
  const matrix = objectiveMatrix(result);
  const conflict = hasObjectiveConflict(result);
  const drift = weightDrift(result);
  const leader = ranked[0];
  const runnerUp = ranked[1];
  const close = result.ranking_margin !== null && Math.abs(result.ranking_margin) < 0.01;

  return (
    <div className="space-y-4">
      <Panel
        title="Neural objective"
        aside={
          <span className="tv-label">
            {result.objectives.length} objective{result.objectives.length === 1 ? "" : "s"} ·
            scoring v{result.metadata.scoring_version}
          </span>
        }
      >
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {ranked.slice(0, 4).map((variant) => (
            <div key={variant.variant_id}>
              <Label>
                {variant.rank !== null ? `#${variant.rank} · ` : ""}
                {variant.variant_id}
              </Label>
              <div className="mt-2">
                <Metric
                  value={formatRaw(variant.total_score)}
                  caption="composite score"
                  emphasis
                />
              </div>
              {variant.baseline_delta !== null ? (
                <div className="tv-num mt-2 text-[11px] text-[var(--muted-strong)]">
                  {formatDelta(variant.baseline_delta)} vs baseline
                  {variant.baseline_relative_delta !== null
                    ? ` · ${formatPercent(variant.baseline_relative_delta)}`
                    : ""}
                </div>
              ) : null}
            </div>
          ))}
        </div>

        {leader && runnerUp ? (
          <div className="mt-6 border-t border-[var(--line)] pt-4">
            <p className="text-[12px] leading-relaxed text-[var(--muted-strong)]">
              <span className="text-[var(--foreground)]">{leader.variant_id}</span> ranked
              highest for this selected neural-response objective set, by{" "}
              <span className="tv-num">{formatRaw(result.ranking_margin)}</span>.
            </p>
            {close ? (
              <p className="mt-2 text-[11px] text-[var(--muted)]">
                Scores are very close under this objective. The ordering should not be treated
                as a meaningful difference between the variants.
              </p>
            ) : null}
          </div>
        ) : null}
      </Panel>

      <Panel title="Per-objective values">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-[12px]">
            <thead>
              <tr>
                <th className="tv-label px-2 pb-3 text-left font-medium">Objective</th>
                <th className="tv-label px-2 pb-3 text-right font-medium">Weight</th>
                {ranked.map((variant) => (
                  <th
                    key={variant.variant_id}
                    className="tv-label px-2 pb-3 text-right font-medium"
                  >
                    {variant.variant_id}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map(({ objective, cells, leaderId }) => {
                const applied = drift.find((d) => d.objectiveId === objective.objective_id);
                return (
                  <tr key={objective.objective_id} className="border-t border-[var(--line)]">
                    <td className="px-2 py-3 align-top">
                      <div className="text-[var(--foreground)]">{objective.name}</div>
                      <div className="tv-label mt-1">
                        {objective.metric.replaceAll("_", " ").toLowerCase()} ·{" "}
                        {objective.direction}
                      </div>
                    </td>
                    <td className="tv-num px-2 py-3 text-right align-top text-[var(--muted-strong)]">
                      {applied ? `${(applied.stated * 100).toFixed(0)}%` : "—"}
                      {applied && Math.abs(applied.drift) > 0.15 ? (
                        <div className="tv-label mt-1 text-right">
                          acted as {(applied.effective * 100).toFixed(0)}%
                        </div>
                      ) : null}
                    </td>
                    {cells.map(({ variantId, evaluation }) => (
                      <td key={variantId} className="px-2 py-3 text-right align-top">
                        <div
                          className={`tv-num ${
                            variantId === leaderId
                              ? "text-[var(--foreground)]"
                              : "text-[var(--muted-strong)]"
                          }`}
                        >
                          {formatRaw(evaluation?.raw_value)}
                        </div>
                        <div className="tv-label mt-1">
                          {evaluation?.window
                            ? `${evaluation.window.start_seconds.toFixed(1)}–${evaluation.window.end_seconds.toFixed(1)}s · n=${evaluation.window.sample_count}`
                            : "unresolved"}
                        </div>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <p className="mt-4 text-[10.5px] leading-relaxed text-[var(--muted)]">
          Values shown are raw metrics in model units, over each variant&rsquo;s own resolved
          window. Composite scores are derived from these; the raw numbers are the measurement.
        </p>

        {conflict ? (
          <div className="mt-4">
            <Notice>
              Different variants lead different objectives, so the composite ordering depends on
              the weights that were chosen. The per-objective values above are the reliable
              comparison.
            </Notice>
          </div>
        ) : null}
      </Panel>

      {result.pareto ? (
        <Panel title="Pareto analysis" aside={<span className="tv-label">no weights used</span>}>
          <div className="grid gap-6 sm:grid-cols-2">
            <div>
              <Label>Non-dominated</Label>
              <div className="mt-2 space-y-1">
                {result.pareto.non_dominated.map((id) => (
                  <div key={id} className="text-[12px] text-[var(--foreground)]">
                    {id}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <Label>Dominated</Label>
              <div className="mt-2 space-y-1">
                {result.pareto.dominated.length ? (
                  result.pareto.dominated.map((id) => (
                    <div key={id} className="text-[12px] text-[var(--muted)]">
                      {id} · beaten by {result.pareto?.dominated_by[id]?.join(", ")}
                    </div>
                  ))
                ) : (
                  <div className="text-[12px] text-[var(--muted)]">none</div>
                )}
              </div>
            </div>
          </div>
          <p className="mt-4 text-[10.5px] leading-relaxed text-[var(--muted)]">
            A variant is dominated when another is at least as good on every objective and
            strictly better on at least one. This needs no weights, so it carries no opinion
            about tradeoffs.
          </p>
        </Panel>
      ) : null}

      <TemporalContributions result={result} />

      {result.sensitivity.length ? (
        <Panel title="Weight sensitivity">
          <div className="space-y-4">
            {result.sensitivity.map((analysis) => (
              <div key={analysis.swept_objective_id}>
                <div className="flex items-baseline justify-between">
                  <Label>{analysis.swept_objective_id}</Label>
                  <span className="tv-label">
                    {analysis.stable ? "ranking stable" : `flips at ${analysis.flip_weights.join(", ")}`}
                  </span>
                </div>
                <div className="mt-2 flex h-4 w-full overflow-hidden">
                  {analysis.points.map((point, index) => (
                    <div
                      key={index}
                      className="h-full flex-1"
                      title={`w=${point.weight} → ${point.winner_variant_id}`}
                      style={{
                        background:
                          point.winner_variant_id === result.ranking[0]
                            ? "var(--muted-strong)"
                            : "var(--line-strong)",
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[10.5px] leading-relaxed text-[var(--muted)]">
            Each strip sweeps that objective&rsquo;s weight from 0 to 1. A ranking that changes
            inside the sweep depends on the weighting, not only on the variants.
          </p>
        </Panel>
      ) : null}

      {result.warnings.length ? (
        <Panel title="Warnings">
          <div className="space-y-3">
            {result.warnings.map((warning, index) => (
              <Notice key={index}>{warning}</Notice>
            ))}
          </div>
        </Panel>
      ) : null}

      <p className="px-1 text-[10.5px] leading-relaxed text-[var(--muted)]">
        {result.metadata.interpretation_notice}
      </p>
    </div>
  );
}

function TemporalContributions({ result }: { result: ExperimentScoreResult }) {
  const rows = result.variant_scores.flatMap((variant) =>
    variant.objective_scores
      .filter((evaluation) => evaluation.temporal_contribution !== null)
      .map((evaluation) => ({ variantId: variant.variant_id, evaluation })),
  );
  if (!rows.length) return null;

  return (
        <Panel title="Raw metric contribution over time">
      <div className="space-y-6">
        {rows.map(({ variantId, evaluation }) => {
          const contribution = evaluation.temporal_contribution!;
          const { points, zeroY } = contributionPoints(
            contribution.times,
            contribution.contributions,
          );
          return (
            <div key={`${variantId}-${evaluation.objective_id}`}>
              <div className="flex items-baseline justify-between">
                <Label>
                  {variantId} · {evaluation.objective_id}
                </Label>
                <span className="tv-num text-[11px] text-[var(--muted-strong)]">
                  peak {contribution.peak_time_seconds.toFixed(2)}s
                  {contribution.peak_fraction !== null
                    ? ` · ${(contribution.peak_fraction * 100).toFixed(0)}% of magnitude`
                    : ""}
                </span>
              </div>
              <svg
                viewBox="0 0 640 96"
                preserveAspectRatio="none"
                className="mt-2 h-16 w-full"
                role="img"
                aria-label={`Raw metric contribution over time for ${variantId}`}
              >
                <line
                  x1="0"
                  x2="640"
                  y1={zeroY}
                  y2={zeroY}
                  stroke="var(--line-strong)"
                  strokeWidth="1"
                />
                <polyline
                  points={points}
                  fill="none"
                  stroke="var(--foreground)"
                  strokeWidth="1.5"
                />
              </svg>
            </div>
          );
        })}
      </div>
      <p className="mt-4 text-[10.5px] leading-relaxed text-[var(--muted)]">
        Contributions sum exactly to the raw value, so a share of the curve is a share of the
        score. Metrics that are not sums over samples report no decomposition rather than an
        invented one.
      </p>
    </Panel>
  );
}

export function Leaderboard({ result }: { result: ExperimentScoreResult }) {
  const ranked = rankedVariants(result);
  const best = Math.max(...ranked.map((v) => v.total_score ?? 0), 0.0001);
  return (
    <Panel title="Variant leaderboard">
      <div className="space-y-3">
        {ranked.map((variant) => (
          <div key={variant.variant_id} className="grid grid-cols-[2rem_1fr_6rem_6rem] items-center gap-3">
            <div className="tv-num tv-label">{variant.rank ?? "—"}</div>
            <div>
              <div className="text-[12px] text-[var(--foreground)]">{variant.variant_id}</div>
              <div className="mt-1.5">
                <Bar
                  fraction={(variant.total_score ?? 0) / best}
                  muted={variant.variant_id !== result.ranking[0]}
                />
              </div>
            </div>
            <div className="tv-num text-right text-[12px] text-[var(--foreground)]">
              {formatRaw(variant.total_score)}
            </div>
            <div className="tv-num text-right text-[11px] text-[var(--muted)]">
              {variant.variant_id === result.baseline_variant_id
                ? "baseline"
                : formatDelta(variant.baseline_delta)}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-4 text-[10.5px] text-[var(--muted)]">
        Ranking for the selected objective set only.
      </p>
    </Panel>
  );
}
