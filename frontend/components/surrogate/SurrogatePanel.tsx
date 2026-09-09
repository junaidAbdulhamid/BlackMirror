"use client";

/**
 * Steps 59-61. What the surrogate learned, and how far to trust it.
 *
 * The panel is built around a refusal. A surrogate is a screening model, and
 * every number it produces is an estimate from a model fitted to past
 * evaluations. So the trust verdict sits at the top rather than in a footnote,
 * predicted scores are labelled as estimates wherever they appear, and the
 * validation figure shown is the order-aware one, because random folds let the
 * model see points that had not happened yet and read higher than reality.
 */

import { Bar, Label, Metric, Notice, Panel } from "@/components/scoring/Primitives";
import {
  errorOverTime,
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
  SurrogateSummary,
} from "@/types/surrogate";

export function SurrogatePanel({
  summary,
  diagnostics,
  rounds,
}: {
  summary: SurrogateSummary;
  diagnostics: SurrogateDiagnostics | null;
  rounds: AcquisitionRound[];
}) {
  const trust = trustSummary(summary);
  const gating = diagnostics ? gatingMetrics(diagnostics) : null;
  const modes = modeCounts(rounds);

  return (
    <div className="space-y-4">
      <Panel
        title="Surrogate optimizer"
        aside={<span className="tv-label">{trust.label}</span>}
      >
        <div className="grid gap-6 sm:grid-cols-3">
          <Metric
            value={String(summary.training_samples)}
            caption="real evaluations trained on"
            emphasis
          />
          <Metric
            value={summary.model?.model_type.replace(/_/g, " ") ?? "—"}
            caption={`version ${summary.model?.model_version ?? 0}`}
          />
          <Metric
            value={formatEstimate(summary.target_spread)}
            caption="label spread"
          />
        </div>

        {summary.trust_reason ? (
          <div className="mt-4 border-t border-[var(--line)] pt-3">
            <Notice>{summary.trust_reason}</Notice>
          </div>
        ) : null}

        <div className="mt-3">
          <Notice>
            Every score below is a surrogate estimate from a model fitted to past
            evaluations. It is not a measured objective, and it does not replace the
            real pipeline, which remains the only authority on what a candidate scores.
          </Notice>
        </div>
      </Panel>

      {gating ? (
        <Panel title="Validation" aside={<span className="tv-label">{gating.metrics.scheme}</span>}>
          <div className="grid gap-5 sm:grid-cols-4">
            <Metric value={formatEstimate(gating.metrics.mae)} caption="MAE" />
            <Metric value={formatEstimate(gating.metrics.rmse)} caption="RMSE" />
            <Metric
              value={formatEstimate(gating.metrics.spearman, 3)}
              caption="rank correlation"
            />
            <Metric
              value={
                gating.metrics.top_k_recall === null
                  ? "—"
                  : `${(gating.metrics.top_k_recall * 100).toFixed(0)}%`
              }
              caption={`top-${gating.metrics.top_k} recall`}
            />
          </div>

          <div className="mt-4 space-y-3 border-t border-[var(--line)] pt-3">
            <Notice>{gating.note}</Notice>
            <Notice>
              Rank correlation decides whether the surrogate is useful. Its job is to
              order candidates so the expensive evaluations go to promising ones, not
              to reproduce every score exactly.
            </Notice>
            {diagnostics ? (
              <div className="text-[11px] text-[var(--muted-strong)]">
                Uncertainty is a {diagnostics.uncertainty.kind.replace(/_/g, " ")}
                {diagnostics.uncertainty.error_correlation !== null
                  ? `, correlated ${diagnostics.uncertainty.error_correlation.toFixed(2)} with actual error`
                  : ", with too few points to check against error"}
                .
              </div>
            ) : null}
          </div>
        </Panel>
      ) : null}

      <Panel title="Predicted against actual">
        <PredictedVersusActual rounds={rounds} />
      </Panel>

      <Panel title={`Rounds · ${rounds.length}`}>
        <div className="mb-3 flex flex-wrap gap-4">
          {Object.entries(modes).map(([mode, count]) => (
            <div key={mode}>
              <div className="tv-num text-[13px] text-[var(--foreground)]">{count}</div>
              <div className="tv-label mt-0.5">{mode}</div>
            </div>
          ))}
        </div>

        <div className="max-h-80 space-y-2 overflow-y-auto">
          {rounds.map((row) => (
            <div key={row.round_number} className="border-t border-[var(--line)] pt-2">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-mono text-[11px] text-[var(--foreground)]">
                  {row.round_number}. {row.candidate_id}
                </span>
                <span className="tv-label">{row.mode}</span>
              </div>
              <div className="tv-num mt-1 flex flex-wrap gap-4 text-[11px] text-[var(--muted-strong)]">
                <span>
                  estimate {formatEstimate(row.predicted_score)}
                  {row.predicted_uncertainty === null
                    ? ""
                    : ` ± ${row.predicted_uncertainty.toFixed(4)}`}
                </span>
                <span>actual {formatEstimate(row.actual_score)}</span>
                <span>error {formatSigned(row.prediction_error)}</span>
              </div>
              {row.selection_reason ? (
                <div className="mt-1 text-[10.5px] text-[var(--muted)]">
                  {row.selection_reason}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </Panel>

      <ErrorOverTime rounds={rounds} />
    </div>
  );
}

/** Step 60. Scatter against the identity line; points on it were predicted well. */
function PredictedVersusActual({ rounds }: { rounds: AcquisitionRound[] }) {
  const points = predictedVersusActual(rounds);
  if (points.length === 0) {
    return (
      <p className="text-[12px] leading-relaxed text-[var(--muted)]">
        Nothing to compare yet. Points appear once the surrogate has proposed a
        candidate and the real pipeline has measured it.
      </p>
    );
  }

  const all = points.flatMap((p) => [p.predicted, p.actual]);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const size = 220;
  const px = (value: number) => ((value - min) / span) * size;
  const py = (value: number) => size - ((value - min) / span) * size;

  return (
    <div className="space-y-2">
      <svg
        viewBox={`-6 -6 ${size + 12} ${size + 12}`}
        width={size + 12}
        height={size + 12}
        role="img"
        aria-label="Surrogate estimate against measured objective"
      >
        <line
          x1={0} y1={size} x2={size} y2={0}
          stroke="var(--line-strong)" strokeWidth={1} strokeDasharray="3 3"
        />
        {points.map((point) => (
          <circle
            key={point.candidateId}
            cx={px(point.predicted)}
            cy={py(point.actual)}
            r={3}
            fill="var(--foreground)"
          >
            <title>
              {point.candidateId}: estimate {point.predicted.toFixed(4)}, measured{" "}
              {point.actual.toFixed(4)}
            </title>
          </circle>
        ))}
      </svg>
      <div className="tv-label">
        x: surrogate estimate · y: measured objective · dashed line is perfect agreement
      </div>
    </div>
  );
}

/** Step 61. Running error, to show whether more data is helping. */
function ErrorOverTime({ rounds }: { rounds: AcquisitionRound[] }) {
  const series = errorOverTime(rounds);
  if (series.length < 2) return null;
  const worst = Math.max(...series.map((point) => point.mae)) || 1;

  return (
    <Panel title="Prediction error as data arrives">
      <div className="space-y-2">
        {series.map((point) => (
          <div key={point.round} className="flex items-center gap-3">
            <span className="tv-label w-16 shrink-0">round {point.round}</span>
            <div className="flex-1">
              <Bar fraction={point.mae / worst} muted />
            </div>
            <span className="tv-num w-20 shrink-0 text-right text-[11px] text-[var(--muted-strong)]">
              {point.mae.toFixed(4)}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-3">
        <Label>Running mean absolute error</Label>
        <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--muted)]">
          Falling means the surrogate is learning the surface. Flat or rising means
          more data is not helping, which is a reason to widen the space or stop
          relying on the model.
        </p>
      </div>
    </Panel>
  );
}
