"use client";

/**
 * The structured explanation.
 *
 * Everything rendered here was computed by the backend from measured values.
 * Content shown beside an interval is labelled as having occurred *during* that
 * interval, never as having produced the score, and the caveats are rendered in
 * full rather than collapsed behind a disclosure.
 */

import { Label, Notice, Panel } from "@/components/scoring/Primitives";
import { formatDelta, formatPercent, formatRaw } from "@/lib/scoring";
import type { ScoreExplanation } from "@/types/scoring";

export function ExplanationPanel({ explanation }: { explanation: ScoreExplanation }) {
  return (
    <div className="space-y-4">
      <Panel title="Why this variant ranked highest">
        <ul className="space-y-2.5">
          {explanation.statements.map((statement, index) => (
            <li
              key={index}
              className={`text-[12px] leading-relaxed ${
                statement.startsWith("  ")
                  ? "pl-4 text-[var(--muted)]"
                  : "text-[var(--muted-strong)]"
              }`}
            >
              {statement.trim()}
            </li>
          ))}
        </ul>

        {explanation.objectives.length ? (
          <div className="mt-5 space-y-3 border-t border-[var(--line)] pt-4">
            {explanation.objectives.map((comparison) => (
              <div
                key={comparison.objective_id}
                className="grid grid-cols-[1fr_5rem_5rem] items-baseline gap-3"
              >
                <div>
                  <div className="text-[12px] text-[var(--foreground)]">
                    {comparison.objective_name}
                  </div>
                  <div className="tv-label mt-1">{comparison.target_label}</div>
                </div>
                <div className="tv-num text-right text-[12px] text-[var(--muted-strong)]">
                  {formatDelta(comparison.absolute_difference)}
                </div>
                <div className="tv-num text-right text-[11px] text-[var(--muted)]">
                  {comparison.relative_difference !== null
                    ? formatPercent(comparison.relative_difference)
                    : "—"}
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </Panel>

      {Object.keys(explanation.context).length ? (
        <Panel
          title="Content during the highest-contribution interval"
          aside={<span className="tv-label">co-occurrence, not cause</span>}
        >
          <div className="grid gap-5 md:grid-cols-2">
            {Object.entries(explanation.context).map(([variantId, context]) => (
              <div key={variantId} className="space-y-2">
                <Label>
                  {variantId} · {context.start_seconds.toFixed(2)}–
                  {context.end_seconds.toFixed(2)}s
                </Label>
                {context.visual_description ? (
                  <Row label="Visual" value={context.visual_description} />
                ) : null}
                {context.speech_text ? <Row label="Speech" value={context.speech_text} /> : null}
                {context.audio_description ? (
                  <Row label="Audio" value={context.audio_description} />
                ) : null}
                {context.objects.length ? (
                  <Row label="Objects" value={context.objects.join(", ")} />
                ) : null}
                {context.on_screen_text.length ? (
                  <Row label="On screen" value={context.on_screen_text.join(" · ")} />
                ) : null}
                {context.event_types.length ? (
                  <Row label="Events" value={context.event_types.join(", ")} />
                ) : null}
              </div>
            ))}
          </div>
        </Panel>
      ) : null}

      <Panel title="What this does not establish">
        <div className="space-y-3">
          {explanation.caveats.map((caveat, index) => (
            <Notice key={index}>{caveat}</Notice>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[4.5rem_1fr] gap-3">
      <div className="tv-label pt-0.5">{label}</div>
      <div className="text-[12px] leading-relaxed text-[var(--muted-strong)]">{value}</div>
    </div>
  );
}

export function ScoreSummaryRow({
  label,
  value,
}: {
  label: string;
  value: string | number | null;
}) {
  return (
    <div className="flex items-baseline justify-between border-b border-[var(--line)] py-2">
      <span className="tv-label">{label}</span>
      <span className="tv-num text-[12px] text-[var(--foreground)]">
        {typeof value === "number" ? formatRaw(value) : (value ?? "—")}
      </span>
    </div>
  );
}
