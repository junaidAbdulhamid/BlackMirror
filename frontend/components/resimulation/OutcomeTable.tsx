"use client";

/**
 * What actually moved, per declared objective.
 *
 * Raw values lead and scores follow, for the same reason Phase 6 orders them
 * that way: normalization can turn a negligible raw difference into a decisive
 * looking score. The direction rule is printed on every row so that a fall
 * under a minimize objective is not read as a regression.
 */

import { Label } from "@/components/scoring/Primitives";
import { describeOutcome, directionRule, formatSigned, formatValue } from "@/lib/resimulation";
import type { ObjectiveDelta } from "@/types/resimulation";

const OUTCOME_LABEL: Record<ObjectiveDelta["outcome"], string> = {
  improved: "Moved toward",
  worsened: "Moved away",
  unchanged: "Within tolerance",
  inconclusive: "Not measurable",
};

export function OutcomeTable({
  deltas,
  parentRunId,
  candidateRunId,
}: {
  deltas: ObjectiveDelta[];
  parentRunId: string;
  candidateRunId: string | null;
}) {
  if (!deltas.length) {
    return (
      <p className="text-[12px] leading-relaxed text-[var(--muted)]">
        Nothing measured yet. Objective deltas are written only after the comparison stage
        completes.
      </p>
    );
  }

  return (
    <div className="space-y-5">
      {deltas.map((delta) => (
        <div key={delta.objective_id} className="space-y-2">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <Label>{delta.objective_id}</Label>
            <span className="tv-label text-[var(--muted-strong)]">
              {OUTCOME_LABEL[delta.outcome]}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 border-t border-[var(--line)] pt-2.5">
            <Cell caption={`parent · ${parentRunId}`} value={formatValue(delta.parent_raw_value)} />
            <Cell
              caption={`candidate · ${candidateRunId ?? "pending"}`}
              value={formatValue(delta.candidate_raw_value)}
            />
            <Cell caption="raw delta" value={formatSigned(delta.raw_delta)} emphasis />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <Cell caption="parent score" value={formatValue(delta.parent_score)} muted />
            <Cell caption="candidate score" value={formatValue(delta.candidate_score)} muted />
            <Cell caption="score delta" value={formatSigned(delta.score_delta)} muted />
          </div>

          <p className="text-[11px] leading-relaxed text-[var(--muted)]">
            {delta.direction}, {directionRule(delta)}. {describeOutcome(delta)}
          </p>
        </div>
      ))}
    </div>
  );
}

function Cell({
  caption,
  value,
  emphasis = false,
  muted = false,
}: {
  caption: string;
  value: string;
  emphasis?: boolean;
  muted?: boolean;
}) {
  return (
    <div>
      <div
        className={`tv-num ${emphasis ? "tv-display text-[1.125rem]" : "text-[12px]"} ${
          muted ? "text-[var(--muted-strong)]" : "text-[var(--foreground)]"
        }`}
      >
        {value}
      </div>
      <div className="tv-label mt-1 truncate" title={caption}>
        {caption}
      </div>
    </div>
  );
}
