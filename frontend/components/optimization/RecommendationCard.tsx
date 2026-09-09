"use client";

/**
 * One recommendation, with its evidence and its approval controls.
 *
 * Rendering rules:
 *   - the expected direction is always "test for ...", never an outcome claim;
 *   - evidence confidence is labelled and its caveat is rendered verbatim, so
 *     nobody reads it as a probability of success;
 *   - cited evidence is shown with the measured numbers behind it;
 *   - nothing becomes a candidate without an explicit human decision.
 */

import { useState } from "react";

import { Label, Notice } from "@/components/scoring/Primitives";
import { directionLabel, evidenceFor, formatInterval } from "@/lib/optimization";
import type {
  ApprovalState,
  Evidence,
  OptimizationRecommendation,
} from "@/types/optimization";

const STATE_LABEL: Record<ApprovalState, string> = {
  pending: "Awaiting review",
  approved: "Approved",
  modified: "Modified",
  rejected: "Rejected",
};

export function RecommendationCard({
  recommendation,
  evidence,
  state,
  rank,
  onReview,
}: {
  recommendation: OptimizationRecommendation;
  evidence: Evidence[];
  state: ApprovalState;
  rank: number;
  onReview: (state: ApprovalState, reason?: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");
  const cited = evidenceFor(recommendation, evidence);
  const confidence = recommendation.evidence_confidence;

  return (
    <article className="tv-panel">
      <div className="flex items-start justify-between gap-4 border-b border-[var(--line)] px-5 py-4">
        <div className="min-w-0">
          <Label>
            #{rank} · {formatInterval(
              recommendation.target_interval_start,
              recommendation.target_interval_end,
            )}
            {recommendation.is_bundle ? " · bundle" : ""}
          </Label>
          <h3 className="tv-display mt-2 text-[15px] leading-snug">{recommendation.title}</h3>
        </div>
        <div className="shrink-0 text-right">
          <div className="tv-num tv-display text-[1.5rem] leading-none">
            {confidence.value.toFixed(2)}
          </div>
          <Label className="mt-1">evidence conf.</Label>
        </div>
      </div>

      <div className="space-y-4 p-5">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Fact label="Direction" value={directionLabel(recommendation.expected_direction)} />
          <Fact label="Risk" value={recommendation.risk} />
          <Fact label="Edit cost" value={recommendation.edit_cost} />
          <Fact label="Priority" value={recommendation.priority.toFixed(3)} />
        </div>

        <p className="text-[12px] leading-relaxed text-[var(--muted-strong)]">
          {recommendation.rationale}
        </p>

        <div>
          <Label>Cited evidence · {cited.length}</Label>
          <ul className="mt-2 space-y-2">
            {cited.map((item) => (
              <li key={item.evidence_id} className="grid grid-cols-[5.5rem_1fr] gap-3">
                <code className="font-mono text-[10px] text-[var(--muted)]">
                  {item.evidence_id}
                </code>
                <div className="text-[11px] leading-relaxed text-[var(--muted-strong)]">
                  {item.description}
                  {item.consistency !== null ? (
                    <span className="tv-label ml-2">
                      {(item.consistency * 100).toFixed(0)}% of {item.reference_count}{" "}
                      references agree
                    </span>
                  ) : item.reference_count === 1 ? (
                    <span className="tv-label ml-2">
                      1 reference — consistency undefined
                    </span>
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
        </div>

        <button className="tv-btn px-3 py-1.5" onClick={() => setOpen((value) => !value)}>
          {open ? "Hide detail" : "Show detail"}
        </button>

        {open ? (
          <div className="space-y-4 border-t border-[var(--line)] pt-4">
            <div>
              <Label>Proposed edits</Label>
              <ul className="mt-2 space-y-1.5">
                {recommendation.interventions.flatMap((intervention) =>
                  intervention.edit_instructions.map((instruction, index) => (
                    <li
                      key={`${intervention.intervention_id}-${index}`}
                      className="font-mono text-[11px] text-[var(--muted-strong)]"
                    >
                      {instruction.operation}
                      {instruction.feature ? ` · ${instruction.feature}` : ""}
                      {instruction.target_value !== null &&
                      instruction.target_value !== undefined
                        ? ` → ${instruction.target_value}`
                        : ""}
                      {instruction.to_time !== null && instruction.to_time !== undefined
                        ? ` @ ${instruction.to_time}s`
                        : ""}
                    </li>
                  )),
                )}
              </ul>
            </div>
            <div>
              <Label>How this confidence was computed</Label>
              <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Fact
                  label="Consistency"
                  value={confidence.reference_consistency.toFixed(2)}
                />
                <Fact
                  label="Difference"
                  value={confidence.feature_difference_strength.toFixed(2)}
                />
                <Fact label="Gap" value={confidence.objective_gap_strength.toFixed(2)} />
                <Fact label="Items" value={String(confidence.evidence_count)} />
              </div>
              <code className="mt-2 block font-mono text-[10px] leading-relaxed text-[var(--muted)]">
                {confidence.formula}
              </code>
            </div>
            <Notice>{confidence.caveat}</Notice>
            {recommendation.warnings.map((warning) => (
              <Notice key={warning}>{warning}</Notice>
            ))}
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-2 border-t border-[var(--line)] pt-4">
          <span className="tv-label mr-2">{STATE_LABEL[state]}</span>
          <button
            className="tv-btn tv-btn--primary px-3 py-1.5"
            disabled={state === "approved"}
            onClick={() => onReview("approved")}
          >
            Approve
          </button>
          <button
            className="tv-btn px-3 py-1.5"
            disabled={state === "rejected"}
            onClick={() => onReview("rejected", reason || undefined)}
          >
            Reject
          </button>
          <input
            className="tv-input min-w-0 flex-1 px-2.5 py-1.5"
            placeholder="Reason (optional, kept if rejected)"
            value={reason}
            onChange={(event) => setReason(event.target.value)}
          />
        </div>
      </div>
    </article>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Label>{label}</Label>
      <div className="tv-num mt-1 text-[12px] text-[var(--foreground)]">{value}</div>
    </div>
  );
}
