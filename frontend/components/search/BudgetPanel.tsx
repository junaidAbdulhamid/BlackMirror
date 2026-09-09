"use client";

/**
 * Step 65's budget block. What has been spent, against what was allowed.
 *
 * An unlimited dimension shows its count with no bar rather than a full or
 * empty one, because a bar implies a ceiling and there isn't one.
 */

import { Bar, Label } from "@/components/scoring/Primitives";
import { budgetFraction, formatDuration } from "@/lib/search";
import type { BudgetView } from "@/types/search";

export function BudgetPanel({ view }: { view: BudgetView }) {
  const { budget, ledger } = view;
  const rows: Array<{ label: string; used: number; limit: number | null; text: string }> = [
    {
      label: "Candidates",
      used: ledger.candidates_generated,
      limit: budget.max_candidates,
      text: `${ledger.candidates_generated} / ${budget.max_candidates ?? "∞"}`,
    },
    {
      label: "Inference runs",
      used: ledger.tribe_runs,
      limit: budget.max_tribe_runs,
      text: `${ledger.tribe_runs} / ${budget.max_tribe_runs ?? "∞"}`,
    },
    {
      label: "Generations",
      used: ledger.generations_completed,
      limit: budget.max_generations,
      text: `${ledger.generations_completed} / ${budget.max_generations ?? "∞"}`,
    },
    {
      label: "Wall clock",
      used: ledger.wall_seconds,
      limit: budget.max_wall_seconds,
      text: `${formatDuration(ledger.wall_seconds)} / ${
        budget.max_wall_seconds ? formatDuration(budget.max_wall_seconds) : "∞"
      }`,
    },
  ];

  if (budget.cost_per_wall_hour > 0) {
    const spent = (ledger.wall_seconds / 3600) * budget.cost_per_wall_hour;
    rows.push({
      label: "Estimated cost",
      used: spent,
      limit: budget.max_cost,
      text: `${spent.toFixed(2)} / ${budget.max_cost?.toFixed(2) ?? "∞"}`,
    });
  }

  return (
    <div className="space-y-3.5">
      {rows.map((row) => {
        const fraction = budgetFraction(row.used, row.limit);
        return (
          <div key={row.label} className="space-y-1.5">
            <div className="flex items-baseline justify-between">
              <Label>{row.label}</Label>
              <span className="tv-num text-[11px] text-[var(--muted-strong)]">
                {row.text}
              </span>
            </div>
            {fraction === null ? null : <Bar fraction={fraction} />}
          </div>
        );
      })}

      <div className="border-t border-[var(--line)] pt-3">
        <Label>Reused from cache</Label>
        <div className="tv-num mt-1 text-[11px] text-[var(--muted-strong)]">
          {ledger.cache_hits} of {ledger.candidates_evaluated} evaluated
          {ledger.evaluation_failures > 0
            ? ` · ${ledger.evaluation_failures} failed`
            : ""}
        </div>
      </div>
    </div>
  );
}
