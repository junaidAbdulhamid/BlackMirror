"use client";

/**
 * Where the source underperforms, and where it already carries the objective.
 *
 * Strong intervals are drawn as prominently as weak ones on purpose: an
 * optimizer that only shows weakness invites edits that destroy what already
 * works, so the preserve regions are visible in the same glance.
 */

import { Label } from "@/components/scoring/Primitives";
import { formatInterval, trackDuration, trackSegments } from "@/lib/optimization";
import type { OptimizationResult } from "@/types/optimization";

export function GapTimeline({ result }: { result: OptimizationResult }) {
  const duration = trackDuration(result);
  const weak = trackSegments(result.weak_intervals, duration);
  const strong = trackSegments(result.strong_intervals, duration);
  if (!duration) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <Label>Objective contribution across the stimulus</Label>
        <span className="tv-label">0 – {duration.toFixed(1)}s</span>
      </div>

      <div className="relative h-8 w-full border border-[var(--line)] bg-[var(--surface-raised)]">
        {strong.map((segment) => (
          <div
            key={segment.id}
            className="absolute top-0 h-full bg-[var(--foreground)] opacity-70"
            style={{ left: `${segment.leftPercent}%`, width: `${segment.widthPercent}%` }}
            title="Carries the objective — preserve"
          />
        ))}
        {weak.map((segment) => (
          <div
            key={segment.id}
            className="absolute top-0 h-full border-x border-[var(--muted-strong)] bg-[var(--muted)] opacity-40"
            style={{ left: `${segment.leftPercent}%`, width: `${segment.widthPercent}%` }}
            title="Underperforming for this objective"
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-5">
        <LegendSwatch className="bg-[var(--foreground)] opacity-70" label="Strong — preserve" />
        <LegendSwatch className="bg-[var(--muted)] opacity-40" label="Weak — candidate" />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <IntervalList title="Weak intervals" intervals={result.weak_intervals} />
        <IntervalList title="Strong intervals" intervals={result.strong_intervals} />
      </div>
    </div>
  );
}

function LegendSwatch({ className, label }: { className: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`inline-block h-2.5 w-5 ${className}`} />
      <span className="tv-label">{label}</span>
    </div>
  );
}

function IntervalList({
  title,
  intervals,
}: {
  title: string;
  intervals: OptimizationResult["weak_intervals"];
}) {
  return (
    <div>
      <Label>{title}</Label>
      <ul className="mt-2 space-y-1.5">
        {intervals.length ? (
          intervals.map((interval) => (
            <li key={interval.interval_id} className="text-[11px] text-[var(--muted-strong)]">
              <span className="tv-num">
                {formatInterval(interval.start_seconds, interval.end_seconds)}
              </span>
              <span className="tv-label ml-2">
                {(interval.contribution_share * 100).toFixed(1)}% share · n=
                {interval.sample_count}
              </span>
            </li>
          ))
        ) : (
          <li className="text-[11px] text-[var(--muted)]">none detected</li>
        )}
      </ul>
    </div>
  );
}
