"use client";

/**
 * Run index — every completed Phase 1 inference, newest first.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchComparisons, fetchRuns } from "@/lib/api";
import type { NeuralComparisonSummary, RunListItem } from "@/types/neural";
import { AppHeader } from "@/components/layout/AppHeader";

export default function HomePage() {
  const [runs, setRuns] = useState<RunListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [comparisons, setComparisons] = useState<NeuralComparisonSummary[]>([]);

  useEffect(() => {
    fetchRuns()
      .then(setRuns)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : "Could not reach the visualization API."),
      );
    fetchComparisons().then(setComparisons).catch(() => undefined);
  }, []);

  return (
    <div className="min-h-screen">
      <AppHeader subtitle="Experiments" />
      <main className="mx-auto max-w-[1100px] px-6 py-10">
        <h1 className="text-2xl font-semibold tracking-tight text-white">
          Neural simulations
        </h1>
        <p className="mt-2 max-w-2xl text-[13px] leading-relaxed text-slate-400">
          Completed TRIBE v2 inference runs. Each contains predicted cortical fMRI responses
          for a stimulus — model predictions, not measurements from any viewer.
        </p>

        {error && (
          <div className="mt-8 rounded-xl border border-red-500/25 bg-red-500/[0.06] p-4">
            <div className="text-[13px] font-medium text-red-300">
              Visualization API unavailable
            </div>
            <p className="mt-1.5 text-[12px] text-slate-400">{error}</p>
            <code className="mt-3 block rounded bg-black/40 px-3 py-2 text-[11px] text-slate-400">
              uvicorn blackmirror.api.app:app --port 8000
            </code>
          </div>
        )}

        {!runs && !error && (
          <div className="mt-8 text-[13px] text-slate-500">Loading runs…</div>
        )}

        {runs && runs.length === 0 && (
          <div className="mt-8 rounded-xl border border-white/10 p-6 text-[13px] text-slate-400">
            No completed runs yet. Produce one with{" "}
            <code className="rounded bg-black/40 px-1.5 py-0.5 text-slate-300">
              blackmirror predict &lt;file&gt;
            </code>
            .
          </div>
        )}

        <div className="mt-8 space-y-2">
          {runs?.map((run) => (
            <Link
              key={run.run_id}
              href={`/experiments/${run.run_id}`}
              className="group flex items-center justify-between gap-6 rounded-xl border border-white/[0.08] bg-white/[0.02] px-5 py-4 transition-colors hover:border-white/20 hover:bg-white/[0.04]"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2.5">
                  <span className="truncate text-[13.5px] font-medium text-slate-100">
                    {run.stimulus_filename}
                  </span>
                  {run.is_synthetic && (
                    <span className="shrink-0 rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-wider text-amber-300">
                      Synthetic
                    </span>
                  )}
                </div>
                <div className="mt-1 font-mono text-[11px] text-slate-500">{run.run_id}</div>
              </div>
              <div className="flex shrink-0 items-center gap-6 text-[11px] text-slate-500">
                <Stat label="Samples" value={String(run.shape[0] ?? "—")} />
                <Stat label="Vertices" value={(run.shape[1] ?? 0).toLocaleString()} />
                <Stat label="Backend" value={run.backend} />
                <span className="text-slate-600 transition-colors group-hover:text-sky-300">
                  →
                </span>
              </div>
            </Link>
          ))}
        </div>

        <section className="mt-12">
            <div className="flex items-center justify-between"><h2 className="text-lg font-semibold text-white">Controlled comparisons</h2><Link href="/comparisons/new" className="rounded bg-sky-500 px-3 py-2 text-xs font-medium text-slate-950">New comparison</Link></div>
            <div className="mt-4 space-y-2">
              {comparisons.map((comparison) => (
                <Link key={comparison.comparison_id} href={`/comparisons/${comparison.comparison_id}`} className="flex justify-between rounded-xl border border-white/[0.08] bg-white/[0.02] px-5 py-4 text-xs hover:border-white/20">
                  <span className="font-mono text-slate-300">{comparison.comparison_id}</span>
                  <span className="text-slate-500">1 reference · {comparison.metadata.candidate_run_ids.length} candidate(s) →</span>
                </Link>
              ))}
            </div>
          </section>
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-right">
      <div className="text-[9.5px] uppercase tracking-[0.12em] text-slate-600">{label}</div>
      <div className="mt-0.5 font-mono tabular-nums text-slate-300">{value}</div>
    </div>
  );
}
