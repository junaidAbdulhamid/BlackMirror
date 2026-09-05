"use client";

/**
 * Phase 6 — goal-conditioned scoring.
 *
 * Composed in the TRIBE v2 demo's visual language: pure-black ground, 2px
 * corners, wide-tracked uppercase labels and muted slate copy.
 */

import { useCallback, useEffect, useState } from "react";

import { ExplanationPanel } from "@/components/scoring/ExplanationPanel";
import { Label, Panel } from "@/components/scoring/Primitives";
import { ObjectiveBuilder } from "@/components/scoring/ObjectiveBuilder";
import { Leaderboard, Scorecard } from "@/components/scoring/Scorecard";
import { fetchRuns } from "@/lib/api";
import { createScore, fetchExplanation } from "@/lib/scoring";
import type { ExperimentScoreResult, NeuralObjective, ScoreExplanation } from "@/types/scoring";

const YEO = [
  "7Networks_1",
  "7Networks_2",
  "7Networks_3",
  "7Networks_4",
  "7Networks_5",
  "7Networks_6",
  "7Networks_7",
];

export default function ScoringPage() {
  const [runs, setRuns] = useState<{ run_id: string }[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [baseline, setBaseline] = useState<string>("");
  const [objectives, setObjectives] = useState<NeuralObjective[]>([]);
  const [regionNames, setRegionNames] = useState<string[]>([]);
  const [result, setResult] = useState<ExperimentScoreResult | null>(null);
  const [explanation, setExplanation] = useState<ScoreExplanation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchRuns(controller.signal)
      .then((items) => setRuns(items.map((item) => ({ run_id: item.run_id }))))
      .catch(() => setRuns([]));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selected.length) return;
    const controller = new AbortController();
    fetch(`/api/runs/${encodeURIComponent(selected[0]!)}/analytics`, {
      signal: controller.signal,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        const regions = payload?.atlas?.regions;
        if (Array.isArray(regions)) {
          setRegionNames(regions.map((region: { name: string }) => region.name));
        }
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [selected]);

  const toggleRun = useCallback((runId: string) => {
    setSelected((current) =>
      current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId],
    );
  }, []);

  async function runScoring() {
    setBusy(true);
    setError(null);
    setExplanation(null);
    const experimentId = `exp-${selected.length}v-${objectives.length}o`;
    try {
      const scored = await createScore(experimentId, {
        run_ids: selected,
        objectives,
        baseline_run_id: baseline || null,
        with_networks: objectives.some((o) => o.target.type === "network"),
        reuse_cache: false,
      });
      setResult(scored);
      try {
        setExplanation(
          await fetchExplanation(experimentId, scored.metadata.objective_set_hash),
        );
      } catch {
        setExplanation(null);
      }
    } catch (caught) {
      const detail = (caught as { detail?: string }).detail;
      setError(detail ?? (caught as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const ready = selected.length >= 1 && objectives.length >= 1;

  return (
    <main className="min-h-screen bg-black">
      <header className="border-b border-[var(--line)] px-8 py-10">
        <div className="mx-auto max-w-[1440px]">
          <Label>NeuroSplit · Phase 6</Label>
          <h1 className="tv-display mt-3 text-[2.5rem] leading-[1.05] sm:text-[3rem]">
            Goal-Conditioned Neural Scoring
          </h1>
          <p className="mt-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
            Define a measurable objective over predicted cortical responses, then score and rank
            variants against it. A score is the value of the function you chose — not a measure
            of persuasion, memory or effectiveness.
          </p>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1440px] gap-4 px-8 py-8 lg:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Panel title="Variants" aside={<span className="tv-label">{selected.length} selected</span>}>
            <div className="max-h-56 space-y-1.5 overflow-y-auto">
              {runs.map((run) => (
                <label
                  key={run.run_id}
                  className="flex cursor-pointer items-center gap-2.5 py-1 text-[11px] text-[var(--muted-strong)]"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(run.run_id)}
                    onChange={() => toggleRun(run.run_id)}
                  />
                  <span className="truncate font-mono">{run.run_id}</span>
                </label>
              ))}
              {!runs.length ? (
                <p className="text-[11px] text-[var(--muted)]">
                  No completed runs found. Start the API and run inference first.
                </p>
              ) : null}
            </div>
            {selected.length ? (
              <div className="mt-4 space-y-1.5 border-t border-[var(--line)] pt-4">
                <Label>Baseline (optional)</Label>
                <select
                  className="tv-input w-full px-2.5 py-2"
                  value={baseline}
                  onChange={(event) => setBaseline(event.target.value)}
                >
                  <option value="">none</option>
                  {selected.map((id) => (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}
          </Panel>

          <ObjectiveBuilder
            onAdd={(objective) => setObjectives((current) => [...current, objective])}
            regionNames={regionNames}
            networkNames={YEO}
          />

          {objectives.length ? (
            <Panel title="Objective set">
              <div className="space-y-2.5">
                {objectives.map((objective, index) => (
                  <div
                    key={objective.objective_id}
                    className="flex items-start justify-between gap-3 border-b border-[var(--line)] pb-2.5 last:border-0"
                  >
                    <div className="min-w-0">
                      <div className="truncate text-[12px] text-[var(--foreground)]">
                        {objective.name}
                      </div>
                      <div className="tv-label mt-1">
                        {objective.direction} · weight {objective.weight}
                      </div>
                    </div>
                    <button
                      className="tv-btn px-2 py-1"
                      onClick={() =>
                        setObjectives((current) => current.filter((_, i) => i !== index))
                      }
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
              <button
                className="tv-btn tv-btn--primary mt-4 w-full px-4 py-2.5"
                disabled={!ready || busy}
                onClick={runScoring}
              >
                {busy ? "Scoring…" : "Score variants"}
              </button>
            </Panel>
          ) : null}

          {error ? (
            <Panel title="Scoring failed">
              <p className="text-[11px] leading-relaxed text-[var(--muted-strong)]">{error}</p>
            </Panel>
          ) : null}
        </div>

        <div className="space-y-4">
          {result ? (
            <>
              <Leaderboard result={result} />
              <Scorecard result={result} />
              {explanation ? <ExplanationPanel explanation={explanation} /> : null}
            </>
          ) : (
            <Panel>
              <div className="py-20 text-center">
                <Label>No score yet</Label>
                <p className="mx-auto mt-3 max-w-md text-[12px] leading-relaxed text-[var(--muted)]">
                  Select at least one completed run and define an objective. Scoring reads saved
                  predictions and never runs inference.
                </p>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </main>
  );
}
