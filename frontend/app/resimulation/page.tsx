"use client";

/**
 * Phase 8 — bounded re-simulation.
 *
 * The only screen in the product that can start a model run, and the only one
 * whose result is a measurement rather than a proposal. Everything up to here
 * produced hypotheses; this is where one is tested against a variant a person
 * actually built.
 *
 * Two things shape the design. First, a pass takes hours, so starting one
 * returns immediately and the page polls the durable state rather than holding
 * a request open. Second, the form asks only for what a person knows — which
 * optimization, which approved candidate, which file — because the objective
 * set, its hashes and the per-hypothesis direction map are derived from stored
 * records. Letting someone type those is how a run ends up measuring an
 * objective nobody approved.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { RunCard } from "@/components/resimulation/RunCard";
import { Label, Notice, Panel } from "@/components/scoring/Primitives";
import { fetchCandidates, fetchOptimizationKeys } from "@/lib/optimization";
import {
  createResimulation,
  fetchResimulations,
  isRunning,
  resumeResimulation,
  stopResimulation,
} from "@/lib/resimulation";
import { fetchExperiments } from "@/lib/scoring";
import type { ProposedVariantSpec } from "@/types/optimization";
import type { ResimulationView } from "@/types/resimulation";

/** Slow enough not to hammer the API, fast enough to see a stage land. */
const POLL_MS = 5000;

export default function ResimulationPage() {
  const [experiments, setExperiments] = useState<string[]>([]);
  const [experimentId, setExperimentId] = useState("");
  const [keys, setKeys] = useState<string[]>([]);
  const [optimizationKey, setOptimizationKey] = useState("");
  const [candidates, setCandidates] = useState<ProposedVariantSpec[]>([]);
  const [candidateId, setCandidateId] = useState("");
  const [variantPath, setVariantPath] = useState("");
  const [sourcePath, setSourcePath] = useState("");
  const [resimulationId, setResimulationId] = useState("");
  const [views, setViews] = useState<ResimulationView[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const describe = (caught: unknown) =>
    (caught as { detail?: string }).detail ?? (caught as Error).message;

  const refresh = useCallback(
    async (signal?: AbortSignal) => {
      try {
        setViews(await fetchResimulations(experimentId.trim() || undefined, signal));
      } catch (caught) {
        if (!signal?.aborted) setError(describe(caught));
      }
    },
    [experimentId],
  );

  useEffect(() => {
    const controller = new AbortController();
    fetchExperiments(controller.signal)
      .then(setExperiments)
      .catch(() => setExperiments([]));
    void refresh(controller.signal);
    return () => controller.abort();
    // Only on mount: later refreshes are driven by the poll and by actions.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll only while something is actually running. A finished board is static,
  // and polling it would be a request per interval that can never change.
  useEffect(() => {
    if (!views.some(isRunning)) return;
    timer.current = setTimeout(() => void refresh(), POLL_MS);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [views, refresh]);

  useEffect(() => {
    if (!experimentId.trim()) {
      setKeys([]);
      setOptimizationKey("");
      return;
    }
    const controller = new AbortController();
    fetchOptimizationKeys(experimentId.trim(), controller.signal)
      .then((found) => {
        setKeys(found);
        setOptimizationKey(found[0] ?? "");
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setError(describe(caught));
      });
    void refresh(controller.signal);
    return () => controller.abort();
  }, [experimentId, refresh]);

  useEffect(() => {
    if (!experimentId.trim() || !optimizationKey) {
      setCandidates([]);
      setCandidateId("");
      return;
    }
    const controller = new AbortController();
    fetchCandidates(experimentId.trim(), optimizationKey, controller.signal)
      .then((found) => {
        setCandidates(found);
        setCandidateId(found[0]?.proposed_variant_id ?? "");
      })
      .catch(() => setCandidates([]));
    return () => controller.abort();
  }, [experimentId, optimizationKey]);

  const selectedCandidate = candidates.find((item) => item.proposed_variant_id === candidateId);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      await createResimulation(experimentId.trim(), {
        resimulation_id: resimulationId.trim(),
        optimization_request_key: optimizationKey,
        proposed_variant_id: candidateId,
        variant_media_path: variantPath.trim(),
        source_media_path: sourcePath.trim() || null,
      });
      await refresh();
    } catch (caught) {
      setError(describe(caught));
    } finally {
      setBusy(false);
    }
  }

  async function act(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      await refresh();
    } catch (caught) {
      setError(describe(caught));
    } finally {
      setBusy(false);
    }
  }

  const ready =
    experimentId.trim() && optimizationKey && candidateId && variantPath.trim() && resimulationId.trim();

  return (
    <main className="min-h-screen bg-black">
      <header className="border-b border-[var(--line)] px-8 py-10">
        <div className="mx-auto max-w-[1440px]">
          <Label>NeuroSplit · Phase 8</Label>
          <h1 className="tv-display mt-3 text-[2.5rem] leading-[1.05] sm:text-[3rem]">
            Bounded Re-Simulation
          </h1>
          <p className="mt-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
            Measures a variant you built against the objective a hypothesis named. This is the
            only part of the product that runs the model, and the only one whose output is a
            measurement rather than a proposal. It never edits media.
          </p>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1440px] gap-4 px-8 py-8 lg:grid-cols-[380px_1fr]">
        <div className="space-y-4">
          <Panel title="Bind a candidate">
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label>Experiment</Label>
                <input
                  className="tv-input w-full px-2.5 py-2"
                  list="experiment-ids"
                  placeholder="experiment id"
                  value={experimentId}
                  onChange={(event) => setExperimentId(event.target.value)}
                />
                <datalist id="experiment-ids">
                  {experiments.map((item) => (
                    <option key={item} value={item} />
                  ))}
                </datalist>
              </div>

              {keys.length ? (
                <div className="space-y-1.5">
                  <Label>Optimization</Label>
                  <select
                    className="tv-input w-full px-2.5 py-2"
                    value={optimizationKey}
                    onChange={(event) => setOptimizationKey(event.target.value)}
                  >
                    {keys.map((key) => (
                      <option key={key} value={key}>
                        {key}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}

              {candidates.length ? (
                <div className="space-y-1.5">
                  <Label>Approved candidate</Label>
                  <select
                    className="tv-input w-full px-2.5 py-2"
                    value={candidateId}
                    onChange={(event) => setCandidateId(event.target.value)}
                  >
                    {candidates.map((item) => (
                      <option key={item.proposed_variant_id} value={item.proposed_variant_id}>
                        {item.proposed_variant_id} · from {item.parent_variant_id}
                      </option>
                    ))}
                  </select>
                </div>
              ) : experimentId.trim() && optimizationKey ? (
                <Notice>
                  This optimization has no approved candidate specification. A recommendation
                  becomes a candidate only after someone approves it on the optimization screen.
                </Notice>
              ) : null}

              <div className="space-y-1.5">
                <Label>Variant media path</Label>
                <input
                  className="tv-input w-full px-2.5 py-2 font-mono text-[11px]"
                  placeholder="/absolute/path/to/variant.mp4"
                  value={variantPath}
                  onChange={(event) => setVariantPath(event.target.value)}
                />
              </div>

              <div className="space-y-1.5">
                <Label>Source media path — optional</Label>
                <input
                  className="tv-input w-full px-2.5 py-2 font-mono text-[11px]"
                  placeholder="defaults to the parent run's stimulus"
                  value={sourcePath}
                  onChange={(event) => setSourcePath(event.target.value)}
                />
              </div>

              <div className="space-y-1.5">
                <Label>Name this pass</Label>
                <input
                  className="tv-input w-full px-2.5 py-2"
                  placeholder="loop-1"
                  value={resimulationId}
                  onChange={(event) => setResimulationId(event.target.value)}
                />
              </div>

              <button
                className="tv-btn tv-btn--primary w-full px-4 py-2.5"
                disabled={busy || !ready}
                onClick={start}
              >
                {busy ? "Binding…" : "Bind and run"}
              </button>

              <Notice>
                Media is read where it lies and never copied or edited. Its bytes are hashed and
                recorded, so a file changed after binding is detected rather than silently used.
              </Notice>
            </div>
          </Panel>

          {selectedCandidate ? (
            <Panel title="What will be tested">
              <div className="space-y-2">
                <Label>{selectedCandidate.hypothesis_ids.length} hypothesis</Label>
                <div className="text-[11px] leading-relaxed text-[var(--muted-strong)]">
                  {selectedCandidate.hypothesis_ids.join(", ")}
                </div>
                <div className="border-t border-[var(--line)] pt-2 text-[11px] text-[var(--muted)]">
                  Target objective {selectedCandidate.target_objective_id}
                </div>
                <Notice>{selectedCandidate.note}</Notice>
              </div>
            </Panel>
          ) : null}

          <Panel title="What this run takes">
            <p className="text-[11.5px] leading-relaxed text-[var(--muted)]">
              Inference, analytics, content analysis, scoring and comparison run in order, each
              checkpointed to disk. A ten-second variant took roughly two hours on this machine.
              Closing this page does not stop the run.
            </p>
          </Panel>

          {error ? (
            <Panel title="Request failed">
              <p className="text-[11px] leading-relaxed text-[var(--muted-strong)]">{error}</p>
            </Panel>
          ) : null}
        </div>

        <div className="space-y-4">
          {views.length ? (
            views.map((view) => (
              <RunCard
                key={view.state.request.resimulation_id}
                view={view}
                busy={busy}
                onResume={() =>
                  void act(() => resumeResimulation(view.state.request.resimulation_id))
                }
                onStop={() => void act(() => stopResimulation(view.state.request.resimulation_id))}
              />
            ))
          ) : (
            <Panel>
              <div className="py-20 text-center">
                <Label>No re-simulation yet</Label>
                <p className="mx-auto mt-3 max-w-md text-[12px] leading-relaxed text-[var(--muted)]">
                  Bind an approved candidate to the variant you built. Until a pass completes,
                  every recommendation from Phase 7 remains an untested hypothesis.
                </p>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </main>
  );
}
