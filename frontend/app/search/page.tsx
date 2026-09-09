"use client";

/**
 * Phase 9 — automated neural search.
 *
 * The screen is arranged around the claim it has to make honestly. The
 * headline is the best *observed* candidate, never an optimum, and the line
 * under it says whether its lead over the root is larger than the pipeline's
 * own sensitivity to choices that should not matter. When it is not, that is
 * the first thing a reader sees rather than a footnote.
 *
 * Everything is read from durable artifacts. A search runs for hours on a
 * background worker, so this polls rather than holding a request open, and it
 * stops polling once nothing is running.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { BudgetPanel } from "@/components/search/BudgetPanel";
import { SearchTree } from "@/components/search/SearchTree";
import { TrajectoryPlot } from "@/components/search/TrajectoryPlot";
import { Label, Metric, Notice, Panel } from "@/components/scoring/Primitives";
import {
  eliminateCandidate,
  fetchBudget,
  fetchEvents,
  fetchPopulation,
  fetchReport,
  fetchResults,
  fetchSearches,
  fetchTrajectory,
  formatDuration,
  formatFitness,
  formatSigned,
  improvementVerdict,
  lineageLabel,
  pauseSearch,
  pinCandidate,
  resumeSearch,
  stopSearch,
} from "@/lib/search";
import type {
  BudgetView,
  CandidateResult,
  Population,
  SearchEvent,
  SearchReport,
  SearchSummary,
  TrajectoryPoint,
} from "@/types/search";

const POLL_MS = 5000;

export default function SearchPage() {
  const [searches, setSearches] = useState<SearchSummary[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);
  const [budget, setBudget] = useState<BudgetView | null>(null);
  const [populations, setPopulations] = useState<Population[]>([]);
  const [results, setResults] = useState<CandidateResult[]>([]);
  const [events, setEvents] = useState<SearchEvent[]>([]);
  const [report, setReport] = useState<SearchReport | null>(null);
  const [candidate, setCandidate] = useState<string | null>(null);
  const [xAxis, setXAxis] = useState<"evaluations" | "compute">("evaluations");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const describe = (caught: unknown) =>
    (caught as { detail?: string }).detail ?? (caught as Error).message;

  const loadList = useCallback(async (signal?: AbortSignal) => {
    try {
      const found = await fetchSearches(signal);
      setSearches(found);
      setSelected((current) => current || found[0]?.search_id || "");
    } catch (caught) {
      if (!signal?.aborted) setError(describe(caught));
    }
  }, []);

  const loadDetail = useCallback(async (id: string, signal?: AbortSignal) => {
    if (!id) return;
    try {
      const [points, budgetView, population, candidates, history] = await Promise.all([
        fetchTrajectory(id, signal),
        fetchBudget(id, signal),
        fetchPopulation(id, signal),
        fetchResults(id, signal),
        fetchEvents(id, signal),
      ]);
      setTrajectory(points);
      setBudget(budgetView);
      setPopulations(population);
      setResults(candidates);
      setEvents(history);
      // A report exists only once the search has stopped, so its absence is
      // expected rather than an error.
      try {
        setReport(await fetchReport(id, signal));
      } catch {
        setReport(null);
      }
    } catch (caught) {
      if (!signal?.aborted) setError(describe(caught));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadList(controller.signal);
    return () => controller.abort();
  }, [loadList]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setCandidate(null);
    void loadDetail(selected, controller.signal);
    return () => controller.abort();
  }, [selected, loadDetail]);

  // Poll only while something is actually running: a finished search is static
  // and polling it would be a request per interval that can never change.
  useEffect(() => {
    const current = searches.find((item) => item.search_id === selected);
    if (!current?.running) return;
    timer.current = setTimeout(() => {
      void loadList();
      void loadDetail(selected);
    }, POLL_MS);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [searches, selected, loadList, loadDetail]);

  async function act(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      await loadList();
      await loadDetail(selected);
    } catch (caught) {
      setError(describe(caught));
    } finally {
      setBusy(false);
    }
  }

  const current = searches.find((item) => item.search_id === selected) ?? null;
  const metrics = current?.metrics ?? {};
  const verdict = improvementVerdict(metrics);
  const chosen = results.find((item) => item.candidate_id === candidate) ?? null;

  return (
    <main className="min-h-screen bg-black">
      <header className="border-b border-[var(--line)] px-8 py-10">
        <div className="mx-auto max-w-[1440px]">
          <Label>NeuroSplit · Phase 9</Label>
          <h1 className="tv-display mt-3 text-[2.5rem] leading-[1.05] sm:text-[3rem]">
            Automated Neural Search
          </h1>
          <p className="mt-4 max-w-2xl text-[13px] leading-relaxed text-[var(--muted)]">
            Explores content variants under a declared objective, space and budget, and
            reports the strongest it observed. Every evaluation runs the full model
            pipeline, so the search is measured in evaluations, not seconds.
          </p>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1440px] gap-4 px-8 py-8 lg:grid-cols-[340px_1fr]">
        <div className="space-y-4">
          <Panel title="Searches">
            {searches.length ? (
              <div className="space-y-1.5">
                {searches.map((item) => (
                  <button
                    key={item.search_id}
                    onClick={() => setSelected(item.search_id)}
                    className={`w-full border-l px-3 py-2 text-left ${
                      item.search_id === selected
                        ? "border-[var(--foreground)]"
                        : "border-[var(--line)]"
                    }`}
                  >
                    <div className="font-mono text-[11px] text-[var(--foreground)]">
                      {item.search_id}
                    </div>
                    <div className="tv-label mt-1">
                      {item.strategy?.replace(/_/g, " ") ?? "—"} ·{" "}
                      {item.running ? "running" : item.status ?? "—"}
                      {item.paused ? " · paused" : ""}
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-[12px] leading-relaxed text-[var(--muted)]">
                No searches recorded. One is created from the command line or the API;
                this screen observes and controls them.
              </p>
            )}
          </Panel>

          {budget ? (
            <Panel title="Budget">
              <BudgetPanel view={budget} />
            </Panel>
          ) : null}

          {current ? (
            <Panel title="Control">
              <div className="flex flex-wrap gap-2">
                <button
                  className="tv-btn px-3 py-2"
                  disabled={busy || !current.running}
                  onClick={() => void act(() => pauseSearch(selected))}
                >
                  Pause
                </button>
                <button
                  className="tv-btn px-3 py-2"
                  disabled={busy || !current.paused}
                  onClick={() => void act(() => resumeSearch(selected))}
                >
                  Resume
                </button>
                <button
                  className="tv-btn px-3 py-2"
                  disabled={busy || !current.running}
                  onClick={() => void act(() => stopSearch(selected))}
                >
                  Stop
                </button>
              </div>
              <Notice>
                A pause or stop takes effect at the next candidate, not mid-evaluation,
                so no partial run is left behind.
              </Notice>
            </Panel>
          ) : null}

          {error ? (
            <Panel title="Request failed">
              <p className="text-[11px] leading-relaxed text-[var(--muted-strong)]">
                {error}
              </p>
            </Panel>
          ) : null}
        </div>

        <div className="space-y-4">
          {current ? (
            <>
              <Panel
                title="Best observed"
                aside={
                  <span className="tv-label">
                    {current.stopping_reason?.replace(/_/g, " ") ?? "running"}
                  </span>
                }
              >
                <div className="grid gap-6 sm:grid-cols-3">
                  <Metric
                    value={formatFitness(metrics.best_fitness)}
                    caption={current.best_candidate_id ?? "no eligible candidate"}
                    emphasis
                  />
                  <Metric
                    value={formatFitness(metrics.root_fitness)}
                    caption="root"
                  />
                  <Metric
                    value={formatSigned(metrics.absolute_improvement)}
                    caption="improvement"
                  />
                </div>

                <div className="mt-4 border-t border-[var(--line)] pt-3">
                  <Notice>{verdict.label}</Notice>
                  {metrics.tied_candidate_ids?.length ? (
                    <Notice>
                      Not measurably ahead of{" "}
                      {metrics.tied_candidate_ids.length} other candidate
                      {metrics.tied_candidate_ids.length === 1 ? "" : "s"}:{" "}
                      {metrics.tied_candidate_ids.join(", ")}. The ranking among them
                      is not a finding.
                    </Notice>
                  ) : null}
                </div>

                <div className="tv-num mt-4 grid grid-cols-2 gap-3 border-t border-[var(--line)] pt-3 text-[11px] text-[var(--muted-strong)] sm:grid-cols-4">
                  <div>
                    <div>{metrics.evaluations ?? 0}</div>
                    <div className="tv-label mt-1">evaluations</div>
                  </div>
                  <div>
                    <div>{metrics.evaluations_to_best ?? "—"}</div>
                    <div className="tv-label mt-1">to best</div>
                  </div>
                  <div>
                    <div>{formatDuration(metrics.wall_seconds)}</div>
                    <div className="tv-label mt-1">wall clock</div>
                  </div>
                  <div>
                    <div>
                      {metrics.failure_rate === null || metrics.failure_rate === undefined
                        ? "—"
                        : `${(metrics.failure_rate * 100).toFixed(0)}%`}
                    </div>
                    <div className="tv-label mt-1">failed</div>
                  </div>
                </div>
              </Panel>

              <Panel
                title="Optimization trajectory"
                aside={
                  <button
                    className="tv-btn px-2.5 py-1"
                    onClick={() =>
                      setXAxis((current) =>
                        current === "evaluations" ? "compute" : "evaluations",
                      )
                    }
                  >
                    {xAxis === "evaluations" ? "by compute" : "by evaluation"}
                  </button>
                }
              >
                <TrajectoryPlot points={trajectory} xAxis={xAxis} />
              </Panel>

              <Panel title="Search tree">
                <SearchTree
                  results={results}
                  populations={populations}
                  bestCandidateId={current.best_candidate_id}
                  selectedId={candidate}
                  onSelect={setCandidate}
                />
              </Panel>

              {chosen ? (
                <Panel title={`Candidate ${chosen.candidate_id}`}>
                  <div className="space-y-3 text-[11.5px] text-[var(--muted-strong)]">
                    <div className="tv-num">
                      fitness {formatFitness(chosen.scalar_fitness)} ·{" "}
                      {chosen.genome.origin.replace(/_/g, " ")} · generation{" "}
                      {chosen.genome.generation}
                    </div>
                    <div>
                      {Object.entries(chosen.genome.values)
                        .map(([name, value]) => `${name} = ${value}`)
                        .join(" · ")}
                    </div>
                    {chosen.genome.parent_genome_ids.length ? (
                      <div>from {chosen.genome.parent_genome_ids.join(", ")}</div>
                    ) : null}
                    {chosen.candidate_run_id ? (
                      <div className="font-mono text-[10.5px]">
                        run {chosen.candidate_run_id}
                      </div>
                    ) : null}
                    {chosen.reason ? <Notice>{chosen.reason}</Notice> : null}
                    {!chosen.feasibility.feasible ? (
                      <Notice>
                        Disqualified:{" "}
                        {chosen.feasibility.violations
                          .map((item) => item.message)
                          .join("; ")}
                      </Notice>
                    ) : null}
                    <div className="flex gap-2 pt-1">
                      <button
                        className="tv-btn px-3 py-2"
                        disabled={busy}
                        onClick={() =>
                          void act(() => pinCandidate(selected, chosen.candidate_id))
                        }
                      >
                        Pin
                      </button>
                      <button
                        className="tv-btn px-3 py-2"
                        disabled={busy}
                        onClick={() =>
                          void act(() =>
                            eliminateCandidate(selected, chosen.candidate_id),
                          )
                        }
                      >
                        Eliminate
                      </button>
                    </div>
                  </div>
                </Panel>
              ) : null}

              {report?.lineage?.length ? (
                <Panel title="Winning lineage">
                  <div className="space-y-2">
                    {report.lineage.map((step, index) => (
                      <div key={step.candidate_id} className="text-[11.5px]">
                        <span className="tv-label mr-2">gen {step.generation}</span>
                        <span className="text-[var(--muted-strong)]">
                          {lineageLabel(step)}
                        </span>
                        <span className="tv-num ml-2 text-[var(--foreground)]">
                          {formatFitness(step.fitness)}
                        </span>
                        {index < report.lineage.length - 1 ? (
                          <span className="tv-label ml-2">then</span>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </Panel>
              ) : null}

              {report ? (
                <Panel title="What this does not establish">
                  <div className="space-y-3">
                    {report.caveats.map((caveat) => (
                      <Notice key={caveat}>{caveat}</Notice>
                    ))}
                    {report.guardrails.length ? (
                      <div>
                        <Label>Guardrails enforced</Label>
                        <div className="mt-1.5 space-y-1 text-[11px] text-[var(--muted)]">
                          {report.guardrails.map((item) => (
                            <div key={item}>{item}</div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </Panel>
              ) : null}

              <Panel title={`History · ${events.length}`}>
                <div className="max-h-72 space-y-1 overflow-y-auto">
                  {events.slice(-60).map((event, index) => (
                    <div
                      key={`${event.at}-${index}`}
                      className="flex gap-3 text-[10.5px] text-[var(--muted)]"
                    >
                      <span className="tv-label shrink-0">
                        {event.type.replace(/_/g, " ")}
                      </span>
                      <span className="truncate">{event.message}</span>
                    </div>
                  ))}
                </div>
              </Panel>
            </>
          ) : (
            <Panel>
              <div className="py-20 text-center">
                <Label>No search selected</Label>
                <p className="mx-auto mt-3 max-w-md text-[12px] leading-relaxed text-[var(--muted)]">
                  A search explores variants of a root under one declared objective. It
                  reports the best it observed among those it evaluated, which is not
                  the same as the best possible.
                </p>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </main>
  );
}
