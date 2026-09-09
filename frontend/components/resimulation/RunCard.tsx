"use client";

/**
 * One re-simulation: its lineage, its pipeline, and whatever it has measured.
 *
 * The controls are deliberately conditional. Resume appears only when nothing
 * is running, because a second start on a live run is refused by the store's
 * lock and offering it would invite the error. Stop appears only while a
 * worker is alive, because a stop is a request observed between stages, not a
 * kill, and asking a finished run to stop does nothing.
 */

import { HypothesisVerdicts } from "@/components/resimulation/HypothesisVerdicts";
import { OutcomeTable } from "@/components/resimulation/OutcomeTable";
import { StagePipeline } from "@/components/resimulation/StagePipeline";
import { Label, Notice, Panel } from "@/components/scoring/Primitives";
import { headline, isResumable, isRunning, latestFailure, shortHash } from "@/lib/resimulation";
import type { ResimulationView } from "@/types/resimulation";

export function RunCard({
  view,
  onResume,
  onStop,
  busy,
}: {
  view: ResimulationView;
  onResume: () => void;
  onStop: () => void;
  busy: boolean;
}) {
  const { state } = view;
  const summary = headline(state);
  const failure = latestFailure(state);

  return (
    <Panel
      title={state.request.resimulation_id}
      aside={
        <span className="tv-label text-[var(--muted-strong)]">
          {isRunning(view) ? "running" : state.status.replace("_", " ")}
        </span>
      }
    >
      <div className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field
            label="Lineage"
            value={`${state.request.parent_run_id} → ${
              state.candidate_run_id ?? state.request.proposed_variant.proposed_variant_id
            }`}
          />
          <Field
            label="Iteration"
            value={`${state.request.iteration_index} of ${state.request.max_iterations}`}
          />
          <Field
            label="Source bytes"
            value={shortHash(state.request.binding.source_sha256)}
            mono
          />
          <Field
            label="Variant bytes"
            value={shortHash(state.request.binding.variant_sha256)}
            mono
          />
        </div>

        <StagePipeline state={state} />

        {summary ? (
          <div className="border-t border-[var(--line)] pt-4">
            <Label>Measured</Label>
            <p className="mt-1.5 text-[12px] text-[var(--foreground)]">{summary}</p>
          </div>
        ) : null}

        <div className="border-t border-[var(--line)] pt-4">
          <OutcomeTable
            deltas={state.objective_deltas}
            parentRunId={state.request.parent_run_id}
            candidateRunId={state.candidate_run_id}
          />
        </div>

        <div className="border-t border-[var(--line)] pt-4">
          <Label>Hypotheses</Label>
          <div className="mt-3">
            <HypothesisVerdicts outcomes={state.hypothesis_outcomes} />
          </div>
        </div>

        {failure ? (
          <div className="border-t border-[var(--line)] pt-4">
            <Label>Last failure</Label>
            <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--muted-strong)]">
              {failure}
            </p>
            <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--muted)]">
              Stages that already completed are kept. Resuming verifies every stored checksum and
              continues from the last durable stage rather than starting over.
            </p>
          </div>
        ) : null}

        {view.worker_error ? (
          <Notice>Worker reported: {view.worker_error}</Notice>
        ) : null}

        {state.status === "active" && !view.worker_alive && !failure ? (
          <Notice>
            This run is recorded as active but no process holds its lock. That is what an
            interrupted worker looks like; resume to verify its artifacts and continue.
          </Notice>
        ) : null}

        <div className="flex flex-wrap gap-2 border-t border-[var(--line)] pt-4">
          {isResumable(view) ? (
            <button className="tv-btn px-4 py-2" disabled={busy} onClick={onResume}>
              Verify and resume
            </button>
          ) : null}
          {isRunning(view) ? (
            <button className="tv-btn px-4 py-2" disabled={busy} onClick={onStop}>
              Request stop
            </button>
          ) : null}
        </div>

        <Notice>{state.interpretation_notice}</Notice>
      </div>
    </Panel>
  );
}

function Field({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <Label>{label}</Label>
      <div
        className={`mt-1 truncate text-[11.5px] text-[var(--muted-strong)] ${
          mono ? "font-mono" : ""
        }`}
        title={value}
      >
        {value}
      </div>
    </div>
  );
}
