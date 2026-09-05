"use client";

/**
 * Content Intelligence — Phase 4's UI surface.
 *
 * Five tabs over one analysis: an overview of quantitative metrics, the scene
 * breakdown with keyframes, a seekable transcript, the fused content-event
 * timeline, and the neural associations.
 *
 * LANGUAGE DISCIPLINE
 *   The associations tab is the one place where content and predicted neural
 *   responses meet, and it is therefore the easiest place to imply causation.
 *   Every association renders the backend's `interpretation` string verbatim,
 *   and the heading says "co-occurring", never "caused".
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchContentAnalysis, keyframeUrl } from "@/lib/api";
import { formatTimecode } from "@/lib/temporalMapping";
import type { ContentAnalysis, ContentEvent } from "@/types/neural";

type Tab = "overview" | "scenes" | "transcript" | "events" | "associations";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "scenes", label: "Scenes" },
  { id: "transcript", label: "Transcript" },
  { id: "events", label: "Content Events" },
  { id: "associations", label: "Neural Associations" },
];

export interface ContentPanelProps {
  runId: string;
  currentTime: number;
  onSeek: (seconds: number) => void;
}

export function ContentPanel({ runId, currentTime, onSeek }: ContentPanelProps) {
  const [analysis, setAnalysis] = useState<ContentAnalysis | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "absent" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  useEffect(() => {
    const controller = new AbortController();
    setState("loading");
    fetchContentAnalysis(runId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setAnalysis(result);
        setState(result ? "ready" : "absent");
      })
      .catch((exc: unknown) => {
        if (controller.signal.aborted) return;
        setError(exc instanceof Error ? exc.message : "Unknown error");
        setState("error");
      });
    return () => controller.abort();
  }, [runId]);

  if (state === "loading") return null;

  if (state === "absent") {
    return (
      <Section>
        <Header />
        <p className="mt-3 text-[12px] leading-relaxed text-slate-500">
          No content analysis for this run yet. Derive it with:
        </p>
        <code className="mt-2 block rounded bg-black/40 px-3 py-2 text-[11px] text-slate-400">
          blackmirror analyze-content {runId}
        </code>
        <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
          It is not run on page load: the pipeline loads several models and takes tens of
          seconds.
        </p>
      </Section>
    );
  }

  if (state === "error" || !analysis) {
    return (
      <Section>
        <Header />
        <p className="mt-3 text-[12px] text-red-300">{error ?? "Could not load analysis."}</p>
      </Section>
    );
  }

  return (
    <Section>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <Header />
        <div className="flex flex-wrap gap-1">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              onClick={() => setTab(entry.id)}
              className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                tab === entry.id
                  ? "bg-sky-500/15 text-sky-200"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {entry.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4">
        {tab === "overview" && <Overview analysis={analysis} />}
        {tab === "scenes" && (
          <Scenes analysis={analysis} runId={runId} onSeek={onSeek} currentTime={currentTime} />
        )}
        {tab === "transcript" && (
          <Transcript analysis={analysis} onSeek={onSeek} currentTime={currentTime} />
        )}
        {tab === "events" && (
          <Events analysis={analysis} onSeek={onSeek} currentTime={currentTime} />
        )}
        {tab === "associations" && <Associations analysis={analysis} onSeek={onSeek} />}
      </div>

      {analysis.metadata.warnings.length > 0 && (
        <ul className="mt-4 space-y-1 border-t border-white/5 pt-3 text-[10.5px] leading-relaxed text-amber-400/80">
          {analysis.metadata.warnings.map((warning) => (
            <li key={warning}>⚠ {warning}</li>
          ))}
        </ul>
      )}

      <p className="mt-3 border-t border-white/5 pt-3 text-[10px] leading-relaxed text-slate-600">
        {analysis.metadata.interpretation_notice}
      </p>
    </Section>
  );
}

function Section({ children }: { children: React.ReactNode }) {
  return (
    <section className="mt-5 rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
      {children}
    </section>
  );
}

function Header() {
  return (
    <h2 className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-400">
      Content Intelligence
    </h2>
  );
}

function Overview({ analysis }: { analysis: ContentAnalysis }) {
  const m = analysis.metrics;
  const stats: [string, string][] = [
    ["Duration", `${m.duration_seconds.toFixed(2)} s`],
    ["Shots", String(m.shot_count)],
    ["Mean shot", m.mean_shot_duration != null ? `${m.mean_shot_duration.toFixed(2)} s` : "—"],
    ["Cuts / min", m.cuts_per_minute != null ? m.cuts_per_minute.toFixed(1) : "—"],
    ["Scenes", String(m.scene_count)],
    ["Speech", pct(m.speech_fraction)],
    ["Music", pct(m.music_fraction)],
    ["Silence", pct(m.silence_fraction)],
    ["Words", `${m.word_count}${m.words_per_minute ? ` (${m.words_per_minute.toFixed(0)} wpm)` : ""}`],
    ["Text overlays", String(m.text_overlay_count)],
    ["CTAs", String(m.cta_count)],
    ["Content events", String(m.content_event_count)],
  ];

  return (
    <div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-4">
        {stats.map(([label, value]) => (
          <div key={label} className="border-b border-white/5 pb-1.5">
            <dt className="text-[10px] uppercase tracking-[0.1em] text-slate-600">{label}</dt>
            <dd className="mt-0.5 font-mono text-[13px] tabular-nums text-slate-200">{value}</dd>
          </div>
        ))}
      </dl>

      {(analysis.objects.length > 0 || analysis.audio_events.length > 0) && (
        <div className="mt-4 grid gap-4 border-t border-white/5 pt-3 sm:grid-cols-2">
          {analysis.objects.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
                Detected objects
              </div>
              <ul className="mt-1.5 space-y-1">
                {analysis.objects.slice(0, 6).map((object) => (
                  <li
                    key={object.label}
                    className="flex justify-between gap-3 text-[11px]"
                  >
                    <span className="text-slate-300">{object.label}</span>
                    <span className="font-mono tabular-nums text-slate-500">
                      {object.screen_time_seconds.toFixed(1)}s · ×
                      {object.detection_count}
                      {object.mean_confidence != null
                        ? ` · ${object.mean_confidence.toFixed(2)}`
                        : ""}
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-1.5 text-[9.5px] leading-snug text-slate-600">
                Open-vocabulary detection: only phrases the run asked about could be
                found, so an absent label means &ldquo;not asked, or not found&rdquo; &mdash;
                never &ldquo;not present&rdquo;. Grouped by label, not by object instance.
              </p>
            </div>
          )}
          {analysis.audio_events.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
                Audio events
              </div>
              <ul className="mt-1.5 space-y-1">
                {analysis.audio_events.slice(0, 6).map((event) => (
                  <li
                    key={`${event.label}-${event.start_time}`}
                    className="flex justify-between gap-3 text-[11px]"
                  >
                    <span className="text-slate-300">{event.label}</span>
                    <span className="font-mono tabular-nums text-slate-500">
                      {formatTimecode(event.start_time)} · {event.confidence.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="mt-4 border-t border-white/5 pt-3">
        <div className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
          Models used
        </div>
        <dl className="mt-1.5 grid gap-x-6 gap-y-1 sm:grid-cols-2">
          {Object.entries(analysis.metadata.models).map(([stage, model]) => (
            <div key={stage} className="flex justify-between gap-3 text-[11px]">
              <dt className="text-slate-500">{stage.replace(/_/g, " ")}</dt>
              <dd className="truncate font-mono text-slate-400">{model}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}

function Scenes({
  analysis,
  runId,
  onSeek,
  currentTime,
}: {
  analysis: ContentAnalysis;
  runId: string;
  onSeek: (s: number) => void;
  currentTime: number;
}) {
  if (analysis.scenes.length === 0) {
    return <Empty>No scenes were segmented for this stimulus.</Empty>;
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {analysis.scenes.map((scene) => {
        const active = currentTime >= scene.start_time && currentTime < scene.end_time;
        const speech = analysis.transcript.filter(
          (t) => t.start_time < scene.end_time && t.end_time > scene.start_time,
        );
        return (
          <button
            key={scene.index}
            type="button"
            onClick={() => onSeek(scene.start_time)}
            className={`rounded-lg border p-3 text-left transition-colors ${
              active
                ? "border-sky-400/40 bg-sky-500/[0.07]"
                : "border-white/10 bg-white/[0.02] hover:border-white/20"
            }`}
          >
            <div className="flex items-baseline justify-between">
              <span className="text-[11px] font-semibold text-slate-200">
                Scene {scene.index + 1}
              </span>
              <span className="font-mono text-[10px] tabular-nums text-slate-500">
                {formatTimecode(scene.start_time)}–{formatTimecode(scene.end_time)}
              </span>
            </div>
            {scene.keyframe_path && (
              <img
                src={keyframeUrl(runId, scene.keyframe_path)}
                alt=""
                className="mt-2 aspect-video w-full rounded object-cover"
              />
            )}
            {scene.description ? (
              <p className="mt-2 text-[11px] leading-snug text-slate-400">{scene.description}</p>
            ) : (
              <p className="mt-2 text-[11px] leading-snug text-slate-600 italic">
                No confident visual label — the classifier abstained rather than guess.
              </p>
            )}
            {speech.length > 0 && (
              <p className="mt-1.5 truncate text-[11px] italic text-slate-500">
                “{speech[0]!.text}”
              </p>
            )}
            <div className="mt-1.5 text-[10px] text-slate-600">
              {scene.shot_indices.length} shot{scene.shot_indices.length === 1 ? "" : "s"}
              {scene.grouping_reason ? ` · ${scene.grouping_reason}` : ""}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function Transcript({
  analysis,
  onSeek,
  currentTime,
}: {
  analysis: ContentAnalysis;
  onSeek: (s: number) => void;
  currentTime: number;
}) {
  if (analysis.transcript.length === 0) {
    return <Empty>No speech was detected in this stimulus.</Empty>;
  }
  return (
    <div className="space-y-1.5">
      {analysis.transcript.map((segment) => {
        const active = currentTime >= segment.start_time && currentTime < segment.end_time;
        return (
          <div
            key={segment.index}
            className={`rounded-lg border px-3 py-2 transition-colors ${
              active ? "border-sky-400/40 bg-sky-500/[0.07]" : "border-white/[0.07]"
            }`}
          >
            <div className="flex items-baseline gap-2">
              <button
                type="button"
                onClick={() => onSeek(segment.start_time)}
                className="font-mono text-[10.5px] tabular-nums text-sky-300/80 hover:text-sky-200"
              >
                {formatTimecode(segment.start_time)}
              </button>
              {segment.speaker && (
                <span className="rounded border border-white/10 px-1.5 py-0.5 text-[9.5px] uppercase tracking-wider text-slate-400">
                  {segment.speaker}
                </span>
              )}
            </div>
            <p className="mt-1 text-[13px] leading-relaxed text-slate-200">
              {segment.words.length > 0
                ? segment.words.map((word, index) => (
                    <button
                      key={`${word.text}-${index}`}
                      type="button"
                      onClick={() => onSeek(word.start_time)}
                      title={`${word.start_time.toFixed(2)} s`}
                      className={`mr-1 rounded px-0.5 transition-colors hover:bg-sky-500/20 ${
                        currentTime >= word.start_time && currentTime < word.end_time
                          ? "bg-sky-500/25 text-white"
                          : ""
                      }`}
                    >
                      {word.text}
                    </button>
                  ))
                : segment.text}
            </p>
            <div className="mt-1 text-[10px] text-slate-600">
              {segment.words.length > 0
                ? `${segment.words.length} word timings · `
                : ""}
              source: {segment.provenance.model_id ?? segment.provenance.source}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Events({
  analysis,
  onSeek,
  currentTime,
}: {
  analysis: ContentAnalysis;
  onSeek: (s: number) => void;
  currentTime: number;
}) {
  const [selected, setSelected] = useState<ContentEvent | null>(null);
  const active = useMemo(
    () =>
      analysis.events.find((e) => currentTime >= e.start_time && currentTime < e.end_time) ??
      null,
    [analysis.events, currentTime],
  );
  const shown = selected ?? active;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="max-h-[340px] space-y-1 overflow-y-auto pr-1">
        {analysis.events.map((event) => (
          <button
            key={event.event_id}
            type="button"
            onClick={() => {
              setSelected(event);
              onSeek(event.start_time);
            }}
            className={`flex w-full items-baseline justify-between gap-3 rounded-md border px-2.5 py-1.5 text-left transition-colors ${
              shown?.event_id === event.event_id
                ? "border-sky-400/40 bg-sky-500/[0.07]"
                : "border-white/[0.07] hover:border-white/20"
            }`}
          >
            <span className="min-w-0 flex-1 truncate text-[11.5px] text-slate-300">
              {event.speech_text ?? event.visual_description ?? event.event_type}
            </span>
            <span className="shrink-0 font-mono text-[10px] tabular-nums text-slate-500">
              {formatTimecode(event.start_time)}
            </span>
          </button>
        ))}
      </div>

      <div className="rounded-lg border border-white/[0.07] p-3">
        {shown ? <EventDetail event={shown} /> : <Empty>Select an event to inspect it.</Empty>}
      </div>
    </div>
  );
}

function EventDetail({ event }: { event: ContentEvent }) {
  const rows: [string, string][] = [
    ["Interval", `${formatTimecode(event.start_time)} – ${formatTimecode(event.end_time)}`],
    ["Type", event.event_type.replace(/_/g, " ")],
    ["Modalities", event.modalities.join(", ") || "—"],
  ];
  return (
    <div className="space-y-2 text-[11.5px]">
      <dl className="space-y-1">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-4">
            <dt className="text-slate-500">{label}</dt>
            <dd className="text-right font-mono text-slate-300">{value}</dd>
          </div>
        ))}
      </dl>
      {event.visual_description && <Field label="Visual" value={event.visual_description} />}
      {event.speech_text && <Field label="Speech" value={`“${event.speech_text}”`} />}
      {event.on_screen_text.length > 0 && (
        <Field label="On-screen text" value={event.on_screen_text.join(" · ")} />
      )}
      {event.audio_description && <Field label="Audio" value={event.audio_description} />}
      {event.objects.length > 0 && <Field label="Objects" value={event.objects.join(", ")} />}
      {Object.keys(event.visual_features).length > 0 && (
        <Field
          label="Visual signals"
          value={Object.entries(event.visual_features)
            .map(([k, v]) => `${k} ${v.toFixed(3)}`)
            .join(" · ")}
        />
      )}
      {Object.keys(event.audio_features).length > 0 && (
        <Field
          label="Audio signals"
          value={Object.entries(event.audio_features)
            .map(([k, v]) => `${k} ${v.toFixed(3)}`)
            .join(" · ")}
        />
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-t border-white/5 pt-1.5">
      <div className="text-[10px] uppercase tracking-[0.1em] text-slate-600">{label}</div>
      <div className="mt-0.5 leading-snug text-slate-300">{value}</div>
    </div>
  );
}

function Associations({
  analysis,
  onSeek,
}: {
  analysis: ContentAnalysis;
  onSeek: (s: number) => void;
}) {
  if (analysis.associations.length === 0) {
    return (
      <Empty>
        No neural associations. Derive Phase 3 analytics first with{" "}
        <code className="text-slate-400">blackmirror analyze {analysis.run_id}</code>, then
        re-run content analysis.
      </Empty>
    );
  }
  return (
    <div className="space-y-3">
      {analysis.associations.map((association) => (
        <div
          key={association.neural_event_id}
          className="rounded-lg border border-white/[0.07] p-3"
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <button
              type="button"
              onClick={() => onSeek(association.neural_time)}
              className="text-[12px] font-semibold text-slate-200 hover:text-white"
            >
              {association.neural_event_type.replace(/_/g, " ")} @{" "}
              {formatTimecode(association.neural_time)}
            </button>
            <span className="font-mono text-[10px] text-slate-500">
              window {association.window_start.toFixed(1)}–{association.window_end.toFixed(1)} s
              · score {association.neural_score.toFixed(3)}
            </span>
          </div>

          <div className="mt-2 text-[10px] uppercase tracking-[0.1em] text-slate-600">
            Co-occurring stimulus content
          </div>
          <dl className="mt-1 space-y-1 text-[11.5px]">
            {association.visual_context.length > 0 && (
              <ContextRow label="Visual" values={association.visual_context} />
            )}
            {association.speech_context.length > 0 && (
              <ContextRow label="Speech" values={association.speech_context} quoted />
            )}
            {association.on_screen_text_context.length > 0 && (
              <ContextRow label="On-screen" values={association.on_screen_text_context} />
            )}
            {association.audio_context.length > 0 && (
              <ContextRow label="Audio" values={association.audio_context} />
            )}
          </dl>

          <p className="mt-2 border-t border-white/5 pt-2 text-[10px] leading-relaxed text-amber-400/70">
            {association.interpretation}
          </p>
        </div>
      ))}
    </div>
  );
}

function ContextRow({
  label,
  values,
  quoted = false,
}: {
  label: string;
  values: string[];
  quoted?: boolean;
}) {
  return (
    <div className="flex gap-3">
      <dt className="w-20 shrink-0 text-slate-500">{label}</dt>
      <dd className="text-slate-300">
        {values.map((value) => (quoted ? `“${value}”` : value)).join(" · ")}
      </dd>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-[12px] leading-relaxed text-slate-500">{children}</p>;
}

function pct(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
