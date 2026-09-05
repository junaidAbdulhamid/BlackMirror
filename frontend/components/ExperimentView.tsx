"use client";

/**
 * The experiment page: stimulus and cortex sharing one timeline.
 *
 * Composition only — it holds no rendering or mapping logic. Time lives in the
 * store, the prediction index is derived from it, and BrainViewer receives a
 * values array. That is what will let Phase 5 mount two viewers here without
 * restructuring anything.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useNeuralPrediction } from "@/hooks/useNeuralPrediction";
import { usePlaybackSync, useSeekVideo } from "@/hooks/usePlaybackSync";
import {
  useCurrentPredictionIndex,
  useIsCovered,
  useNeuralStore,
} from "@/stores/neuralVisualizationStore";
import { BrainViewer } from "@/components/brain/BrainViewer";
import {
  BrainErrorState,
  BrainLoadingState,
} from "@/components/brain/BrainLoadingState";
import type { VertexReadout } from "@/components/brain/BrainTooltip";
import { NeuralTimeline } from "@/components/timeline/NeuralTimeline";
import { PlaybackControls } from "@/components/timeline/PlaybackControls";
import { TimelineScrubber } from "@/components/timeline/TimelineScrubber";
import { StimulusPlayer } from "@/components/media/StimulusPlayer";
import { AppHeader } from "@/components/layout/AppHeader";
import { AnalyticsPanel } from "@/components/analytics/AnalyticsPanel";
import { ContentPanel } from "@/components/content/ContentPanel";
import { ScienceNotice } from "@/components/layout/ScienceNotice";
import { stimulusUrl } from "@/lib/api";
import {
  detectResponseChanges,
  meanResponseOverTime,
} from "@/lib/normalization";
import { formatTimecode, temporalGap } from "@/lib/temporalMapping";

export function ExperimentView({ runId }: { runId: string }) {
  const surface = useNeuralStore((s) => s.surface);
  useNeuralPrediction(runId, surface);

  const dataset = useNeuralStore((s) => s.dataset);
  const status = useNeuralStore((s) => s.status);
  const error = useNeuralStore((s) => s.error);
  const progress = useNeuralStore((s) => s.loadProgress);
  const currentTime = useNeuralStore((s) => s.currentTime);
  const duration = useNeuralStore((s) => s.duration);
  const isPlaying = useNeuralStore((s) => s.isPlaying);
  const playbackRate = useNeuralStore((s) => s.playbackRate);
  const normalization = useNeuralStore((s) => s.normalizationMode);
  const hemisphereView = useNeuralStore((s) => s.hemisphereView);
  const debugMode = useNeuralStore((s) => s.debugMode);

  const setCurrentTime = useNeuralStore((s) => s.setCurrentTime);
  const togglePlaying = useNeuralStore((s) => s.togglePlaying);
  const setPlaying = useNeuralStore((s) => s.setPlaying);
  const setPlaybackRate = useNeuralStore((s) => s.setPlaybackRate);
  const seekToIndex = useNeuralStore((s) => s.seekToIndex);
  const setNormalizationMode = useNeuralStore((s) => s.setNormalizationMode);
  const setHemisphereView = useNeuralStore((s) => s.setHemisphereView);
  const toggleDebug = useNeuralStore((s) => s.toggleDebug);

  const predictionIndex = useCurrentPredictionIndex();
  const covered = useIsCovered();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [pinned, setPinned] = useState<VertexReadout | null>(null);
  const [stats, setStats] = useState({ fps: 0, bufferMs: 0 });
  const [fullscreen, setFullscreen] = useState(false);

  usePlaybackSync(videoRef);
  useSeekVideo(videoRef, isPlaying ? Number.NaN : currentTime);

  // `d` toggles the developer overlay; kept off the visible UI.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === "d") toggleDebug();
      if (event.key === " ") {
        event.preventDefault();
        togglePlaying();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleDebug, togglePlaying]);

  const series = useMemo(
    () =>
      dataset
        ? meanResponseOverTime(
            dataset.predictions,
            dataset.timePoints,
            dataset.vertexCount,
            dataset.medialWall,
          )
        : new Float32Array(),
    [dataset],
  );
  const changeIndices = useMemo(() => detectResponseChanges(series), [series]);
  const changeTimes = useMemo(
    () => (dataset ? changeIndices.map((i) => dataset.timeline[i] as number) : []),
    [changeIndices, dataset],
  );

  const handleSeek = useCallback(
    (seconds: number) => {
      setPlaying(false);
      setCurrentTime(seconds);
      const video = videoRef.current;
      if (video && Number.isFinite(seconds)) video.currentTime = seconds;
    },
    [setCurrentTime, setPlaying],
  );

  if (status === "error") {
    return (
      <Shell runId={runId}>
        <BrainErrorState message={error ?? "Unknown error."} />
      </Shell>
    );
  }
  if (!dataset || status !== "ready") {
    return (
      <Shell runId={runId}>
        <BrainLoadingState progress={progress} />
      </Shell>
    );
  }

  const { summary } = dataset;
  // Distance to the nearest row, shown only to explain a gap. Coverage itself
  // is interval membership, not proximity.
  const gap = temporalGap(currentTime, dataset.timeline);

  return (
    <div className="min-h-screen">
      <AppHeader subtitle="Neural Simulation" />

      <main className="mx-auto max-w-[1600px] px-6 py-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <h1 className="truncate text-xl font-semibold tracking-tight text-white">
                {summary.stimulus.filename}
              </h1>
              {summary.model.is_synthetic && (
                <span className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                  Synthetic test data
                </span>
              )}
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-slate-500">
              <span>{summary.run_id}</span>
              <span className="text-slate-700">·</span>
              <span>{summary.model.name}</span>
              <span className="text-slate-700">·</span>
              <span>
                {summary.temporal.n_time_points} × {summary.cortical.vertex_count.toLocaleString()}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setFullscreen((v) => !v)}
              className="rounded-md border border-white/10 px-2.5 py-1 text-[11px] text-slate-400 transition-colors hover:border-white/25 hover:text-slate-200"
            >
              {fullscreen ? "Exit fullscreen brain" : "Fullscreen brain"}
            </button>
            <ScienceNotice notice={summary.interpretation_notice} />
          </div>
        </div>

        <div
          className={`mt-5 grid gap-5 ${
            fullscreen ? "grid-cols-1" : "lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]"
          }`}
        >
          {!fullscreen && (
            <section className="flex flex-col">
              <SectionLabel>Stimulus</SectionLabel>
              <div className="mt-2 min-h-[280px] flex-1">
                <StimulusPlayer
                  ref={videoRef}
                  runId={runId}
                  stimulus={summary.stimulus}
                  src={stimulusUrl(runId)}
                />
              </div>
            </section>
          )}

          <section className="flex flex-col">
            <div className="flex items-baseline justify-between">
              <SectionLabel>Predicted Neural Response</SectionLabel>
              <span className="text-[10px] text-slate-500">
                {covered ? (
                  <>
                    sample #{predictionIndex} ·{" "}
                    {formatTimecode(dataset.timeline[predictionIndex] as number)}
                  </>
                ) : (
                  <span className="text-amber-400/80">
                    no sample covers this moment · nearest is {gap.toFixed(1)}s away
                  </span>
                )}
              </span>
            </div>
            <div className={`mt-2 ${fullscreen ? "h-[72vh]" : "h-[460px]"}`}>
              <BrainViewer
                hemispheres={dataset.hemispheres}
                values={dataset.predictions}
                vertexCount={dataset.vertexCount}
                predictionIndex={predictionIndex}
                scale={summary.scale}
                medialWall={dataset.medialWall}
                currentTime={currentTime}
                covered={covered}
                normalization={normalization}
                onNormalizationChange={setNormalizationMode}
                view={hemisphereView}
                onViewChange={setHemisphereView}
                separateHemispheres={surface === "inflated"}
                onPin={setPinned}
                onStats={setStats}
              />
            </div>
          </section>
        </div>

        <section className="mt-6 rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
          <TimelineScrubber
            currentTime={currentTime}
            duration={duration}
            timeline={dataset.timeline}
            activeIndex={predictionIndex}
            markers={changeTimes}
            onSeek={handleSeek}
          />
          <div className="mt-4">
            <PlaybackControls
              isPlaying={isPlaying}
              onTogglePlay={togglePlaying}
              onRestart={() => handleSeek(0)}
              onStep={(delta) => {
                setPlaying(false);
                seekToIndex(predictionIndex + delta);
              }}
              playbackRate={playbackRate}
              onRateChange={setPlaybackRate}
              predictionIndex={Math.max(0, predictionIndex)}
              timePoints={dataset.timePoints}
            />
          </div>
        </section>

        <section className="mt-5 rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
          <NeuralTimeline
            series={series}
            timeline={dataset.timeline}
            currentTime={currentTime}
            duration={duration}
            activeIndex={predictionIndex}
            changeIndices={changeIndices}
            onSeek={handleSeek}
          />
        </section>

        <AnalyticsPanel
          runId={runId}
          predictionIndex={predictionIndex}
          onSeek={handleSeek}
        />

        <ContentPanel runId={runId} currentTime={currentTime} onSeek={handleSeek} />

        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <DetailsPanel summary={summary} predictionIndex={predictionIndex} />
          {pinned ? (
            <PinnedPanel
              readout={pinned}
              dataset={dataset}
              onClear={() => setPinned(null)}
            />
          ) : (
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
              <SectionLabel>Selected location</SectionLabel>
              <p className="mt-2 text-[12px] leading-relaxed text-slate-500">
                Click anywhere on the cortex to pin a vertex and see its predicted response
                across time.
              </p>
            </div>
          )}
        </div>

        {debugMode && (
          <DebugPanel
            stats={stats}
            currentTime={currentTime}
            predictionIndex={predictionIndex}
            dataset={dataset}
            covered={covered}
          />
        )}
      </main>
    </div>
  );
}

function Shell({ runId, children }: { runId: string; children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <AppHeader subtitle="Neural Simulation" />
      <main className="mx-auto max-w-[1600px] px-6 py-6">
        <Link href="/" className="text-[11px] text-slate-500 hover:text-slate-300">
          ← All experiments
        </Link>
        <div className="mt-2 font-mono text-[11px] text-slate-600">{runId}</div>
        {children}
      </main>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-400">
      {children}
    </h2>
  );
}

function DetailsPanel({
  summary,
  predictionIndex,
}: {
  summary: import("@/types/neural").RunSummary;
  predictionIndex: number;
}) {
  const t = summary.temporal;
  const rows: [string, string][] = [
    ["Model", summary.model.name],
    ["Surface", summary.cortical.surface_space],
    ["Vertices", summary.cortical.vertex_count.toLocaleString()],
    ["Prediction samples", String(t.n_time_points)],
    ["Current sample", `#${Math.max(0, predictionIndex)}`],
    ["TR", `${t.tr_seconds.toFixed(2)} s`],
    [
      "Segments kept",
      t.n_segments_total
        ? `${t.n_segments_kept} of ${t.n_segments_total} (${Math.round(
            (1 - (t.n_segments_kept ?? 0) / t.n_segments_total) * 100,
          )}% dropped)`
        : String(t.n_segments_kept ?? "—"),
    ],
    ["Timeline", t.timeline_is_contiguous ? "Contiguous" : "Sparse (rows dropped)"],
    [
      "Hemodynamic offset",
      `${t.hemodynamic_offset_seconds.toFixed(1)} s — ${
        t.output_is_stimulus_aligned ? "already applied by model" : "not applied"
      }`,
    ],
    ["Device", `${summary.model.device} · ${summary.model.dtype}`],
    ["License", summary.model.license ?? "—"],
  ];

  return (
    <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
      <SectionLabel>Prediction details</SectionLabel>
      <dl className="mt-3 grid gap-x-8 gap-y-2 sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-4 border-b border-white/5 pb-1.5">
            <dt className="text-[11.5px] text-slate-500">{label}</dt>
            <dd className="text-right font-mono text-[11.5px] tabular-nums text-slate-300">
              {value}
            </dd>
          </div>
        ))}
      </dl>
      {summary.temporal.notes.length > 0 && (
        <ul className="mt-3 space-y-1 text-[11px] leading-relaxed text-amber-400/80">
          {summary.temporal.notes.map((note) => (
            <li key={note}>⚠ {note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PinnedPanel({
  readout,
  dataset,
  onClear,
}: {
  readout: VertexReadout;
  dataset: import("@/types/neural").NeuralDataset;
  onClear: () => void;
}) {
  const series = useMemo(() => {
    const values = new Float32Array(dataset.timePoints);
    for (let t = 0; t < dataset.timePoints; t += 1) {
      values[t] = dataset.predictions[t * dataset.vertexCount + readout.vertex.vertex] ?? 0;
    }
    return values;
  }, [dataset, readout.vertex.vertex]);

  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = max - min || 1;

  return (
    <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
      <div className="flex items-baseline justify-between">
        <SectionLabel>Selected location</SectionLabel>
        <button
          type="button"
          onClick={onClear}
          className="text-[11px] text-slate-500 hover:text-slate-300"
        >
          Clear
        </button>
      </div>
      <dl className="mt-3 space-y-1.5 text-[11.5px]">
        <Row label="Vertex" value={readout.vertex.vertex.toLocaleString()} />
        <Row label="Hemisphere" value={readout.vertex.hemisphere} />
        <Row label="Value now" value={readout.value.toFixed(4)} />
      </dl>
      <div className="mt-3">
        <div className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
          Response across time
        </div>
        <svg viewBox="0 0 200 52" className="mt-1.5 h-[52px] w-full">
          <polyline
            fill="none"
            stroke="rgb(56,189,248)"
            strokeWidth={1.5}
            points={Array.from(series)
              .map(
                (v, i) =>
                  `${(i / Math.max(1, series.length - 1)) * 200},${46 - ((v - min) / span) * 40}`,
              )
              .join(" ")}
          />
        </svg>
      </div>
      <p className="mt-2 text-[10px] leading-relaxed text-slate-600">
        No anatomical region is shown: Phase 1 ships no atlas, and naming one would be
        fabrication. Phase 3 adds ROI metadata.
      </p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-mono tabular-nums text-slate-300">{value}</dd>
    </div>
  );
}

function DebugPanel({
  stats,
  currentTime,
  predictionIndex,
  dataset,
  covered,
}: {
  stats: { fps: number; bufferMs: number };
  currentTime: number;
  predictionIndex: number;
  dataset: import("@/types/neural").NeuralDataset;
  covered: boolean;
}) {
  const offset = Math.max(0, predictionIndex) * dataset.vertexCount;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < dataset.vertexCount; i += 1) {
    const v = dataset.predictions[offset + i] as number;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const meshVertices = dataset.hemispheres.reduce((sum, h) => sum + h.vertexCount, 0);
  let nonFinite = 0;
  for (let i = 0; i < dataset.vertexCount; i += 1) {
    if (!Number.isFinite(dataset.predictions[offset + i] as number)) nonFinite += 1;
  }

  return (
    <div className="fixed bottom-4 right-4 z-40 w-[250px] rounded-lg border border-white/15 bg-black/90 p-3 font-mono text-[10.5px] leading-relaxed text-emerald-300 shadow-2xl backdrop-blur">
      <div className="mb-1.5 text-[9px] uppercase tracking-[0.16em] text-slate-500">
        Debug · press d
      </div>
      <div>FPS: {stats.fps.toFixed(0)}</div>
      <div>Time: {currentTime.toFixed(3)} s</div>
      <div>Prediction index: {predictionIndex}</div>
      <div>Prediction vertices: {dataset.vertexCount.toLocaleString()}</div>
      <div>Mesh vertices: {meshVertices.toLocaleString()}</div>
      <div className={meshVertices === dataset.vertexCount ? "" : "text-red-400"}>
        Aligned: {meshVertices === dataset.vertexCount ? "yes" : "NO"}
      </div>
      <div className={covered ? "" : "text-amber-400"}>
        Covered: {covered ? "yes" : "NO — inert surface"}
      </div>
      <div className={nonFinite ? "text-fuchsia-400" : ""}>Non-finite: {nonFinite}</div>
      <div>Frame min: {min.toFixed(4)}</div>
      <div>Frame max: {max.toFixed(4)}</div>
      <div>Buffer update: {stats.bufferMs.toFixed(2)} ms</div>
    </div>
  );
}
