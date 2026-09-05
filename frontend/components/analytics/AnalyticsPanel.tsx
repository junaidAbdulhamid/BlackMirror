"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchAnalytics, fetchAnalyticsArray } from "@/lib/api";
import type { NeuralAnalyticsSummary } from "@/types/neural";

export function AnalyticsPanel({
  runId,
  predictionIndex,
  onSeek,
}: {
  runId: string;
  predictionIndex: number;
  onSeek: (seconds: number) => void;
}) {
  const [metadata, setMetadata] = useState<NeuralAnalyticsSummary | null>(null);
  const [roi, setRoi] = useState<Float64Array | null>(null);
  const [globalMagnitude, setGlobalMagnitude] = useState<Float64Array | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setUnavailable(false);
    setMetadata(null);
    setRoi(null);
    setGlobalMagnitude(null);
    Promise.all([
      fetchAnalytics(runId, controller.signal),
      fetchAnalyticsArray(runId, "roi_timeseries", controller.signal),
      fetchAnalyticsArray(runId, "global_l2_magnitude", controller.signal),
    ])
      .then(([nextMetadata, nextRoi, nextMagnitude]) => {
        setMetadata(nextMetadata);
        setRoi(nextRoi);
        setGlobalMagnitude(nextMagnitude);
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) setUnavailable(true);
      });
    return () => controller.abort();
  }, [runId]);

  const topRegions = useMemo(() => {
    if (!metadata || !roi || predictionIndex < 0) return [];
    const count = metadata.atlas.region_count;
    const offset = predictionIndex * count;
    return metadata.roi_responses
      .map((summary, regionId) => ({ summary, value: roi[offset + regionId] as number }))
      .filter(({ value }) => Number.isFinite(value))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 5);
  }, [metadata, predictionIndex, roi]);

  if (unavailable) {
    return (
      <section className="mt-5 rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
        <Label>Neural analytics</Label>
        <p className="mt-2 text-xs text-slate-500">
          Analytics have not been derived for this run. Run <code>blackmirror analyze {runId}</code>.
        </p>
      </section>
    );
  }
  if (!metadata || !roi || !globalMagnitude) return null;

  return (
    <section className="mt-5 rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <Label>Neural analytics</Label>
        <span className="font-mono text-[10px] text-slate-600">
          {metadata.atlas.name} · {metadata.atlas.region_count} cortical ROIs · v{metadata.metadata.analytics_version}
        </span>
      </div>
      <div className="mt-4 grid gap-5 md:grid-cols-3">
        <div>
          <SubLabel>Whole-cortex magnitude</SubLabel>
          <div className="mt-2 font-mono text-xl text-sky-300">
            {(globalMagnitude[predictionIndex] ?? 0).toFixed(3)}
          </div>
          <p className="mt-1 text-[10px] text-slate-600">L2 response-vector magnitude; not engagement.</p>
        </div>
        <div>
          <SubLabel>Top regions now</SubLabel>
          <ol className="mt-2 space-y-1.5">
            {topRegions.map(({ summary, value }) => (
              <li key={summary.region.region_id} className="flex justify-between gap-3 text-[11px]">
                <span className="truncate text-slate-400">{summary.region.name} ({summary.region.hemisphere.charAt(0).toUpperCase()})</span>
                <span className="font-mono text-slate-200">{value.toFixed(4)}</span>
              </li>
            ))}
          </ol>
        </div>
        <div>
          <SubLabel>Ranked response events</SubLabel>
          <div className="mt-2 space-y-1.5">
            {metadata.events.slice(0, 5).map((event) => (
              <button
                type="button"
                key={event.event_id}
                onClick={() => onSeek(event.timestamp_seconds)}
                className="flex w-full justify-between text-left text-[11px] text-slate-400 hover:text-sky-300"
              >
                <span>{event.event_type.replaceAll("_", " ")}</span>
                <span className="font-mono">{event.timestamp_seconds.toFixed(1)}s · z {event.score.toFixed(2)}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
      <p className="mt-4 border-t border-white/5 pt-3 text-[10px] leading-relaxed text-slate-600">
        {metadata.metadata.interpretation_notice}
      </p>
    </section>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <h2 className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-400">{children}</h2>;
}

function SubLabel({ children }: { children: React.ReactNode }) {
  return <h3 className="text-[10px] uppercase tracking-[0.12em] text-slate-500">{children}</h3>;
}
