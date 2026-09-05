"use client";

/**
 * Raw values for an inspected cortical location.
 *
 * Shows ONLY verified facts: vertex index, hemisphere, the raw predicted value,
 * and the time. No anatomical region is named — Phase 1 ships no atlas, and
 * inventing a region label would be fabrication. Phase 3 will add ROI metadata
 * here.
 */

import type { PinnedVertex } from "@/stores/neuralVisualizationStore";

export interface VertexReadout {
  vertex: PinnedVertex;
  value: number;
  timeSeconds: number;
  predictionIndex: number;
  isMedialWall: boolean;
}

export function BrainTooltip({ readout }: { readout: VertexReadout }) {
  return (
    <div className="pointer-events-none rounded-lg border border-white/10 bg-[#0d1117]/95 px-3 py-2.5 shadow-2xl backdrop-blur">
      <div className="text-[10px] font-medium uppercase tracking-[0.14em] text-slate-400">
        Predicted Response
      </div>
      <div
        className={`mt-1.5 font-mono text-lg leading-none tabular-nums ${
          Number.isFinite(readout.value) ? "text-slate-100" : "text-fuchsia-300"
        }`}
      >
        {Number.isFinite(readout.value) ? readout.value.toFixed(4) : "no valid value"}
      </div>
      <dl className="mt-2.5 space-y-1 text-[11px] text-slate-400">
        <Row label="Vertex" value={readout.vertex.vertex.toLocaleString()} />
        <Row label="Hemisphere" value={readout.vertex.hemisphere} />
        <Row label="Local index" value={readout.vertex.localIndex.toLocaleString()} />
        <Row label="Time" value={`${readout.timeSeconds.toFixed(2)} s`} />
        <Row label="Sample" value={`#${readout.predictionIndex}`} />
      </dl>
      {!Number.isFinite(readout.value) && (
        <p className="mt-2 border-t border-white/5 pt-1.5 text-[10px] leading-snug text-fuchsia-300/80">
          The model produced NaN or infinity here. Reported, never substituted with a
          neutral value.
        </p>
      )}
      {readout.isMedialWall && (
        <p className="mt-2 border-t border-white/5 pt-1.5 text-[10px] leading-snug text-amber-400/80">
          Medial wall — the model emits a value here, but fMRI signal is not meaningful.
        </p>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-6">
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-mono tabular-nums text-slate-300">{value}</dd>
    </div>
  );
}
