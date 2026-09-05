"use client";

/**
 * The colour scale, and what it means.
 *
 * The gradient is generated from the same ramp the mesh samples, so the legend
 * cannot drift from the surface. Wording is constrained: "Predicted cortical
 * response", never "brain activity", and the active normalization is always
 * stated so a reader knows what the endpoints represent.
 */

import { gradientCss } from "@/lib/colorMapping";
import type { NormalizationRange } from "@/lib/normalization";

export function BrainLegend({
  range,
  units,
}: {
  range: NormalizationRange;
  units?: string | null;
}) {
  const diverging = range.symmetric;
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.03] px-4 py-3">
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-300">
          Predicted Cortical Response
        </span>
        <span className="text-[10px] text-slate-500">{units ?? "model units (unitless)"}</span>
      </div>

      <div
        className="mt-2.5 h-2.5 w-full rounded-full ring-1 ring-inset ring-white/10"
        style={{ background: gradientCss(diverging ? "diverging" : "sequential") }}
      />

      <div className="mt-1.5 flex justify-between text-[11px] tabular-nums text-slate-400">
        <span>{range.low.toFixed(3)}</span>
        {diverging && <span className="text-slate-500">0</span>}
        <span>{range.high.toFixed(3)}</span>
      </div>

      <div className="mt-1 flex justify-between text-[10px] text-slate-500">
        <span>{diverging ? "Below baseline" : "Lower"}</span>
        <span>{diverging ? "Above baseline" : "Higher"}</span>
      </div>

      <p className="mt-2.5 border-t border-white/5 pt-2 text-[10px] leading-relaxed text-slate-500">
        Scale: {range.label}. Normalization affects display only — stored predictions are
        unmodified. Sign indicates the direction of the modeled response, not whether it is
        good or bad.
      </p>
    </div>
  );
}
