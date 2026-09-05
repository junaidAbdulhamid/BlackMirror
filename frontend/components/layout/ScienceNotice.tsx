"use client";

/**
 * "What am I seeing?" — the scientific framing, always one click away.
 *
 * Not optional decoration: it is the guard against the single most likely
 * misreading of this product, that the picture is a scan of someone's brain.
 */

import { useState } from "react";

export function ScienceNotice({ notice }: { notice: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-md border border-white/10 px-2.5 py-1 text-[11px] text-slate-400 transition-colors hover:border-white/25 hover:text-slate-200"
      >
        <span className="grid h-3.5 w-3.5 place-items-center rounded-full border border-current text-[9px]">
          i
        </span>
        What am I seeing?
      </button>

      {open && (
        <div className="absolute right-0 z-30 mt-2 w-[380px] rounded-xl border border-white/10 bg-[#0d1117]/97 p-4 shadow-2xl backdrop-blur">
          <div className="text-[12px] font-semibold text-slate-100">
            Predicted cortical response
          </div>
          <p className="mt-2 text-[11.5px] leading-relaxed text-slate-400">
            This visualization displays cortical fMRI responses <em>predicted</em> by Meta&apos;s
            TRIBE v2 model for the supplied stimulus. Colours represent modeled response
            values mapped onto a standard cortical surface (fsaverage5).
          </p>
          <p className="mt-2 text-[11.5px] leading-relaxed text-slate-400">{notice}</p>
          <ul className="mt-3 space-y-1.5 border-t border-white/5 pt-3 text-[11px] leading-relaxed text-slate-500">
            <li>• Predictions describe an <em>average</em> subject, not any individual.</li>
            <li>• BOLD is a slow metabolic proxy, not neuronal firing.</li>
            <li>
              • A response does not establish emotion, memory, attention or purchase intent.
            </li>
            <li>• Not for medical or diagnostic use.</li>
          </ul>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="mt-3 text-[11px] text-slate-500 hover:text-slate-300"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}
