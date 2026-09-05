"use client";

/**
 * Loading and error surfaces.
 *
 * Loading shows real per-step progress rather than a spinner, because mesh and
 * prediction fetches are distinct and a stalled step should be identifiable.
 *
 * Errors are explicit. A run whose predictions do not align with the mesh must
 * fail visibly: rendering a plausible-but-wrong brain is the worst outcome for
 * a scientific tool.
 */

export function BrainLoadingState({
  progress,
}: {
  progress: { step: string; done: number; total: number } | null;
}) {
  const steps = [
    "Contacting visualization API",
    "Reading run metadata",
    "Loading cortical mesh",
    "Loading predicted responses",
    "Preparing renderer",
  ];
  const done = progress?.done ?? 0;

  return (
    <div className="flex h-full min-h-[420px] items-center justify-center">
      <div className="w-[300px]">
        <div className="text-sm font-medium text-slate-200">Loading neural simulation</div>
        <ul className="mt-4 space-y-2">
          {steps.slice(1).map((step, i) => {
            const complete = i < done;
            const active = i === done;
            return (
              <li key={step} className="flex items-center gap-2.5 text-[12px]">
                <span
                  className={
                    complete
                      ? "text-emerald-400"
                      : active
                        ? "animate-pulse text-sky-400"
                        : "text-slate-600"
                  }
                >
                  {complete ? "✓" : active ? "•" : "○"}
                </span>
                <span className={complete || active ? "text-slate-300" : "text-slate-600"}>
                  {step}
                </span>
              </li>
            );
          })}
        </ul>
        <div className="mt-5 h-0.5 w-full overflow-hidden rounded-full bg-white/5">
          <div
            className="h-full rounded-full bg-sky-400/70 transition-all duration-500"
            style={{ width: `${(done / 4) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}

export function BrainErrorState({ message }: { message: string }) {
  return (
    <div className="flex h-full min-h-[420px] items-center justify-center p-8">
      <div className="max-w-md rounded-xl border border-red-500/25 bg-red-500/[0.06] p-5">
        <div className="text-sm font-medium text-red-300">Cannot display this run</div>
        <p className="mt-2 text-[12px] leading-relaxed text-slate-300">{message}</p>
        <p className="mt-3 border-t border-red-500/15 pt-3 text-[11px] leading-relaxed text-slate-500">
          The visualization refuses to render when predictions cannot be mapped onto the
          cortical surface with confidence. Showing a misaligned brain would be worse than
          showing nothing.
        </p>
      </div>
    </div>
  );
}
