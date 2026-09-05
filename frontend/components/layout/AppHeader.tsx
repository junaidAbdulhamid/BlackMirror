import Link from "next/link";

export function AppHeader({ subtitle }: { subtitle?: string }) {
  return (
    <header className="border-b border-white/[0.07]">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
        <Link href="/" className="group flex items-baseline gap-3">
          <span className="text-[15px] font-semibold tracking-tight text-white">
            NeuroSplit
          </span>
          <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500 transition-colors group-hover:text-slate-400">
            {subtitle ?? "Neural Simulation"}
          </span>
        </Link>
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.14em] text-slate-600">
          <span className="rounded border border-white/10 px-2 py-1">Phase 2</span>
          <span className="rounded border border-white/10 px-2 py-1">TRIBE v2</span>
        </div>
      </div>
    </header>
  );
}
