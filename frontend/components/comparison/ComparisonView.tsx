"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { AppHeader } from "@/components/layout/AppHeader";
import { BrainViewer } from "@/components/brain/BrainViewer";
import { createComparison, fetchComparison, fetchComparisonArray, fetchRuns, loadDataset } from "@/lib/api";
import { rankRegionDifferences, sparklinePoints } from "@/lib/comparison";
import type { NeuralComparisonSummary } from "@/types/neural";
import type { HemisphereView, NeuralDataset, NormalizationMode, RunListItem } from "@/types/neural";

export function ComparisonView({ comparisonId }: { comparisonId: string }) {
  const [comparison, setComparison] = useState<NeuralComparisonSummary | null>(null);
  const [candidateId, setCandidateId] = useState("");
  const [l2, setL2] = useState<Float64Array | null>(null);
  const [times, setTimes] = useState<Float64Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [reference, setReference] = useState("");
  const [candidate, setCandidate] = useState("");
  const [creating, setCreating] = useState(false);
  const [alignmentMethod, setAlignmentMethod] = useState<"exact_observed_intersection" | "linear_candidate_to_reference">("exact_observed_intersection");
  const [maxGap, setMaxGap] = useState(10);
  const [dataset, setDataset] = useState<NeuralDataset | null>(null);
  const [difference, setDifference] = useState<Float32Array | null>(null);
  const [differenceMode, setDifferenceMode] = useState<"signed" | "absolute">("signed");
  const [differenceIndex, setDifferenceIndex] = useState(0);
  const [normalization, setNormalization] = useState<NormalizationMode>("robust_global");
  const [view, setView] = useState<HemisphereView>("both");

  useEffect(() => { fetchRuns().then(setRuns).catch(() => undefined); }, []);

  useEffect(() => {
    if (comparisonId === "new") return;
    const controller = new AbortController();
    fetchComparison(comparisonId, controller.signal)
      .then((result) => {
        setComparison(result);
        setCandidateId(result.metadata.candidate_run_ids[0] ?? "");
      })
      .catch((reason: unknown) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError")) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => controller.abort();
  }, [comparisonId]);

  useEffect(() => {
    if (!candidateId) return;
    const controller = new AbortController();
    setL2(null);
    setTimes(null);
    setError(null);
    Promise.all([
      fetchComparisonArray(comparisonId, candidateId, "cortical_l2_difference", controller.signal),
      fetchComparisonArray(comparisonId, candidateId, "times", controller.signal),
    ])
      .then(([values, observedTimes]) => {
        if (values.length !== observedTimes.length) throw new Error("Comparison arrays are misaligned.");
        setL2(values);
        setTimes(observedTimes);
      })
      .catch((reason: unknown) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError")) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => controller.abort();
  }, [candidateId, comparisonId]);

  useEffect(() => {
    if (!comparison || !candidateId) return;
    const controller = new AbortController();
    setDataset(null);
    setDifference(null);
    Promise.all([
      loadDataset(comparison.metadata.reference_run_id, { surface: "pial", signal: controller.signal }),
      fetchComparisonArray(comparisonId, candidateId, `cortical_${differenceMode}_delta`, controller.signal),
    ]).then(([mesh, values]) => {
      if (!times || values.length !== times.length * mesh.vertexCount) throw new Error("Difference map does not match its declared timeline and cortical mesh.");
      setDataset(mesh);
      setDifference(Float32Array.from(values));
      setDifferenceIndex(0);
    }).catch((reason: unknown) => {
      if (!(reason instanceof DOMException && reason.name === "AbortError")) setError(String(reason));
    });
    return () => controller.abort();
  }, [candidateId, comparison, comparisonId, differenceMode, times]);

  const pair = comparison?.pairs.find((item) => item.candidate_run_id === candidateId);
  const rankedRegions = useMemo(
    () => rankRegionDifferences(pair?.regions ?? [], 12),
    [pair],
  );

  return (
    <div className="min-h-screen">
      <AppHeader subtitle="Controlled Neural Comparison" />
      <main className="mx-auto max-w-[1200px] px-6 py-8">
        <Link href="/" className="text-xs text-slate-500 hover:text-sky-300">← All simulations</Link>
        <form className="mt-5 flex flex-wrap gap-2" onSubmit={(event) => {
          event.preventDefault();
          if (!reference || !candidate || reference === candidate) return;
          setCreating(true);
          createComparison({ reference_run_id: reference, candidate_run_ids: [candidate], alignment_method: alignmentMethod, max_interpolation_gap_seconds: maxGap })
            .then((created) => { window.location.assign(`/comparisons/${created.comparison_id}`); })
            .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
            .finally(() => setCreating(false));
        }}>
          <select aria-label="Reference run" value={reference} onChange={(event) => setReference(event.target.value)} className="rounded border border-white/10 bg-slate-950 px-3 py-2 text-xs"><option value="">Reference run…</option>{runs.map((run) => <option key={run.run_id}>{run.run_id}</option>)}</select>
          <select aria-label="Candidate run" value={candidate} onChange={(event) => setCandidate(event.target.value)} className="rounded border border-white/10 bg-slate-950 px-3 py-2 text-xs"><option value="">Candidate run…</option>{runs.map((run) => <option key={run.run_id}>{run.run_id}</option>)}</select>
          <select aria-label="Timeline alignment" value={alignmentMethod} onChange={(event) => setAlignmentMethod(event.target.value as typeof alignmentMethod)} className="rounded border border-white/10 bg-slate-950 px-3 py-2 text-xs"><option value="exact_observed_intersection">Exact observed overlap</option><option value="linear_candidate_to_reference">Linear interpolation (opt-in)</option></select>
          {alignmentMethod === "linear_candidate_to_reference" && <input aria-label="Maximum interpolation gap seconds" type="number" min="0.001" step="0.1" value={maxGap} onChange={(event) => setMaxGap(Number(event.target.value))} className="w-28 rounded border border-white/10 bg-slate-950 px-3 py-2 text-xs" />}
          <button disabled={creating || !reference || !candidate || reference === candidate} className="rounded bg-sky-500 px-3 py-2 text-xs font-medium text-slate-950 disabled:opacity-40">{creating ? "Creating…" : "Create comparison"}</button>
        </form>
        {error && <div className="mt-6 rounded-lg border border-red-500/30 p-4 text-sm text-red-300">{error}</div>}
        {!comparison && !error && comparisonId !== "new" && <div className="mt-8 text-sm text-slate-500">Loading comparison…</div>}
        {comparisonId === "new" && <p className="mt-4 text-xs text-slate-500">Choose two compatible completed runs. Exact observed overlap is the default.</p>}
        {comparison && pair && (
          <>
            <div className="mt-5 flex flex-wrap items-end justify-between gap-4">
              <div>
                <h1 className="text-2xl font-semibold text-white">Candidate minus reference</h1>
                <p className="mt-1 font-mono text-[11px] text-slate-500">A: {comparison.metadata.reference_run_id}</p>
              </div>
              <label className="text-[11px] uppercase tracking-wider text-slate-500">
                Candidate
                <select value={candidateId} onChange={(event) => setCandidateId(event.target.value)} className="ml-3 rounded border border-white/10 bg-slate-950 px-3 py-2 font-mono normal-case text-slate-200">
                  {comparison.metadata.candidate_run_ids.map((id) => <option key={id}>{id}</option>)}
                </select>
              </label>
            </div>
            <div className="mt-6 grid gap-4 sm:grid-cols-3">
              <Stat label={pair.alignment.interpolation_applied ? "Aligned samples (includes interpolation)" : "Aligned observed samples"} value={`${pair.alignment.aligned_samples}`} />
              <Stat label="Reference coverage" value={`${(pair.alignment.reference_coverage_fraction * 100).toFixed(1)}%`} />
              <Stat label="Candidate coverage" value={`${(pair.alignment.candidate_coverage_fraction * 100).toFixed(1)}%`} />
            </div>
            <section className="mt-5 rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
              <h2 className="text-xs uppercase tracking-wider text-slate-400">Cortical pattern divergence</h2>
              <Sparkline values={l2} times={times} />
              <p className="mt-2 text-[10px] text-slate-600">{pair.alignment.interpolation_applied ? `${pair.alignment.interpolated_candidate_samples} candidate samples were linearly interpolated without extrapolation.` : "L2 difference at matched observed timestamps. No samples were interpolated."}</p>
            </section>
            {dataset && difference && times && (
              <section className="mt-5 rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
                <div className="flex items-center justify-between"><h2 className="text-xs uppercase tracking-wider text-slate-400">Cortical difference map</h2><select value={differenceMode} onChange={(event) => setDifferenceMode(event.target.value as "signed" | "absolute")} className="rounded border border-white/10 bg-slate-950 px-2 py-1 text-xs"><option value="signed">Signed Δ</option><option value="absolute">Absolute |Δ|</option></select></div>
                <div className="mt-3 h-[460px]"><BrainViewer hemispheres={dataset.hemispheres} values={difference} vertexCount={dataset.vertexCount} predictionIndex={differenceIndex} scale={differenceScale(difference, differenceMode)} medialWall={dataset.medialWall} currentTime={times[differenceIndex] ?? 0} normalization={normalization} onNormalizationChange={setNormalization} view={view} onViewChange={setView} separateHemispheres={false} /></div>
                <input aria-label="Difference timestamp" className="mt-3 w-full" type="range" min={0} max={Math.max(0, times.length - 1)} value={differenceIndex} onChange={(event) => setDifferenceIndex(Number(event.target.value))} />
                <p className="text-[10px] text-slate-600">Candidate minus reference on the verified shared cortical vertex order. Absolute mode shows magnitude only.</p>
              </section>
            )}
            <div className="mt-5 grid gap-5 lg:grid-cols-2">
              <section className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
                <h2 className="text-xs uppercase tracking-wider text-slate-400">Largest regional differences</h2>
                <div className="mt-3 space-y-2">{rankedRegions.map((region) => <div key={region.region_id} className="flex justify-between gap-4 text-xs"><span className="truncate text-slate-400">{region.name} ({region.hemisphere[0]?.toUpperCase()})</span><span className="font-mono text-slate-200">|Δ| {region.mean_absolute_delta.toFixed(4)} · Δ {region.mean_signed_delta.toFixed(4)}</span></div>)}</div>
              </section>
              <section className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
                <h2 className="text-xs uppercase tracking-wider text-slate-400">Ranked divergence events</h2>
                <div className="mt-3 space-y-2">{pair.events.slice(0, 12).map((event) => <div key={event.rank} className="flex justify-between text-xs"><span className="text-slate-400">#{event.rank} · {event.timestamp_seconds.toFixed(2)}s</span><span className="font-mono text-slate-200">L2 {event.cortical_l2_difference.toFixed(3)}</span></div>)}</div>
              </section>
            </div>
            <div className="mt-5 grid gap-5 lg:grid-cols-2">
              <section className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5"><h2 className="text-xs uppercase tracking-wider text-slate-400">Yeo-7 functional networks</h2><div className="mt-3 space-y-2">{pair.networks.map((network) => <div key={network.network_id} className="flex justify-between text-xs"><span className="text-slate-400">{network.name}</span><span className="font-mono">|Δ| {network.mean_absolute_delta.toFixed(4)} · Δ {network.mean_signed_delta.toFixed(4)}</span></div>)}</div></section>
              <section className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5"><h2 className="text-xs uppercase tracking-wider text-slate-400">Content features</h2><p className="mt-2 text-[10px] text-slate-600">{pair.content_status}</p><div className="mt-3 space-y-2">{pair.content_features.slice(0, 12).map((feature) => <div key={feature.name} className="flex justify-between text-xs"><span className="text-slate-400">{feature.name} · {(feature.jointly_available_fraction * 100).toFixed(0)}% available</span><span className="font-mono">{feature.mean_signed_delta === null ? "unavailable" : `Δ ${feature.mean_signed_delta.toFixed(4)}`}</span></div>)}</div></section>
            </div>
            <p className="mt-5 text-[11px] leading-relaxed text-slate-600">{comparison.metadata.interpretation_notice}</p>
          </>
        )}
      </main>
    </div>
  );
}

function differenceScale(values: Float32Array, mode: "signed" | "absolute") {
  let maximum = 0;
  for (const value of values) if (Number.isFinite(value)) maximum = Math.max(maximum, Math.abs(value));
  maximum = maximum || 1;
  return { min: mode === "signed" ? -maximum : 0, max: maximum, mean: 0, std: maximum / 2, p01: mode === "signed" ? -maximum : 0, p50: 0, p99: maximum, abs_max: maximum, negative_fraction: 0, diverging_recommended: mode === "signed" };
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4"><div className="text-[10px] uppercase tracking-wider text-slate-600">{label}</div><div className="mt-2 font-mono text-lg text-sky-300">{value}</div></div>;
}

function Sparkline({ values, times }: { values: Float64Array | null; times: Float64Array | null }) {
  if (!values?.length || !times?.length) return <div className="mt-4 text-xs text-slate-600">No values.</div>;
  const points = sparklinePoints(times, values);
  return <svg viewBox="0 0 800 110" className="mt-4 h-28 w-full"><polyline points={points} fill="none" stroke="rgb(56,189,248)" strokeWidth="2" /></svg>;
}
