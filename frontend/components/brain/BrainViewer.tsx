"use client";

/**
 * A self-contained cortical viewer.
 *
 * REUSABILITY IS THE POINT
 * ------------------------
 * It takes geometry plus a values array, and knows nothing about experiments,
 * pages or where the numbers came from. Phase 5 renders two of these side by
 * side; a delta map ΔR = R_B − R_A is just a different `values` array. Nothing
 * inside needs to change.
 *
 * IN:  hemispheres, a [T*V] (or [V]) values array, a frame index, a scale
 * OUT: an interactive, coloured, inspectable cortex
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  HemisphereGeometry,
  HemisphereView,
  NormalizationMode,
  ResponseScale,
} from "@/types/neural";
import {
  BrainScene,
  defaultPresetFor,
  type CameraPreset,
} from "@/components/brain/BrainScene";
import { BrainControls } from "@/components/brain/BrainControls";
import { BrainLegend } from "@/components/brain/BrainLegend";
import { BrainTooltip, type VertexReadout } from "@/components/brain/BrainTooltip";
import { buildColorLut } from "@/lib/colorMapping";
import {
  buildNormalizationRange,
  frameStatistics,
  normalizeValue,
} from "@/lib/normalization";

export interface BrainViewerProps {
  hemispheres: HemisphereGeometry[];
  /** Flat values. For a full run this is [T*V]; for a single map, [V]. */
  values: Float32Array;
  vertexCount: number;
  /** Row to display. */
  predictionIndex: number;
  scale: ResponseScale;
  medialWall: Uint8Array | null;
  currentTime: number;
  /**
   * Whether a prediction row genuinely covers `currentTime`.
   *
   * When false the surface is drawn inert instead of showing the nearest
   * row's values, which describe a different moment of the stimulus.
   */
  covered?: boolean;
  normalization: NormalizationMode;
  onNormalizationChange: (mode: NormalizationMode) => void;
  view: HemisphereView;
  onViewChange: (view: HemisphereView) => void;
  /** Inflated surfaces overlap and need artificial separation; pial does not. */
  separateHemispheres: boolean;
  showControls?: boolean;
  onPin?: (readout: VertexReadout | null) => void;
  onStats?: (stats: { fps: number; bufferMs: number }) => void;
  className?: string;
}

export function BrainViewer({
  hemispheres,
  values,
  vertexCount,
  predictionIndex,
  scale,
  medialWall,
  currentTime,
  covered = true,
  normalization,
  onNormalizationChange,
  view,
  onViewChange,
  separateHemispheres,
  showControls = true,
  onPin,
  onStats,
  className = "",
}: BrainViewerProps) {
  const paired = view === "both" && separateHemispheres;
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>(() =>
    defaultPresetFor(paired),
  );
  const [resetSignal, setResetSignal] = useState(0);

  // Switching between one and two hemispheres changes which views are useful:
  // a lateral camera would hide the far hemisphere when both are shown.
  useEffect(() => {
    setCameraPreset(defaultPresetFor(paired));
  }, [paired]);
  const [hover, setHover] = useState<VertexReadout | null>(null);
  const [invalidCounts, setInvalidCounts] = useState<Record<string, number>>({});
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fpsRef = useRef(0);

  const frameOffset = Math.max(0, predictionIndex) * vertexCount;

  // Recomputed only when the mode or the frame changes — not per rendered frame.
  const range = useMemo(() => {
    const stats =
      normalization === "per_frame"
        ? frameStatistics(values, Math.max(0, predictionIndex), vertexCount)
        : undefined;
    return buildNormalizationRange(normalization, scale, stats);
  }, [normalization, scale, values, predictionIndex, vertexCount]);

  const lut = useMemo(
    () => buildColorLut(range.symmetric ? "diverging" : "sequential"),
    [range.symmetric],
  );

  const normalize = useCallback(
    (value: number) => normalizeValue(value, range),
    [range],
  );

  const invalidTotal = useMemo(
    () => Object.values(invalidCounts).reduce((sum, n) => sum + n, 0),
    [invalidCounts],
  );

  const readoutFor = useCallback(
    (hemisphere: string, localIndex: number): VertexReadout | null => {
      const geometry = hemispheres.find((h) => h.name === hemisphere);
      if (!geometry) return null;
      const globalVertex = geometry.predictionOffset + localIndex;
      return {
        vertex: { vertex: globalVertex, hemisphere, localIndex },
        value: values[frameOffset + globalVertex] ?? Number.NaN,
        timeSeconds: currentTime,
        predictionIndex: Math.max(0, predictionIndex),
        isMedialWall: medialWall ? medialWall[globalVertex] === 1 : false,
      };
    },
    [hemispheres, values, frameOffset, currentTime, predictionIndex, medialWall],
  );

  return (
    <div className={`relative flex h-full flex-col ${className}`}>
      <div className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-white/10 bg-[#0a0c10]">
        <BrainScene
          hemispheres={hemispheres}
          values={values}
          frameOffset={frameOffset}
          normalize={normalize}
          lut={lut}
          medialWall={medialWall}
          view={view}
          separateHemispheres={separateHemispheres}
          covered={covered}
          cameraPreset={cameraPreset}
          resetSignal={resetSignal}
          onHover={(hemisphere, localIndex) =>
            setHover(localIndex === null ? null : readoutFor(hemisphere, localIndex))
          }
          onSelect={(hemisphere, localIndex) => onPin?.(readoutFor(hemisphere, localIndex))}
          onFps={(fps) => {
            fpsRef.current = fps;
            onStats?.({ fps, bufferMs: 0 });
          }}
          onBufferUpdate={(bufferMs) => onStats?.({ fps: fpsRef.current, bufferMs })}
          onInvalidCount={(hemisphere, count) =>
            setInvalidCounts((previous) =>
              previous[hemisphere] === count
                ? previous
                : { ...previous, [hemisphere]: count },
            )
          }
          onCanvasReady={(canvas) => {
            canvasRef.current = canvas;
          }}
        />

        {hover && (
          <div className="absolute left-3 top-3 z-10">
            <BrainTooltip readout={hover} />
          </div>
        )}

        {!covered && (
          <div className="pointer-events-none absolute inset-x-0 top-3 z-10 flex justify-center">
            <div className="rounded-md border border-amber-500/30 bg-[#0d1117]/95 px-3 py-1.5 text-[11px] text-amber-200 shadow-lg backdrop-blur">
              No prediction covers this moment — surface shown inert
            </div>
          </div>
        )}

        {invalidTotal > 0 && (
          <div className="pointer-events-none absolute bottom-3 left-3 z-10 rounded-md border border-fuchsia-500/30 bg-[#0d1117]/95 px-3 py-1.5 text-[11px] text-fuchsia-200 shadow-lg backdrop-blur">
            {invalidTotal.toLocaleString()} vertices have no valid value (NaN/Inf),
            drawn in magenta
          </div>
        )}
      </div>

      {showControls && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <BrainLegend range={range} />
          <BrainControls
            view={view}
            onViewChange={onViewChange}
            cameraPreset={cameraPreset}
            onCameraPresetChange={setCameraPreset}
            normalization={normalization}
            onNormalizationChange={onNormalizationChange}
            onResetCamera={() => setResetSignal((n) => n + 1)}
          />
        </div>
      )}
    </div>
  );
}
