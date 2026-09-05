"use client";

/**
 * Compact display controls: camera presets, hemisphere view, normalization.
 *
 * Deliberately small. The default experience should be correct without any
 * adjustment; these exist for inspection, not configuration for its own sake.
 */

import type { HemisphereView, NormalizationMode } from "@/types/neural";
import type { CameraPreset } from "@/components/brain/BrainScene";

const SEGMENT =
  "px-2.5 py-1 text-[11px] font-medium transition-colors first:rounded-l-md last:rounded-r-md";

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-[11px] text-slate-500">{label}</span>
      <div className="flex rounded-md border border-white/10 bg-white/[0.02]">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={value === option.value}
            className={`${SEGMENT} ${
              value === option.value
                ? "bg-sky-500/15 text-sky-200"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export interface BrainControlsProps {
  view: HemisphereView;
  onViewChange: (view: HemisphereView) => void;
  cameraPreset: CameraPreset;
  onCameraPresetChange: (preset: CameraPreset) => void;
  normalization: NormalizationMode;
  onNormalizationChange: (mode: NormalizationMode) => void;
  onResetCamera: () => void;
}

export function BrainControls({
  view,
  onViewChange,
  cameraPreset,
  onCameraPresetChange,
  normalization,
  onNormalizationChange,
  onResetCamera,
}: BrainControlsProps) {
  return (
    <div className="space-y-2.5 rounded-lg border border-white/10 bg-white/[0.03] px-4 py-3">
      <Segmented<HemisphereView>
        label="Hemisphere"
        value={view}
        onChange={onViewChange}
        options={[
          { value: "both", label: "Both" },
          { value: "left", label: "Left" },
          { value: "right", label: "Right" },
        ]}
      />
      <Segmented<CameraPreset>
        label="View"
        value={cameraPreset}
        onChange={onCameraPresetChange}
        options={[
          { value: "lateral", label: "Lateral" },
          { value: "medial", label: "Medial" },
          { value: "superior", label: "Top" },
          { value: "anterior", label: "Front" },
        ]}
      />
      <Segmented<NormalizationMode>
        label="Scale"
        value={normalization}
        onChange={onNormalizationChange}
        options={[
          { value: "robust_global", label: "Robust" },
          { value: "global", label: "Global" },
          { value: "per_frame", label: "Frame" },
        ]}
      />
      <button
        type="button"
        onClick={onResetCamera}
        className="w-full rounded-md border border-white/10 px-3 py-1.5 text-[11px] font-medium text-slate-300 transition-colors hover:border-white/20 hover:text-white"
      >
        Reset view
      </button>
    </div>
  );
}
