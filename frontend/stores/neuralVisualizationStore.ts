/**
 * Single source of truth for visualization state.
 *
 * WHY A STORE
 * -----------
 * Time appears in four places: the video element, the scrubber, the temporal
 * graph, and the brain. If each derived its own notion of "now", they would
 * drift. Here `currentTime` is authoritative and everything else is derived
 * from it.
 *
 * WHAT IS DELIBERATELY *NOT* HERE
 * -------------------------------
 * Per-vertex colours. Those are ~61k floats per hemisphere per frame; putting
 * them in React state would re-render the tree on every animation frame. They
 * live in refs inside the renderer and are written straight into GPU buffers.
 * The store holds only small scalars.
 */

import { create } from "zustand";
import type {
  HemisphereView,
  NeuralDataset,
  NormalizationMode,
  SurfaceKind,
} from "@/types/neural";
import {
  coveringPredictionIndex,
  mapTimeToPredictionIndex,
  timelineBounds,
} from "@/lib/temporalMapping";

export interface PinnedVertex {
  /** Index on the prediction vertex axis (global, not hemisphere-local). */
  vertex: number;
  hemisphere: string;
  localIndex: number;
}

interface NeuralState {
  dataset: NeuralDataset | null;
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  loadProgress: { step: string; done: number; total: number } | null;

  // --- Time (authoritative) ---
  currentTime: number;
  isPlaying: boolean;
  playbackRate: number;
  duration: number;

  // --- Display settings ---
  normalizationMode: NormalizationMode;
  hemisphereView: HemisphereView;
  surface: SurfaceKind;
  showMedialWall: boolean;
  debugMode: boolean;

  // --- Inspection ---
  hoveredVertex: PinnedVertex | null;
  pinnedVertex: PinnedVertex | null;

  setDataset: (dataset: NeuralDataset) => void;
  setStatus: (status: NeuralState["status"], error?: string | null) => void;
  setLoadProgress: (progress: NeuralState["loadProgress"]) => void;
  setCurrentTime: (seconds: number) => void;
  setPlaying: (playing: boolean) => void;
  togglePlaying: () => void;
  setPlaybackRate: (rate: number) => void;
  seekToIndex: (index: number) => void;
  setNormalizationMode: (mode: NormalizationMode) => void;
  setHemisphereView: (view: HemisphereView) => void;
  setSurface: (surface: SurfaceKind) => void;
  setShowMedialWall: (show: boolean) => void;
  toggleDebug: () => void;
  setHoveredVertex: (vertex: PinnedVertex | null) => void;
  setPinnedVertex: (vertex: PinnedVertex | null) => void;
  reset: () => void;
}

const INITIAL = {
  dataset: null,
  status: "idle" as const,
  error: null,
  loadProgress: null,
  currentTime: 0,
  isPlaying: false,
  playbackRate: 1,
  duration: 0,
  normalizationMode: "robust_global" as NormalizationMode,
  hemisphereView: "both" as HemisphereView,
  surface: "inflated" as SurfaceKind,
  showMedialWall: true,
  debugMode: false,
  hoveredVertex: null,
  pinnedVertex: null,
};

export const useNeuralStore = create<NeuralState>((set, get) => ({
  ...INITIAL,

  setDataset: (dataset) => {
    // Duration is the longer of the stimulus and the covered prediction window,
    // so a prediction row is never unreachable by the scrubber.
    const bounds = timelineBounds(dataset.timeline, dataset.summary.temporal.tr_seconds);
    const stimulusDuration = dataset.summary.stimulus.duration_seconds ?? 0;
    set({
      dataset,
      status: "ready",
      error: null,
      duration: Math.max(stimulusDuration, bounds.end),
      currentTime: 0,
      isPlaying: false,
    });
  },

  setStatus: (status, error = null) => set({ status, error }),
  setLoadProgress: (loadProgress) => set({ loadProgress }),

  setCurrentTime: (seconds) => {
    const { duration } = get();
    set({ currentTime: Math.min(Math.max(seconds, 0), duration || seconds) });
  },

  setPlaying: (isPlaying) => set({ isPlaying }),
  togglePlaying: () => set((s) => ({ isPlaying: !s.isPlaying })),
  setPlaybackRate: (playbackRate) => set({ playbackRate }),

  seekToIndex: (index) => {
    const { dataset } = get();
    if (!dataset) return;
    const clamped = Math.min(Math.max(index, 0), dataset.timePoints - 1);
    set({ currentTime: dataset.timeline[clamped] as number });
  },

  setNormalizationMode: (normalizationMode) => set({ normalizationMode }),
  setHemisphereView: (hemisphereView) => set({ hemisphereView }),
  setSurface: (surface) => set({ surface }),
  setShowMedialWall: (showMedialWall) => set({ showMedialWall }),
  toggleDebug: () => set((s) => ({ debugMode: !s.debugMode })),
  setHoveredVertex: (hoveredVertex) => set({ hoveredVertex }),
  setPinnedVertex: (pinnedVertex) => set({ pinnedVertex }),
  reset: () => set(INITIAL),
}));

/** Nearest prediction row to the current time. -1 when no dataset is loaded. */
export function useCurrentPredictionIndex(): number {
  return useNeuralStore((state) =>
    state.dataset ? mapTimeToPredictionIndex(state.currentTime, state.dataset.timeline) : -1,
  );
}

/**
 * Whether a prediction row genuinely covers the current moment.
 *
 * Distinct from "a row is nearby". During a gap in a sparse timeline the
 * nearest row exists but describes a different part of the stimulus, so the
 * renderer must not present it as the response to what is on screen now.
 */
export function useIsCovered(): boolean {
  return useNeuralStore((state) =>
    state.dataset
      ? coveringPredictionIndex(
          state.currentTime,
          state.dataset.timeline,
          state.dataset.durations,
        ) >= 0
      : false,
  );
}
