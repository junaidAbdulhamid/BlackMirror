/**
 * Loads a run into the store.
 *
 * Kept out of components so the page stays declarative and the same loading
 * path can later serve two runs side by side for A/B comparison.
 */

import { useEffect } from "react";
import { ApiError, loadDataset } from "@/lib/api";
import { useNeuralStore } from "@/stores/neuralVisualizationStore";
import type { SurfaceKind } from "@/types/neural";

export function useNeuralPrediction(runId: string, surface: SurfaceKind = "inflated"): void {
  const setDataset = useNeuralStore((s) => s.setDataset);
  const setStatus = useNeuralStore((s) => s.setStatus);
  const setLoadProgress = useNeuralStore((s) => s.setLoadProgress);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setLoadProgress({ step: "Contacting visualization API", done: 0, total: 4 });

    loadDataset(runId, {
      surface,
      signal: controller.signal,
      onProgress: setLoadProgress,
    })
      .then((dataset) => {
        if (controller.signal.aborted) return;
        setDataset(dataset);
        setLoadProgress(null);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message =
          error instanceof ApiError
            ? [error.message, error.detail].filter(Boolean).join(" — ")
            : error instanceof Error
              ? error.message
              : "Unknown error while loading the run.";
        setStatus("error", message);
      });

    return () => controller.abort();
  }, [runId, surface, setDataset, setStatus, setLoadProgress]);
}
