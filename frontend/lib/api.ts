/**
 * Client for the Phase 2 visualization API.
 *
 * Every call is a plain read of saved Phase 1 artifacts. Nothing here triggers
 * inference — that separation is the core architectural rule of Phase 2.
 *
 * Arrays arrive as raw binary and are wrapped in typed arrays with zero copying
 * and zero JSON parsing.
 */

import type {
  HemisphereGeometry,
  NeuralDataset,
  RunListItem,
  RunSummary,
  NeuralAnalyticsSummary,
  NeuralComparisonSummary,
  SurfaceKind,
} from "@/types/neural";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    let detail: string | undefined;
    try {
      detail = ((await response.json()) as { detail?: string }).detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(
      describeStatus(response.status, url),
      response.status,
      detail,
    );
  }
  return (await response.json()) as T;
}

function describeStatus(status: number, url: string): string {
  if (status === 404) return "Run not found.";
  if (status === 409) {
    return "This run cannot be visualized: its predictions do not align with the cortical mesh.";
  }
  if (status === 422) return "This run's artifacts are incomplete.";
  return `Request failed (${status}): ${url}`;
}

async function getBuffer(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new ApiError(describeStatus(response.status, url), response.status);
  return response.arrayBuffer();
}

export async function fetchRuns(signal?: AbortSignal): Promise<RunListItem[]> {
  return getJson<RunListItem[]>("/api/runs", signal);
}

export async function fetchRunSummary(runId: string, signal?: AbortSignal): Promise<RunSummary> {
  return getJson<RunSummary>(`/api/runs/${encodeURIComponent(runId)}`, signal);
}

export function stimulusUrl(runId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/stimulus`;
}

export async function fetchAnalytics(
  runId: string,
  signal?: AbortSignal,
): Promise<NeuralAnalyticsSummary> {
  return getJson<NeuralAnalyticsSummary>(
    `/api/runs/${encodeURIComponent(runId)}/analytics`,
    signal,
  );
}

export async function fetchAnalyticsArray(
  runId: string,
  key: string,
  signal?: AbortSignal,
): Promise<Float64Array> {
  const url = `/api/runs/${encodeURIComponent(runId)}/analytics/arrays/${encodeURIComponent(key)}`;
  const response = await fetch(url, { signal });
  if (!response.ok) throw new ApiError(describeStatus(response.status, url), response.status);
  const dtype = response.headers.get("X-Array-Dtype");
  if (dtype !== "float64") {
    throw new ApiError(`Analytics array '${key}' has unsupported dtype '${dtype ?? "missing"}'.`, 500);
  }
  const shape = response.headers
    .get("X-Array-Shape")
    ?.split(",")
    .map(Number);
  if (!shape?.length || shape.some((dimension) => !Number.isInteger(dimension) || dimension < 0)) {
    throw new ApiError(`Analytics array '${key}' has an invalid or missing shape header.`, 500);
  }
  const buffer = await response.arrayBuffer();
  const expectedBytes = shape.reduce((product, dimension) => product * dimension, 1) * 8;
  if (buffer.byteLength !== expectedBytes) {
    throw new ApiError(
      `Analytics array '${key}' has ${buffer.byteLength} bytes; metadata declares ${expectedBytes}.`,
      500,
    );
  }
  return new Float64Array(buffer);
}

export async function fetchComparisons(
  signal?: AbortSignal,
): Promise<NeuralComparisonSummary[]> {
  return getJson<NeuralComparisonSummary[]>("/api/comparisons", signal);
}

export async function createComparison(request: {
  reference_run_id: string;
  candidate_run_ids: string[];
  alignment_method: "exact_observed_intersection" | "linear_candidate_to_reference";
  max_interpolation_gap_seconds: number;
}): Promise<NeuralComparisonSummary> {
  const response = await fetch("/api/comparisons", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new ApiError(describeStatus(response.status, "/api/comparisons"), response.status);
  return (await response.json()) as NeuralComparisonSummary;
}

export async function fetchComparison(
  comparisonId: string,
  signal?: AbortSignal,
): Promise<NeuralComparisonSummary> {
  return getJson<NeuralComparisonSummary>(
    `/api/comparisons/${encodeURIComponent(comparisonId)}`,
    signal,
  );
}

export async function fetchComparisonArray(
  comparisonId: string,
  candidateRunId: string,
  key: string,
  signal?: AbortSignal,
): Promise<Float64Array> {
  const url = `/api/comparisons/${encodeURIComponent(comparisonId)}/pairs/${encodeURIComponent(candidateRunId)}/arrays/${encodeURIComponent(key)}`;
  const response = await fetch(url, { signal });
  if (!response.ok) throw new ApiError(describeStatus(response.status, url), response.status);
  if (response.headers.get("X-Array-Dtype") !== "float64") {
    throw new ApiError(`Comparison array '${key}' is not float64.`, 500);
  }
  const shape = response.headers.get("X-Array-Shape")?.split(",").map(Number);
  if (!shape?.length || shape.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new ApiError(`Comparison array '${key}' has an invalid or missing shape header.`, 500);
  }
  const buffer = await response.arrayBuffer();
  const expectedBytes = shape.reduce((product, value) => product * value, 1) * 8;
  if (buffer.byteLength !== expectedBytes) {
    throw new ApiError(
      `Comparison array '${key}' has ${buffer.byteLength} bytes; metadata declares ${expectedBytes}.`,
      500,
    );
  }
  return new Float64Array(buffer);
}

export interface LoadProgress {
  step: string;
  done: number;
  total: number;
}

/**
 * Load everything needed to render a run.
 *
 * Mesh and prediction buffers are fetched in parallel, then validated against
 * each other before anything is handed to the renderer.
 */
export async function loadDataset(
  runId: string,
  options: { surface?: SurfaceKind; signal?: AbortSignal; onProgress?: (p: LoadProgress) => void } = {},
): Promise<NeuralDataset> {
  const { surface = "inflated", signal, onProgress } = options;
  const base = `/api/runs/${encodeURIComponent(runId)}`;
  const total = 4;
  const report = (step: string, done: number) => onProgress?.({ step, done, total });

  report("Reading run metadata", 0);
  const summary = await fetchRunSummary(runId, signal);

  report("Loading cortical mesh", 1);
  const hemisphereNames = summary.cortical.hemispheres.map((h) => h.name as "left" | "right");
  const meshBuffers = await Promise.all(
    hemisphereNames.flatMap((name) => [
      getBuffer(`${base}/mesh/${name}/${surface}/vertices`, signal),
      getBuffer(`${base}/mesh/${name}/faces`, signal),
    ]),
  );

  report("Loading predicted responses", 2);
  const [predictionBuffer, timelineBuffer, durationBuffer, wallBuffer] = await Promise.all([
    getBuffer(`${base}/predictions`, signal),
    getBuffer(`${base}/timeline`, signal),
    getBuffer(`${base}/durations`, signal),
    summary.cortical.medial_wall_available
      ? getBuffer(`${base}/mesh/medial-wall`, signal)
      : Promise.resolve(null),
  ]);

  report("Preparing renderer", 3);
  const predictions = new Float32Array(predictionBuffer);
  const timeline = new Float32Array(timelineBuffer);
  const durations = new Float32Array(durationBuffer);
  const medialWall = wallBuffer ? new Uint8Array(wallBuffer) : null;

  const timePoints = summary.temporal.n_time_points;
  const vertexCount = summary.cortical.vertex_count;

  // Validate before rendering: a plausible-looking but misaligned brain is the
  // worst possible failure mode for a scientific tool.
  if (predictions.length !== timePoints * vertexCount) {
    throw new ApiError(
      `Prediction buffer holds ${predictions.length} values but the manifest declares ` +
        `${timePoints} x ${vertexCount} = ${timePoints * vertexCount}.`,
      500,
    );
  }
  if (timeline.length !== timePoints) {
    throw new ApiError(
      `Timeline has ${timeline.length} entries but there are ${timePoints} prediction rows.`,
      500,
    );
  }
  if (durations.length !== timePoints) {
    throw new ApiError(
      `Durations has ${durations.length} entries but there are ${timePoints} prediction rows.`,
      500,
    );
  }
  if (medialWall && medialWall.length !== vertexCount) {
    throw new ApiError(
      `Medial-wall mask has ${medialWall.length} entries but there are ${vertexCount} vertices.`,
      500,
    );
  }

  const hemispheres: HemisphereGeometry[] = hemisphereNames.map((name, i) => {
    const positions = new Float32Array(meshBuffers[i * 2] as ArrayBuffer);
    const indices = new Uint32Array(new Int32Array(meshBuffers[i * 2 + 1] as ArrayBuffer));
    const range = summary.cortical.hemispheres.find((h) => h.name === name);
    if (!range) throw new ApiError(`Missing hemisphere range for '${name}'.`, 500);

    const meshVertices = positions.length / 3;
    const expected = range.end - range.start;
    if (meshVertices !== expected) {
      throw new ApiError(
        `Hemisphere '${name}': mesh has ${meshVertices} vertices but predictions supply ` +
          `${expected}. Refusing to render misaligned data.`,
        409,
      );
    }
    for (let k = 0; k < indices.length; k += 1) {
      if ((indices[k] as number) >= meshVertices) {
        throw new ApiError(
          `Hemisphere '${name}': face index ${indices[k]} exceeds vertex count ${meshVertices}.`,
          500,
        );
      }
    }
    return {
      name,
      positions,
      indices,
      vertexCount: meshVertices,
      predictionOffset: range.start,
    };
  });

  return {
    summary,
    predictions,
    timePoints,
    vertexCount,
    timeline,
    durations,
    hemispheres,
    medialWall,
  };
}

/**
 * Phase 4 content analysis.
 *
 * Returns null when the run has not been analysed — a 404 here is an expected
 * state, not an error: content analysis is derived on demand via
 * `blackmirror analyze-content` and takes tens of seconds.
 */
export async function fetchContentAnalysis(
  runId: string,
  signal?: AbortSignal,
): Promise<import("@/types/neural").ContentAnalysis | null> {
  const response = await fetch(
    `/api/runs/${encodeURIComponent(runId)}/content-analysis`,
    { signal },
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new ApiError(describeStatus(response.status, "content-analysis"), response.status);
  }
  return (await response.json()) as import("@/types/neural").ContentAnalysis;
}

/** URL for an extracted keyframe image. */
export function keyframeUrl(runId: string, name: string): string {
  const file = name.replace(/^keyframes\//, "");
  return `/api/runs/${encodeURIComponent(runId)}/content-analysis/keyframes/${encodeURIComponent(file)}`;
}
