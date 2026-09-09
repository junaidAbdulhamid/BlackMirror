/**
 * Stimulus time -> prediction row index.
 *
 * WHY THIS IS NOT DIVISION
 * ------------------------
 * TRIBE drops per-TR segments that contain no events. In a real run we observed
 * 4 rows kept out of 100 generated. The prediction timeline is therefore sparse
 * and may be non-uniform, so `Math.floor(time / TR)` is wrong: it would index
 * past the end of the array and silently desynchronise the brain from the video.
 *
 * The authoritative clock is `timeline[i]` — the per-row `stimulus_time_seconds`
 * written by Phase 1. We search it rather than compute an index.
 *
 * Because `output_is_stimulus_aligned` is true for TRIBE v2 (the model already
 * compensated the ~5 s hemodynamic lag during training), `timeline[i]` is
 * directly comparable with video playback time. No extra shift is applied here.
 */

/**
 * Index of the row whose time is closest to `seconds`.
 *
 * Uses binary search, so scrubbing stays O(log T) even for long stimuli.
 * Clamps at both ends: there is no index -1 and no index T.
 */
export function mapTimeToPredictionIndex(
  seconds: number,
  timeline: Float32Array | number[],
): number {
  const length = timeline.length;
  if (length === 0) return -1;
  if (length === 1) return 0;

  const first = timeline[0] as number;
  const last = timeline[length - 1] as number;
  if (!Number.isFinite(seconds) || seconds <= first) return 0;
  if (seconds >= last) return length - 1;

  // Find the first index whose time is >= seconds.
  let low = 0;
  let high = length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if ((timeline[mid] as number) < seconds) low = mid + 1;
    else high = mid;
  }

  // `low` is the upper neighbour; compare against the lower one.
  const upper = low;
  const lower = Math.max(0, low - 1);
  const upperDelta = Math.abs((timeline[upper] as number) - seconds);
  const lowerDelta = Math.abs((timeline[lower] as number) - seconds);
  return upperDelta < lowerDelta ? upper : lower;
}

/** Playback time of a prediction row. Clamped to valid rows. */
export function predictionIndexToTime(
  index: number,
  timeline: Float32Array | number[],
): number {
  if (timeline.length === 0) return 0;
  const clamped = Math.min(Math.max(index, 0), timeline.length - 1);
  return timeline[clamped] as number;
}

/**
 * How far `seconds` is from the row actually being shown.
 *
 * Surfaced in the UI because with a sparse timeline the nearest row can be
 * seconds away from the current frame, and the user deserves to know that
 * rather than believe the brain is tracking the video exactly.
 */
export function temporalGap(
  seconds: number,
  timeline: Float32Array | number[],
): number {
  const index = mapTimeToPredictionIndex(seconds, timeline);
  if (index < 0) return Number.POSITIVE_INFINITY;
  return Math.abs((timeline[index] as number) - seconds);
}

/**
 * Index of the row whose interval actually contains `seconds`, or -1.
 *
 * This is the honest question. `mapTimeToPredictionIndex` answers "which row is
 * nearest", which always returns something — even in the middle of a five-second
 * gap where no prediction exists. Row `i` covers
 * `[timeline[i], timeline[i] + durations[i])`, half-open, so a moment exactly on
 * the next row's start belongs to that next row.
 *
 * Distance-based coverage is wrong in a way that is easy to miss: a moment one
 * second *before* a row is near it, but is not covered by it.
 */
export function coveringPredictionIndex(
  seconds: number,
  timeline: Float32Array | number[],
  durations: Float32Array | number[],
): number {
  if (timeline.length === 0 || !Number.isFinite(seconds)) return -1;

  // The covering row is the nearest row or one of its neighbours, so a small
  // window avoids a linear scan without assuming uniform spacing.
  const nearest = mapTimeToPredictionIndex(seconds, timeline);
  const lo = Math.max(0, nearest - 1);
  const hi = Math.min(timeline.length - 1, nearest + 1);
  for (let i = lo; i <= hi; i += 1) {
    const start = timeline[i] as number;
    const duration = (durations[i] as number) ?? 0;
    if (seconds >= start && seconds < start + duration) return i;
  }
  return -1;
}

/** Whether any prediction row genuinely covers `seconds`. */
export function isWithinCoverage(
  seconds: number,
  timeline: Float32Array | number[],
  durations: Float32Array | number[],
): boolean {
  return coveringPredictionIndex(seconds, timeline, durations) >= 0;
}

/** Playback bounds of the prediction timeline, in seconds. */
export function timelineBounds(
  timeline: Float32Array | number[],
  trSeconds: number,
): { start: number; end: number } {
  if (timeline.length === 0) return { start: 0, end: 0 };
  return {
    start: timeline[0] as number,
    end: (timeline[timeline.length - 1] as number) + trSeconds,
  };
}

/** `13.24` -> `"00:13.2"`. */
export function formatTimecode(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}
