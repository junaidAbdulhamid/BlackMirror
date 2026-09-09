import type { RegionDifferenceSummary } from "@/types/neural";

/** Rank by magnitude while preserving signed candidate-minus-reference values. */
export function rankRegionDifferences(
  regions: RegionDifferenceSummary[],
  limit: number,
): RegionDifferenceSummary[] {
  if (!Number.isInteger(limit) || limit < 0) throw new Error("limit must be a nonnegative integer");
  return [...regions]
    .filter(
      (region) =>
        Number.isFinite(region.mean_absolute_delta) && Number.isFinite(region.mean_signed_delta),
    )
    .sort(
      (first, second) =>
        second.mean_absolute_delta - first.mean_absolute_delta || first.region_id - second.region_id,
    )
    .slice(0, limit);
}

/** Plot observed samples at their actual timestamps so gaps remain visible. */
export function sparklinePoints(times: Float64Array, values: Float64Array): string {
  if (times.length !== values.length) throw new Error("sparkline times and values must align");
  const pairs = Array.from(times, (time, index) => [time, values[index]!] as const).filter(
    ([time, value]) => Number.isFinite(time) && Number.isFinite(value),
  );
  if (!pairs.length) return "";
  // Derived by scanning rather than by assuming the first and last entries are
  // the extremes: a caller passing unsorted timestamps would otherwise place
  // points outside the viewBox instead of failing visibly. Scanning also avoids
  // Math.max(...array), which throws on a long enough run.
  let minTime = pairs[0]![0];
  let maxTime = pairs[0]![0];
  let maxValue = 0;
  for (const [time, value] of pairs) {
    if (time < minTime) minTime = time;
    if (time > maxTime) maxTime = time;
    if (value > maxValue) maxValue = value;
  }
  const scale = maxValue || 1;
  return pairs
    .map(([time, value]) => {
      const x = maxTime === minTime ? 0 : ((time - minTime) / (maxTime - minTime)) * 800;
      // Clamp so a negative value cannot be drawn outside the plot area.
      const y = 100 - (Math.max(value, 0) / scale) * 90;
      return `${x},${y}`;
    })
    .join(" ");
}
