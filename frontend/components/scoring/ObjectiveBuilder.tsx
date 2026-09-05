"use client";

/**
 * Builds a NeuralObjective from explicit choices.
 *
 * Every control corresponds to a field the backend validates. The metric list
 * is fetched from the registry rather than hard-coded, so a metric that cannot
 * state its formula never reaches the UI, and each metric's limitations are
 * shown next to it rather than buried in documentation.
 */

import { useEffect, useMemo, useState } from "react";

import { Label, Panel } from "@/components/scoring/Primitives";
import { fetchMetricCatalog } from "@/lib/scoring";
import type {
  MetricCatalog,
  NeuralObjective,
  ObjectiveDirection,
  TargetType,
  TemporalScopeType,
} from "@/types/scoring";

const TARGETS: { value: TargetType; label: string }[] = [
  { value: "whole_cortex", label: "Whole cortex" },
  { value: "roi", label: "Region (Destrieux)" },
  { value: "network", label: "Network (Yeo 7)" },
  { value: "hemisphere", label: "Hemisphere" },
];

const SCOPES: { value: TemporalScopeType; label: string }[] = [
  { value: "full_stimulus", label: "Full stimulus" },
  { value: "absolute_time_window", label: "Time window" },
  { value: "normalized_time_window", label: "Relative window (%)" },
  { value: "content_event", label: "Content event" },
  { value: "content_event_relative_window", label: "Around content event" },
];

const DIRECTIONS: { value: ObjectiveDirection; label: string }[] = [
  { value: "maximize", label: "Maximize" },
  { value: "minimize", label: "Minimize" },
  { value: "target", label: "Target value" },
];

/** Only the event types Phase 4 actually produces. Hook and Product Reveal
 *  are deliberately absent: this pipeline does not detect them. */
const EVENT_TYPES = [
  "cta",
  "speech_segment",
  "scene",
  "shot",
  "text_overlay",
  "music_segment",
  "silence",
];

export function ObjectiveBuilder({
  onAdd,
  regionNames,
  networkNames,
}: {
  onAdd: (objective: NeuralObjective) => void;
  regionNames: string[];
  networkNames: string[];
}) {
  const [catalog, setCatalog] = useState<MetricCatalog>({});
  const [metric, setMetric] = useState("MEAN_RESPONSE");
  const [targetType, setTargetType] = useState<TargetType>("roi");
  const [regionId, setRegionId] = useState(0);
  const [networkId, setNetworkId] = useState(0);
  const [hemisphere, setHemisphere] = useState<"left" | "right">("left");
  const [scope, setScope] = useState<TemporalScopeType>("full_stimulus");
  const [startSeconds, setStartSeconds] = useState(0);
  const [endSeconds, setEndSeconds] = useState(5);
  const [eventType, setEventType] = useState("speech_segment");
  const [skipEmpty, setSkipEmpty] = useState(true);
  const [direction, setDirection] = useState<ObjectiveDirection>("maximize");
  const [targetValue, setTargetValue] = useState(0.5);
  const [tolerance, setTolerance] = useState(1);
  const [weight, setWeight] = useState(1);
  const [name, setName] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetchMetricCatalog(controller.signal)
      .then(setCatalog)
      .catch(() => setCatalog({}));
    return () => controller.abort();
  }, []);

  const metricNames = useMemo(() => Object.keys(catalog).sort(), [catalog]);
  const describe = catalog[metric];

  const targetLabel =
    targetType === "roi"
      ? (regionNames[regionId] ?? `region ${regionId}`)
      : targetType === "network"
        ? (networkNames[networkId] ?? `network ${networkId}`)
        : targetType === "hemisphere"
          ? `${hemisphere} hemisphere`
          : "whole cortex";

  function build(): NeuralObjective {
    const id = `${metric.toLowerCase()}-${targetType}-${Date.now().toString(36)}`;
    return {
      objective_id: id,
      name: name.trim() || `${metric} · ${targetLabel}`,
      metric,
      target: {
        type: targetType,
        region_id: targetType === "roi" ? regionId : null,
        network_id: targetType === "network" ? networkId : null,
        hemisphere: targetType === "hemisphere" ? hemisphere : null,
      },
      temporal_scope: {
        type: scope,
        start_seconds: scope === "absolute_time_window" ? startSeconds : null,
        end_seconds: scope === "absolute_time_window" ? endSeconds : null,
        start_fraction: scope === "normalized_time_window" ? startSeconds / 100 : null,
        end_fraction: scope === "normalized_time_window" ? endSeconds / 100 : null,
        event_type:
          scope === "content_event" || scope === "content_event_relative_window"
            ? eventType
            : null,
        event_selection: "first",
        offset_start_seconds: scope === "content_event_relative_window" ? startSeconds : null,
        offset_end_seconds: scope === "content_event_relative_window" ? endSeconds : null,
        skip_events_without_samples: skipEmpty,
      },
      direction,
      normalization:
        direction === "target"
          ? {
              strategy: "target_distance",
              target_value: targetValue,
              target_tolerance: tolerance,
            }
          : { strategy: "none" },
      weight,
    };
  }

  const field = "tv-input w-full px-2.5 py-2";

  return (
    <Panel title="Create objective">
      <div className="grid gap-5 md:grid-cols-2">
        <div className="space-y-1.5">
          <Label>Metric</Label>
          <select className={field} value={metric} onChange={(e) => setMetric(e.target.value)}>
            {metricNames.map((value) => (
              <option key={value} value={value}>
                {value.replaceAll("_", " ").toLowerCase()}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5">
          <Label>Target</Label>
          <select
            className={field}
            value={targetType}
            onChange={(e) => setTargetType(e.target.value as TargetType)}
          >
            {TARGETS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        {targetType === "roi" ? (
          <div className="space-y-1.5 md:col-span-2">
            <Label>Region · {regionNames.length} available</Label>
            <select
              className={field}
              value={regionId}
              onChange={(e) => setRegionId(Number(e.target.value))}
            >
              {regionNames.map((label, index) => (
                <option key={label + index} value={index}>
                  {index} · {label}
                </option>
              ))}
            </select>
          </div>
        ) : null}

        {targetType === "network" ? (
          <div className="space-y-1.5 md:col-span-2">
            <Label>Network · verified Yeo 2011 mapping</Label>
            <select
              className={field}
              value={networkId}
              onChange={(e) => setNetworkId(Number(e.target.value))}
            >
              {networkNames.map((label, index) => (
                <option key={label} value={index}>
                  {label}
                </option>
              ))}
            </select>
          </div>
        ) : null}

        {targetType === "hemisphere" ? (
          <div className="space-y-1.5 md:col-span-2">
            <Label>Hemisphere</Label>
            <select
              className={field}
              value={hemisphere}
              onChange={(e) => setHemisphere(e.target.value as "left" | "right")}
            >
              <option value="left">Left</option>
              <option value="right">Right</option>
            </select>
          </div>
        ) : null}

        <div className="space-y-1.5">
          <Label>Time window</Label>
          <select
            className={field}
            value={scope}
            onChange={(e) => setScope(e.target.value as TemporalScopeType)}
          >
            {SCOPES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5">
          <Label>Direction</Label>
          <select
            className={field}
            value={direction}
            onChange={(e) => setDirection(e.target.value as ObjectiveDirection)}
          >
            {DIRECTIONS.map((d) => (
              <option key={d.value} value={d.value}>
                {d.label}
              </option>
            ))}
          </select>
        </div>

        {scope === "content_event" || scope === "content_event_relative_window" ? (
          <div className="space-y-1.5 md:col-span-2">
            <Label>Event type</Label>
            <select
              className={field}
              value={eventType}
              onChange={(e) => setEventType(e.target.value)}
            >
              {EVENT_TYPES.map((value) => (
                <option key={value} value={value}>
                  {value.replaceAll("_", " ")}
                </option>
              ))}
            </select>
            <label className="mt-2 flex items-center gap-2 text-[11px] text-[var(--muted)]">
              <input
                type="checkbox"
                checked={skipEmpty}
                onChange={(e) => setSkipEmpty(e.target.checked)}
              />
              Skip occurrences shorter than one sample
            </label>
            <p className="text-[10.5px] leading-relaxed text-[var(--muted)]">
              Content events are detected below one second while predictions are sampled at
              the TR, so a short occurrence can contain no sample at all.
            </p>
          </div>
        ) : null}

        {scope === "absolute_time_window" ||
        scope === "normalized_time_window" ||
        scope === "content_event_relative_window" ? (
          <>
            <div className="space-y-1.5">
              <Label>
                {scope === "normalized_time_window"
                  ? "Start %"
                  : scope === "content_event_relative_window"
                    ? "Offset start (s)"
                    : "Start (s)"}
              </Label>
              <input
                type="number"
                step="0.1"
                className={field}
                value={startSeconds}
                onChange={(e) => setStartSeconds(Number(e.target.value))}
              />
            </div>
            <div className="space-y-1.5">
              <Label>
                {scope === "normalized_time_window"
                  ? "End %"
                  : scope === "content_event_relative_window"
                    ? "Offset end (s)"
                    : "End (s)"}
              </Label>
              <input
                type="number"
                step="0.1"
                className={field}
                value={endSeconds}
                onChange={(e) => setEndSeconds(Number(e.target.value))}
              />
            </div>
          </>
        ) : null}

        {direction === "target" ? (
          <>
            <div className="space-y-1.5">
              <Label>Target value</Label>
              <input
                type="number"
                step="0.01"
                className={field}
                value={targetValue}
                onChange={(e) => setTargetValue(Number(e.target.value))}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Tolerance</Label>
              <input
                type="number"
                step="0.01"
                min="0.0001"
                className={field}
                value={tolerance}
                onChange={(e) => setTolerance(Number(e.target.value))}
              />
            </div>
          </>
        ) : null}

        <div className="space-y-1.5">
          <Label>Weight</Label>
          <input
            type="number"
            step="0.05"
            min="0"
            className={field}
            value={weight}
            onChange={(e) => setWeight(Number(e.target.value))}
          />
        </div>

        <div className="space-y-1.5">
          <Label>Name (optional)</Label>
          <input
            className={field}
            value={name}
            placeholder={`${metric} · ${targetLabel}`}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
      </div>

      {describe ? (
        <div className="mt-5 space-y-2 border-t border-[var(--line)] pt-4">
          <Label>Formula</Label>
          <code className="block font-mono text-[11px] text-[var(--muted-strong)]">
            {describe.formula}
          </code>
          <Label className="pt-1">Limitations</Label>
          <p className="text-[11px] leading-relaxed text-[var(--muted)]">{describe.limitations}</p>
        </div>
      ) : null}

      <button className="tv-btn mt-5 w-full px-4 py-2.5" onClick={() => onAdd(build())}>
        Add objective
      </button>
    </Panel>
  );
}
