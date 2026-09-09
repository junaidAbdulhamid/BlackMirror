/**
 * Deterministic performance baseline for the full 20,484-vertex hot path.
 * Synthetic numeric buffers isolate rendering cost and require no live API.
 */

import { describe, expect, it } from "vitest";
import { buildColorLut, fillVertexColors } from "@/lib/colorMapping";

describe("render hot path", () => {
  it("colours a full hemisphere fast enough for 60 FPS", () => {
    const V = 20_484;
    const T = 12;
    const perHemi = 10_242;
    const absMax = 2;
    const predictions = new Float32Array(T * V);
    const medialWall = new Uint8Array(V);
    for (let i = 0; i < predictions.length; i += 1) {
      predictions[i] = Math.sin(i * 0.001) * absMax;
    }
    for (let i = 0; i < V; i += 97) medialWall[i] = 1;
    const normalize = (v: number) => Math.min(Math.max((v + absMax) / (2 * absMax), 0), 1);
    const lut = buildColorLut("diverging", 256);
    const target = new Float32Array(perHemi * 3);

    for (let i = 0; i < 100; i += 1) {
      fillVertexColors(target, predictions, 0, 0, perHemi, normalize, lut, medialWall);
    }

    const samples: number[] = [];
    for (let i = 0; i < 3000; i += 1) {
      const frameOffset = (i % T) * V;
      const hemiOffset = (i % 2) * perHemi;
      const t0 = performance.now();
      fillVertexColors(
        target, predictions, frameOffset, hemiOffset, perHemi, normalize, lut, medialWall,
      );
      samples.push(performance.now() - t0);
    }
    samples.sort((a, b) => a - b);
    const at = (p: number) => samples[Math.floor(samples.length * p)] as number;
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const bothHemispheres = mean * 2;

    console.log(
      JSON.stringify(
        {
          vertices_total: V,
          vertices_per_hemisphere: perHemi,
          timesteps: T,
          buffer_fill_ms: {
            mean: +mean.toFixed(4),
            p50: +at(0.5).toFixed(4),
            p95: +at(0.95).toFixed(4),
            p99: +at(0.99).toFixed(4),
          },
          both_hemispheres_ms: +bothHemispheres.toFixed(4),
          colouring_headroom_fps: Math.round(1000 / bothHemispheres),
          prediction_bytes: predictions.byteLength,
          colour_buffer_bytes_per_hemisphere: target.byteLength,
        },
        null,
        2,
      ),
    );

    // A frame budget at 60 FPS is 16.7 ms; colouring both hemispheres must be a
    // small fraction of it, leaving the rest for the actual GPU draw.
    expect(bothHemispheres).toBeLessThan(5);
  }, 60000);
});
