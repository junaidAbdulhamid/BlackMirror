"use client";

/**
 * Renders a cortical scalar field across both hemispheres.
 *
 * DELIBERATELY GENERIC
 * --------------------
 * This component takes `values` — any per-vertex scalar field — not "a TRIBE
 * prediction". Phase 5 will pass a difference map ΔR = R_B − R_A through the
 * exact same component. Nothing here knows where the numbers came from.
 *
 * IN:  geometry + a values array + a normalizer
 * OUT: two coloured hemispheres, positioned so they do not overlap
 */

import { useMemo } from "react";
import type { HemisphereGeometry, HemisphereView } from "@/types/neural";
import { Hemisphere } from "@/components/brain/Hemisphere";

/**
 * Lateral offset applied to each hemisphere, in millimetres.
 *
 * Inflated fsaverage5 hemispheres are BOTH centred on x = 0, so drawn as-is
 * they occupy the same space. Pial coordinates are already lateralised and need
 * no correction. Verified in tests/unit/test_mesh_alignment.py.
 */
export const INFLATED_SEPARATION_MM = 48;

export interface CorticalMeshProps {
  hemispheres: HemisphereGeometry[];
  values: Float32Array;
  /** Offset of the current frame within `values` (0 for a single frame). */
  frameOffset: number;
  normalize: (value: number) => number;
  lut: Float32Array;
  medialWall: Uint8Array | null;
  view: HemisphereView;
  /** True when the surface needs artificial hemisphere separation. */
  separateHemispheres: boolean;
  /** False when no prediction row covers the current moment. */
  covered: boolean;
  opacity?: number;
  onInvalidCount?: (hemisphere: string, count: number) => void;
  onHover?: (hemisphere: string, localIndex: number | null) => void;
  onSelect?: (hemisphere: string, localIndex: number) => void;
  onBufferUpdate?: (milliseconds: number) => void;
}

export function CorticalMesh({
  hemispheres,
  values,
  frameOffset,
  normalize,
  lut,
  medialWall,
  view,
  separateHemispheres,
  covered,
  opacity = 1,
  onHover,
  onSelect,
  onBufferUpdate,
  onInvalidCount,
}: CorticalMeshProps) {
  const offsets = useMemo(() => {
    const map = new Map<string, number>();
    for (const hemisphere of hemispheres) {
      if (!separateHemispheres) {
        map.set(hemisphere.name, 0);
        continue;
      }
      // When only one hemisphere is shown, centre it instead of leaving it
      // hanging off to one side.
      if (view !== "both") {
        map.set(hemisphere.name, 0);
        continue;
      }
      map.set(
        hemisphere.name,
        hemisphere.name === "left" ? -INFLATED_SEPARATION_MM : INFLATED_SEPARATION_MM,
      );
    }
    return map;
  }, [hemispheres, separateHemispheres, view]);

  return (
    <group>
      {hemispheres.map((hemisphere) => (
        <Hemisphere
          key={hemisphere.name}
          hemisphere={hemisphere}
          predictions={values}
          frameOffset={frameOffset}
          normalize={normalize}
          lut={lut}
          medialWall={medialWall}
          offsetX={offsets.get(hemisphere.name) ?? 0}
          opacity={opacity}
          visible={view === "both" || view === hemisphere.name}
          covered={covered}
          onInvalidCount={(count) => onInvalidCount?.(hemisphere.name, count)}
          onHover={(index) => onHover?.(hemisphere.name, index)}
          onSelect={(index) => onSelect?.(hemisphere.name, index)}
          onBufferUpdate={onBufferUpdate}
        />
      ))}
    </group>
  );
}
