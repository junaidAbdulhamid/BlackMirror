"use client";

/**
 * One cortical hemisphere as a Three.js mesh.
 *
 * WHAT IT DOES
 * ------------
 * Builds a BufferGeometry once from the vertex/face arrays, then repaints its
 * colour attribute whenever the timestep or scale changes.
 *
 * WHY IT IS BUILT THIS WAY
 * ------------------------
 * A BufferGeometry is a flat block of GPU memory: positions as
 * [x,y,z, x,y,z, ...], triangles as index triples into that list. Uploading it
 * once and mutating only the colour buffer means a timestep change costs one
 * ~123 KB memcpy, not a mesh rebuild.
 *
 * React never sees the 61,452 colour floats. If it did, every animation frame
 * would reconcile them through the virtual DOM. Instead they live in a ref, are
 * written by a plain loop, and are flagged to Three.js with `needsUpdate`.
 *
 * IN:  geometry buffers + a response frame + a normalizer
 * OUT: a rendered, hoverable cortical surface
 */

import { useEffect, useMemo, useRef } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import type { HemisphereGeometry } from "@/types/neural";
import { fillVertexColors } from "@/lib/colorMapping";

export interface HemisphereProps {
  hemisphere: HemisphereGeometry;
  predictions: Float32Array;
  /** Row offset into `predictions` for the current timestep (index * vertexCount). */
  frameOffset: number;
  normalize: (value: number) => number;
  lut: Float32Array;
  medialWall: Uint8Array | null;
  /** Lateral separation. Inflated hemispheres are both centred on x=0. */
  offsetX: number;
  opacity: number;
  visible: boolean;
  /** False when no prediction row covers the current moment. */
  covered: boolean;
  onInvalidCount?: (count: number) => void;
  onHover?: (localIndex: number | null) => void;
  onSelect?: (localIndex: number) => void;
  onBufferUpdate?: (milliseconds: number) => void;
}

export function Hemisphere({
  hemisphere,
  predictions,
  frameOffset,
  normalize,
  lut,
  medialWall,
  offsetX,
  opacity,
  visible,
  covered,
  onInvalidCount,
  onHover,
  onSelect,
  onBufferUpdate,
}: HemisphereProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Built once per hemisphere/surface. Rebuilding per frame would be the single
  // biggest performance mistake available here.
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(hemisphere.positions, 3));
    geo.setIndex(new THREE.BufferAttribute(hemisphere.indices, 1));
    geo.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(hemisphere.vertexCount * 3), 3),
    );
    // Smooth shading needs per-vertex normals; without them the cortex reads as
    // a faceted polygon soup and folds become impossible to see.
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
    return geo;
  }, [hemisphere]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  // The hot path: repaint colours in place, then tell the GPU to re-upload.
  useEffect(() => {
    const attribute = geometry.getAttribute("color") as THREE.BufferAttribute;
    const target = attribute.array as Float32Array;
    const started = performance.now();

    const { invalidCount } = fillVertexColors(
      target,
      predictions,
      frameOffset,
      hemisphere.predictionOffset,
      hemisphere.vertexCount,
      normalize,
      lut,
      medialWall,
      { covered },
    );

    attribute.needsUpdate = true;
    onBufferUpdate?.(performance.now() - started);
    onInvalidCount?.(invalidCount);
  }, [
    geometry,
    predictions,
    frameOffset,
    hemisphere.predictionOffset,
    hemisphere.vertexCount,
    normalize,
    lut,
    medialWall,
    covered,
    onBufferUpdate,
    onInvalidCount,
  ]);

  const handleMove = (event: ThreeEvent<PointerEvent>) => {
    if (!onHover) return;
    event.stopPropagation();
    // `face.a` is a vertex index of the hit triangle — close enough for
    // inspection and far cheaper than nearest-vertex search.
    const index = event.face?.a;
    onHover(typeof index === "number" ? index : null);
  };

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      position={[offsetX, 0, 0]}
      visible={visible}
      onPointerMove={handleMove}
      onPointerOut={() => onHover?.(null)}
      onClick={(event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        const index = event.face?.a;
        if (typeof index === "number") onSelect?.(index);
      }}
    >
      <meshStandardMaterial
        vertexColors
        transparent={opacity < 1}
        opacity={opacity}
        roughness={0.82}
        metalness={0.04}
        side={THREE.DoubleSide}
        flatShading={false}
      />
    </mesh>
  );
}
