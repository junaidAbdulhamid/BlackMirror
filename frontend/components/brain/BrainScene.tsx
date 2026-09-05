"use client";

/**
 * The 3D scene: camera, lighting, controls, and the cortical surface.
 *
 * React Three Fiber renders a React tree into a Three.js scene graph. `<mesh>`
 * is not a DOM node — it is a declarative description of a Three.js object.
 * React reconciles the *structure*; the WebGL draw loop runs outside React
 * entirely, which is what keeps 60 FPS achievable while React handles the UI.
 *
 * Lighting is deliberately neutral (key + fill + rim, no coloured lights) so
 * that every colour a viewer sees comes from the data, not from the scene.
 */

import { Suspense, useCallback, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import type { HemisphereGeometry, HemisphereView } from "@/types/neural";
import { CorticalMesh } from "@/components/brain/CorticalMesh";

export type CameraPreset = "lateral" | "medial" | "anterior" | "posterior" | "superior";

/**
 * Camera positions in millimetres, matched to fsaverage5's scale.
 *
 * Two sets, because the geometry differs. When both hemispheres are shown they
 * are separated along **X**, so a camera on the X axis puts one hemisphere
 * directly behind the other and the far one is entirely occluded. The paired
 * views therefore use off-axis positions that keep both in frame.
 */
const SINGLE_PRESETS: Record<CameraPreset, [number, number, number]> = {
  lateral: [-330, 0, 0],
  medial: [330, 0, 0],
  anterior: [0, 330, 0],
  posterior: [0, -330, 0],
  superior: [0, 0, 330],
};

const PAIRED_PRESETS: Record<CameraPreset, [number, number, number]> = {
  // Oblique three-quarter views: both hemispheres stay visible.
  lateral: [-250, -300, 200],
  medial: [250, 300, 200],
  anterior: [0, 360, 90],
  posterior: [0, -360, 90],
  // Dorsal view — the natural way to see both hemispheres at once.
  superior: [0, 0, 420],
};

export function cameraPosition(
  preset: CameraPreset,
  paired: boolean,
): [number, number, number] {
  return paired ? PAIRED_PRESETS[preset] : SINGLE_PRESETS[preset];
}

/** The view that best shows the current geometry, used as the default. */
export function defaultPresetFor(paired: boolean): CameraPreset {
  return paired ? "superior" : "lateral";
}

function FrameCounter({ onFps }: { onFps?: (fps: number) => void }) {
  const frames = useRef(0);
  const since = useRef(performance.now());
  useFrame(() => {
    if (!onFps) return;
    frames.current += 1;
    const now = performance.now();
    if (now - since.current >= 500) {
      onFps((frames.current * 1000) / (now - since.current));
      frames.current = 0;
      since.current = now;
    }
  });
  return null;
}

function CameraRig({
  preset,
  paired,
  controlsRef,
}: {
  preset: CameraPreset;
  paired: boolean;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
}) {
  const { camera } = useThree();
  useEffect(() => {
    const [x, y, z] = cameraPosition(preset, paired);
    camera.position.set(x, y, z);
    camera.up.set(0, 0, 1); // Neuroanatomical convention: +Z is superior.
    camera.lookAt(0, 0, 0);
    controlsRef.current?.target.set(0, 0, 0);
    controlsRef.current?.update();
  }, [preset, paired, camera, controlsRef]);
  return null;
}

export interface BrainSceneProps {
  hemispheres: HemisphereGeometry[];
  values: Float32Array;
  frameOffset: number;
  normalize: (value: number) => number;
  lut: Float32Array;
  medialWall: Uint8Array | null;
  view: HemisphereView;
  separateHemispheres: boolean;
  /** False when no prediction row covers the current moment. */
  covered: boolean;
  cameraPreset: CameraPreset;
  /** Bumped by the parent to force a camera reset. */
  resetSignal: number;
  opacity?: number;
  onHover?: (hemisphere: string, localIndex: number | null) => void;
  onSelect?: (hemisphere: string, localIndex: number) => void;
  onFps?: (fps: number) => void;
  onBufferUpdate?: (milliseconds: number) => void;
  onInvalidCount?: (hemisphere: string, count: number) => void;
  onCanvasReady?: (canvas: HTMLCanvasElement) => void;
}

export function BrainScene({
  hemispheres,
  values,
  frameOffset,
  normalize,
  lut,
  medialWall,
  view,
  separateHemispheres,
  covered,
  cameraPreset,
  resetSignal,
  opacity = 1,
  onHover,
  onSelect,
  onFps,
  onBufferUpdate,
  onInvalidCount,
  onCanvasReady,
}: BrainSceneProps) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);

  const handleCreated = useCallback(
    ({ gl }: { gl: THREE.WebGLRenderer }) => {
      // preserveDrawingBuffer is required for snapshot export.
      onCanvasReady?.(gl.domElement);
    },
    [onCanvasReady],
  );

  // Remounting the rig is what re-applies the camera on preset/reset changes.
  const paired = view === "both" && separateHemispheres;
  const key = useMemo(
    () => `${cameraPreset}-${paired}-${resetSignal}`,
    [cameraPreset, paired, resetSignal],
  );

  return (
    <Canvas
      camera={{
        position: cameraPosition(cameraPreset, paired),
        fov: 35,
        near: 1,
        far: 4000,
        up: [0, 0, 1],
      }}
      gl={{ antialias: true, preserveDrawingBuffer: true }}
      dpr={[1, 2]}
      onCreated={handleCreated}
      style={{ background: "transparent" }}
    >
      <color attach="background" args={["#0a0c10"]} />
      <hemisphereLight intensity={0.45} groundColor="#141821" color="#dfe6f2" />
      <directionalLight position={[200, 260, 320]} intensity={1.15} />
      <directionalLight position={[-260, -180, -200]} intensity={0.4} />
      <ambientLight intensity={0.25} />

      <Suspense fallback={null}>
        <CorticalMesh
          hemispheres={hemispheres}
          values={values}
          frameOffset={frameOffset}
          normalize={normalize}
          lut={lut}
          medialWall={medialWall}
          view={view}
          separateHemispheres={separateHemispheres}
          covered={covered}
          opacity={opacity}
          onHover={onHover}
          onSelect={onSelect}
          onBufferUpdate={onBufferUpdate}
          onInvalidCount={onInvalidCount}
        />
      </Suspense>

      <CameraRig key={key} preset={cameraPreset} paired={paired} controlsRef={controlsRef} />
      <OrbitControls
        ref={controlsRef}
        enablePan
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.55}
        panSpeed={0.6}
        zoomSpeed={0.7}
        // Keeps the cortex on screen: too close clips inside the surface, too
        // far loses it entirely.
        minDistance={120}
        maxDistance={900}
        makeDefault
      />
      <FrameCounter onFps={onFps} />
    </Canvas>
  );
}
