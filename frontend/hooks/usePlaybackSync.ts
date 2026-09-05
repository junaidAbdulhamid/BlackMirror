/**
 * Drives `currentTime` during playback and keeps the video element in step.
 *
 * TIMING SOURCE
 * -------------
 * When a playable video exists it is the clock: `timeupdate`/rAF reads the
 * element's own `currentTime`. Browsers decode video on their own schedule, so
 * deriving time from anything else guarantees drift.
 *
 * When there is no video (audio-only or text stimuli) a requestAnimationFrame
 * loop advances time using real elapsed milliseconds — never a fixed increment
 * per frame, which would run at different speeds on different displays.
 */

import { useEffect, useRef } from "react";
import { useNeuralStore } from "@/stores/neuralVisualizationStore";

export function usePlaybackSync(videoRef: React.RefObject<HTMLVideoElement | null>): void {
  const isPlaying = useNeuralStore((s) => s.isPlaying);
  const playbackRate = useNeuralStore((s) => s.playbackRate);
  const duration = useNeuralStore((s) => s.duration);
  const setCurrentTime = useNeuralStore((s) => s.setCurrentTime);
  const setPlaying = useNeuralStore((s) => s.setPlaying);

  const frameRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  // The video element follows store state rather than the reverse, so the
  // scrubber, keyboard shortcuts and the play button all take one path.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = playbackRate;
    if (isPlaying && video.paused) void video.play().catch(() => setPlaying(false));
    if (!isPlaying && !video.paused) video.pause();
  }, [isPlaying, playbackRate, videoRef, setPlaying]);

  useEffect(() => {
    if (!isPlaying) {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
      return;
    }

    lastTickRef.current = performance.now();

    const tick = (now: number) => {
      const video = videoRef.current;
      if (video && !video.paused && Number.isFinite(video.currentTime)) {
        // The video owns the clock.
        setCurrentTime(video.currentTime);
        if (video.ended) setPlaying(false);
      } else {
        const elapsed = (now - lastTickRef.current) / 1000;
        lastTickRef.current = now;
        const next = useNeuralStore.getState().currentTime + elapsed * playbackRate;
        if (duration > 0 && next >= duration) {
          setCurrentTime(duration);
          setPlaying(false);
          return;
        }
        setCurrentTime(next);
      }
      frameRef.current = requestAnimationFrame(tick);
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    };
  }, [isPlaying, playbackRate, duration, setCurrentTime, setPlaying, videoRef]);
}

/** Seek the video when the user scrubs, without fighting normal playback. */
export function useSeekVideo(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  seconds: number,
): void {
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(seconds)) return;
    // Only correct meaningful divergence; sub-frame nudges cause stutter.
    if (Math.abs(video.currentTime - seconds) > 0.25) {
      video.currentTime = seconds;
    }
  }, [seconds, videoRef]);
}
