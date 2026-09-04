"use client";

import { useCallback, useRef, useState } from "react";

/**
 * Shake-then-burst reaction for the sidebar logo.
 *
 * canvas-confetti is imported dynamically rather than at module scope: it
 * touches `document` on load, and a static import would run during Next.js
 * server rendering and fail the build. Loading it inside the handler also
 * keeps it out of the initial bundle, since almost nobody triggers this.
 *
 * The burst is fired from the logo's own screen position rather than the
 * page centre, so it reads as coming *out of* the logo instead of being an
 * unrelated full-page effect.
 */

const SHAKE_MS = 700;

export function useLogoCelebration() {
  const [shaking, setShaking] = useState(false);
  const anchorRef = useRef<HTMLDivElement | null>(null);

  const celebrate = useCallback(async () => {
    setShaking(true);
    window.setTimeout(() => setShaking(false), SHAKE_MS);

    const confetti = (await import("canvas-confetti")).default;

    // Origin in viewport-relative 0..1 coordinates, centred on the logo.
    const rect = anchorRef.current?.getBoundingClientRect();
    const origin = rect
      ? {
          x: (rect.left + rect.width / 2) / window.innerWidth,
          y: (rect.top + rect.height / 2) / window.innerHeight,
        }
      : { x: 0.15, y: 0.1 };

    // Fire after the shake so the burst reads as the release, not the cause.
    window.setTimeout(() => {
      // Main burst — wide spread, biased upward and to the right, away from
      // the screen edge the sidebar sits against.
      confetti({
        particleCount: 90,
        spread: 80,
        startVelocity: 42,
        angle: 65,
        origin,
        colors: ["#F2B33D", "#3DDC97", "#5B9DF5", "#E5484D", "#FFFFFF"],
        scalar: 0.9,
        ticks: 220,
        disableForReducedMotion: true,
      });

      // A second, narrower burst a beat later gives the explosion some depth
      // rather than a single flat pop.
      window.setTimeout(
        () =>
          confetti({
            particleCount: 45,
            spread: 45,
            startVelocity: 30,
            angle: 80,
            origin,
            colors: ["#F2B33D", "#FFFFFF"],
            scalar: 0.7,
            ticks: 180,
            disableForReducedMotion: true,
          }),
        140,
      );
    }, SHAKE_MS - 180);
  }, []);

  return { celebrate, shaking, anchorRef };
}
