"use client";

import { useRef, useCallback } from "react";

/**
 * Click-sequence detector for the sidebar logo.
 *
 * Listens for a specific burst pattern — a run of clicks, a deliberate pause,
 * then a second run — and fires once when it matches.
 *
 * Two details make it reliable rather than fiddly:
 *
 * 1. The logo is a NextLink. Rapid clicks would navigate, unmounting this
 *    component mid-sequence and losing the count. So any click that lands
 *    within the burst window has its default suppressed; a normal, deliberate
 *    single click still navigates as usual.
 *
 * 2. Bursts are separated by a gap, not by counting alone, so an imprecise
 *    sequence fails cleanly and resets instead of half-matching.
 */

const BURST_GAP_MS = 650; // longer than this ends the current burst
const SEQUENCE_TIMEOUT_MS = 6000; // whole pattern must complete inside this
const PATTERN = [6, 7];

export function useLogoSequence(onMatch: () => void) {
  const bursts = useRef<number[]>([]);
  const currentBurst = useRef(0);
  const lastClick = useRef(0);
  const burstTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reset = useCallback(() => {
    bursts.current = [];
    currentBurst.current = 0;
    lastClick.current = 0;
    if (burstTimer.current) clearTimeout(burstTimer.current);
    if (resetTimer.current) clearTimeout(resetTimer.current);
  }, []);

  const closeBurst = useCallback(() => {
    if (currentBurst.current === 0) return;
    bursts.current.push(currentBurst.current);
    currentBurst.current = 0;

    // Bail as soon as the pattern can no longer match, rather than waiting for
    // the full sequence to play out.
    const i = bursts.current.length - 1;

    if (bursts.current[i] !== PATTERN[i]) {
      reset();

      return;
    }
    if (bursts.current.length === PATTERN.length) {
      reset();
      onMatch();
    }
  }, [onMatch, reset]);

  const onClick = useCallback(
    (e: React.MouseEvent) => {
      const now = Date.now();
      const sinceLast = now - lastClick.current;

      // Suppress navigation only for clicks that are part of a burst, so the
      // logo still works as a home link under normal use.
      if (lastClick.current !== 0 && sinceLast < BURST_GAP_MS)
        e.preventDefault();

      lastClick.current = now;
      currentBurst.current += 1;

      if (burstTimer.current) clearTimeout(burstTimer.current);
      burstTimer.current = setTimeout(closeBurst, BURST_GAP_MS);

      if (resetTimer.current) clearTimeout(resetTimer.current);
      resetTimer.current = setTimeout(reset, SEQUENCE_TIMEOUT_MS);
    },
    [closeBurst, reset],
  );

  return onClick;
}
