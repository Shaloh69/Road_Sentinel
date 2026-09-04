"use client";

import { useEffect, useRef, useState } from "react";

import { getSocket } from "@/lib/socket";

export interface DetectionBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
  speed?: number;
  expiresAt: number;
}

interface CameraLike {
  id: string;
  resolution?: string | null;
}

interface DetectionEvent {
  camera_id: string;
  vehicle_type: string;
  confidence: number;
  speed?: number | null;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
}

/**
 * Subscribes to detection events and maintains per-camera overlay boxes.
 *
 * Shared by the Dashboard and Live Monitor so the two cannot drift — the
 * dashboard previously passed a hardcoded empty array and so never drew a box
 * at all, while the monitor had a working copy of this logic.
 *
 * Two behaviours worth knowing:
 *
 *   * Boxes ACCUMULATE. Detections arrive one socket event per vehicle, so
 *     replacing the array (as the monitor used to) meant a frame with three
 *     vehicles rendered a single box that flickered between them. Boxes are
 *     now keyed by position so several can be on screen together.
 *
 *   * Each box carries its own expiry and is pruned by one shared interval
 *     rather than a timer per camera. A per-camera timer could only ever clear
 *     the whole set at once, which is why accumulating boxes needed this to
 *     change too.
 */

const BOX_TTL_MS = 3000;
const PRUNE_INTERVAL_MS = 500;
const MAX_BOXES_PER_CAMERA = 12;

export function useDetectionBoxes(cameras: CameraLike[]) {
  const [boxes, setBoxes] = useState<Record<string, DetectionBox[]>>({});
  const camerasRef = useRef<CameraLike[]>(cameras);

  // Kept in a ref so the socket effect does not resubscribe on every camera
  // refresh — the list is re-fetched on a timer and is usually identical.
  useEffect(() => {
    camerasRef.current = cameras;
  }, [cameras]);

  useEffect(() => {
    const socket = getSocket();

    const subscribe = () => {
      camerasRef.current.forEach((cam) =>
        socket.emit("subscribe_camera", cam.id),
      );
    };

    socket.on("connect", subscribe);
    if (socket.connected) subscribe();

    const onDetection = (event: { type: string; data: DetectionEvent }) => {
      const det = event.data;

      if (!det || det.bbox_width <= 0 || det.bbox_height <= 0) return;

      // Scale from frame pixels to percentages using the camera's own declared
      // resolution, so a camera swap does not silently misplace every box.
      const cam = camerasRef.current.find((c) => c.id === det.camera_id);
      const [resW, resH] = (cam?.resolution ?? "640x480")
        .split("x")
        .map(Number);

      if (!resW || !resH) return;

      const box: DetectionBox = {
        // Position-derived id: the same vehicle re-detected in nearly the same
        // place replaces its own box instead of stacking duplicates on top of
        // each other, which looked like a thickening border.
        id: `${det.camera_id}-${Math.round(det.bbox_x)}-${Math.round(det.bbox_y)}`,
        x: (det.bbox_x / resW) * 100,
        y: (det.bbox_y / resH) * 100,
        width: (det.bbox_width / resW) * 100,
        height: (det.bbox_height / resH) * 100,
        label: det.vehicle_type,
        confidence: det.confidence,
        speed: det.speed ?? undefined,
        expiresAt: Date.now() + BOX_TTL_MS,
      };

      setBoxes((prev) => {
        const existing = prev[det.camera_id] ?? [];
        const merged = [...existing.filter((b) => b.id !== box.id), box];

        return {
          ...prev,
          [det.camera_id]: merged.slice(-MAX_BOXES_PER_CAMERA),
        };
      });
    };

    socket.on("detection", onDetection);

    // One pruner for every camera. Cheap, and it keeps expiry independent per
    // box rather than clearing a camera's whole set on a single timer.
    const pruner = setInterval(() => {
      const now = Date.now();

      setBoxes((prev) => {
        let changed = false;
        const next: Record<string, DetectionBox[]> = {};

        for (const [camId, list] of Object.entries(prev)) {
          const kept = list.filter((b) => b.expiresAt > now);

          if (kept.length !== list.length) changed = true;
          next[camId] = kept;
        }

        return changed ? next : prev;
      });
    }, PRUNE_INTERVAL_MS);

    return () => {
      socket.off("connect", subscribe);
      socket.off("detection", onDetection);
      clearInterval(pruner);
    };
  }, []);

  return boxes;
}
