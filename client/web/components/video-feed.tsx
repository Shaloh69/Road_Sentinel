"use client";

import { Card, CardBody, CardHeader } from "@heroui/card";
import { Chip } from "@heroui/chip";
import { useCallback, useEffect, useRef, useState } from "react";

import { getSocket } from "@/lib/socket";

interface BoundingBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
  speed?: number;
}

interface VideoFeedProps {
  cameraId: string;
  cameraName: string;
  isLive?: boolean;
  showBoundingBoxes?: boolean;
  boundingBoxes?: BoundingBox[];
  videoUrl?: string;
  fps?: number;
  latency?: number;
}

export const VideoFeed = ({
  cameraId,
  cameraName,
  isLive = true,
  showBoundingBoxes = true,
  boundingBoxes = [],
  videoUrl,
  fps = 15,
  latency = 0,
}: VideoFeedProps) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hasFrames, setHasFrames] = useState(false);

  // Real-time FPS and frame-interval measured from frame load events
  const imgRef = useRef<HTMLImageElement>(null);
  const blobUrlRef = useRef<string | null>(null);
  const frameTimestamps = useRef<number[]>([]);
  const lastFrameAt = useRef<number>(0);

  const [liveFps, setLiveFps] = useState<number | null>(null);
  const [frameIntervalMs, setFrameIntervalMs] = useState<number | null>(null);
  const [lastFrameAge, setLastFrameAge] = useState<number>(0);

  // Measure FPS and inter-frame interval on each frame display
  const handleLoad = useCallback(() => {
    const now = performance.now();

    if (lastFrameAt.current > 0) {
      setFrameIntervalMs(Math.round(now - lastFrameAt.current));
    }

    lastFrameAt.current = now;
    setLastFrameAge(0);

    frameTimestamps.current = [
      ...frameTimestamps.current.filter((t) => now - t < 3000),
      now,
    ];

    const count = frameTimestamps.current.length;

    if (count >= 2) {
      const span = (now - frameTimestamps.current[0]) / 1000;

      setLiveFps(Math.round((count - 1) / span));
    }
  }, []);

  // Update stale-frame age every second
  useEffect(() => {
    const id = setInterval(() => {
      if (lastFrameAt.current > 0) {
        setLastFrameAge(
          Math.round((performance.now() - lastFrameAt.current) / 1000),
        );
      }
    }, 1000);

    return () => clearInterval(id);
  }, []);

  // WebSocket binary frame subscription with MJPEG fallback
  const [useMjpegFallback, setUseMjpegFallback] = useState(false);

  useEffect(() => {
    if (!isLive) return;

    const socket = getSocket();
    const eventName = `camera_frame:${cameraId}`;

    socket.emit("subscribe_stream", cameraId);

    // Fall back to MJPEG if no WebSocket frames arrive within 5s
    const fallbackTimer = setTimeout(() => {
      setUseMjpegFallback(true);
      setHasFrames(true);
    }, 5_000);

    // Ping every 55s to prevent Cloudflare's 100s WebSocket timeout
    const pingId = setInterval(() => socket.emit("ping"), 55_000);

    const handleFrame = (data: ArrayBuffer | Uint8Array) => {
      clearTimeout(fallbackTimer);
      setUseMjpegFallback(false);

      const blob = new Blob([data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);

      if (imgRef.current) {
        imgRef.current.src = url;
      }

      // Revoke previous blob URL to free memory
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
      }

      blobUrlRef.current = url;
      setHasFrames(true);
    };

    socket.on(eventName, handleFrame);

    return () => {
      clearTimeout(fallbackTimer);
      socket.emit("unsubscribe_stream", cameraId);
      socket.off(eventName, handleFrame);
      clearInterval(pingId);

      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [cameraId, isLive]);

  const displayFps = liveFps ?? fps;
  const displayLatency = frameIntervalMs ?? latency;

  return (
    <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
      <CardHeader className="flex justify-between items-center bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div
            className={`w-3 h-3 rounded-full ${isLive ? "bg-green-400 animate-pulse shadow-lg shadow-green-400/50" : "bg-gray-400"}`}
          />
          <div>
            <h3 className="text-lg font-bold text-white">{cameraName}</h3>
            <p className="text-xs text-white/70">ID: {cameraId}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isLive && (
            <Chip
              className="bg-red-500 text-white text-xs font-semibold"
              size="sm"
              variant="solid"
            >
              LIVE
            </Chip>
          )}
          <div className="flex flex-col text-right">
            <span
              className={`text-xs font-mono font-semibold ${
                liveFps === null
                  ? "text-white/50"
                  : liveFps >= 10
                    ? "text-green-400"
                    : liveFps >= 5
                      ? "text-yellow-400"
                      : "text-red-400"
              }`}
            >
              {displayFps} FPS
            </span>
            <span
              className={`text-xs font-mono ${
                frameIntervalMs === null
                  ? "text-white/50"
                  : frameIntervalMs < 200
                    ? "text-green-400"
                    : frameIntervalMs < 500
                      ? "text-yellow-400"
                      : "text-red-400"
              }`}
            >
              {displayLatency}ms
            </span>
            {lastFrameAge > 2 && (
              <span className="text-xs text-red-400 font-mono">
                {lastFrameAge}s ago
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardBody className="p-0">
        <div className="relative aspect-video bg-black overflow-hidden">
          {/* WebSocket binary frames (primary) */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imgRef}
            alt={cameraName}
            className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-200 ${hasFrames && isLive && !useMjpegFallback ? "opacity-100" : "opacity-0"}`}
            decoding="async"
            fetchPriority="high"
            onLoad={handleLoad}
          />

          {/* MJPEG fallback — used when server not yet updated with WS support */}
          {useMjpegFallback && isLive && videoUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              alt={cameraName}
              className="absolute inset-0 w-full h-full object-cover"
              src={videoUrl}
              onLoad={handleLoad}
            />
          )}

          {/* Placeholder — shown while connecting or offline */}
          {(!hasFrames || !isLive) && (
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-black/40 via-[#44174E]/30 to-[#862249]/30 backdrop-blur-sm">
              <div className="text-center">
                <svg
                  className="w-20 h-20 text-white mx-auto mb-4 opacity-50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                  />
                </svg>
                <p className="text-[#E8BCB8] text-sm">
                  {isLive ? "Connecting…" : "No Signal"}
                </p>
              </div>
            </div>
          )}

          {/* Bounding boxes overlay */}
          {showBoundingBoxes &&
            boundingBoxes.map((box) => (
              <div
                key={box.id}
                className="absolute border-2 border-[#ED9E59] rounded"
                style={{
                  left: `${box.x}%`,
                  top: `${box.y}%`,
                  width: `${box.width}%`,
                  height: `${box.height}%`,
                }}
              >
                <div className="absolute -top-6 left-0 bg-[#ED9E59] text-[#1B1931] px-2 py-0.5 rounded text-xs font-bold">
                  {box.label} {(box.confidence * 100).toFixed(0)}%
                  {box.speed && ` • ${box.speed} km/h`}
                </div>
              </div>
            ))}

          {/* Detection count overlay */}
          {showBoundingBoxes && boundingBoxes.length > 0 && (
            <div className="absolute top-4 left-4 bg-[#1B1931] bg-opacity-80 px-3 py-2 rounded-lg border border-[#ED9E59]">
              <p className="text-[#ED9E59] text-sm font-bold">
                {boundingBoxes.length} Vehicle
                {boundingBoxes.length !== 1 ? "s" : ""} Detected
              </p>
            </div>
          )}

          {/* Fullscreen toggle */}
          <button
            className="absolute bottom-4 right-4 bg-[#1B1931] bg-opacity-80 p-2 rounded-lg hover:bg-opacity-100 transition-all border border-[#44174E] hover:border-[#ED9E59]"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            <svg
              className="w-5 h-5 text-[#E8BCB8]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
              />
            </svg>
          </button>
        </div>
      </CardBody>
    </Card>
  );
};
