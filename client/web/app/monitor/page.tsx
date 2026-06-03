"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Switch } from "@heroui/switch";
import { Select, SelectItem } from "@heroui/select";

import { VideoFeed } from "@/components/video-feed";
import { getSocket } from "@/lib/socket";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

interface Camera {
  id: string;
  name: string;
  location: string;
  status: "online" | "offline" | "error";
  fps: number;
  resolution?: string;
}

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

interface Detection {
  id?: number;
  camera_id: string;
  vehicle_type: string;
  speed?: number;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  timestamp?: string | Date;
}

interface LogEntry {
  key: string;
  camera_id: string;
  camera_name: string;
  vehicle_type: string;
  speed?: number;
  confidence: number;
  time: string;
}

const streamUrl = (cameraId: string) => `${API}/api/cameras/${cameraId}/stream`;

function nowLabel() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function MonitorPage() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [speeds, setSpeeds] = useState<Record<string, number | undefined>>({});
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [showBoxes, setShowBoxes] = useState(false);
  const [viewMode, setViewMode] = useState("grid");
  const [boxes, setBoxes] = useState<Record<string, BoundingBox[]>>({});
  const logRef = useRef<HTMLDivElement>(null);
  const boxTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  // Fetch cameras list
  const fetchCameras = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/cameras`);
      const json = await res.json();

      if (json.success) setCameras(json.data);
    } catch {
      /* silent — WebSocket provides real-time camera status */
    }
  }, []);

  useEffect(() => {
    fetchCameras();
    const id = setInterval(fetchCameras, 15000);

    return () => clearInterval(id);
  }, [fetchCameras]);

  // WebSocket — subscribe to each camera's detection events
  useEffect(() => {
    const socket = getSocket();

    const handleConnect = () => {
      cameras.forEach((cam) => socket.emit("subscribe_camera", cam.id));
    };

    socket.on("connect", handleConnect);

    // Subscribe immediately if already connected
    if (socket.connected) {
      cameras.forEach((cam) => socket.emit("subscribe_camera", cam.id));
    }

    socket.on("detection", (event: { type: string; data: Detection }) => {
      const det = event.data;
      const cam = cameras.find((c) => c.id === det.camera_id);

      // Update per-camera counters and latest speed
      setCounts((prev) => ({
        ...prev,
        [det.camera_id]: (prev[det.camera_id] ?? 0) + 1,
      }));
      if (det.speed != null) {
        setSpeeds((prev) => ({ ...prev, [det.camera_id]: det.speed }));
      }

      // Convert pixel bbox → percentage for overlay (camera frames are 640×480)
      if (det.bbox_width > 0 && det.bbox_height > 0) {
        const [resW, resH] = (cam?.resolution ?? "640x480")
          .split("x")
          .map(Number);
        const box: BoundingBox = {
          id: `${det.camera_id}-${Date.now()}`,
          x: (det.bbox_x / resW) * 100,
          y: (det.bbox_y / resH) * 100,
          width: (det.bbox_width / resW) * 100,
          height: (det.bbox_height / resH) * 100,
          label: det.vehicle_type,
          confidence: det.confidence,
          speed: det.speed ?? undefined,
        };
        setBoxes((prev) => ({ ...prev, [det.camera_id]: [box] }));

        // Auto-clear box after 3 s so stale boxes don't linger
        clearTimeout(boxTimers.current[det.camera_id]);
        boxTimers.current[det.camera_id] = setTimeout(() => {
          setBoxes((prev) => ({ ...prev, [det.camera_id]: [] }));
        }, 3000);
      }

      // Prepend to detection log (max 50 entries)
      const entry: LogEntry = {
        key: `${det.camera_id}-${Date.now()}`,
        camera_id: det.camera_id,
        camera_name: cam ? `${cam.name} — ${cam.location}` : det.camera_id,
        vehicle_type: det.vehicle_type,
        speed: det.speed ?? undefined,
        confidence: det.confidence,
        time: nowLabel(),
      };

      setLog((prev) => [entry, ...prev].slice(0, 50));
    });

    socket.on(
      "camera_status",
      (event: { data: { camera_id: string; status: Camera["status"] } }) => {
        setCameras((prev) =>
          prev.map((c) =>
            c.id === event.data.camera_id
              ? { ...c, status: event.data.status }
              : c,
          ),
        );
      },
    );

    return () => {
      cameras.forEach((cam) => socket.emit("unsubscribe_camera", cam.id));
      socket.off("connect", handleConnect);
      socket.off("detection");
      socket.off("camera_status");
      Object.values(boxTimers.current).forEach(clearTimeout);
    };
  }, [cameras]);

  // Scroll log to top on new entries
  useEffect(() => {
    logRef.current?.scrollTo({ top: 0 });
  }, [log.length]);

  const totalDetections = Object.values(counts).reduce((a, b) => a + b, 0);
  const avgSpeed = Object.values(speeds).filter((s): s is number => s != null);
  const avgSpeedVal = avgSpeed.length
    ? Math.round(avgSpeed.reduce((a, b) => a + b, 0) / avgSpeed.length)
    : null;

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Live Monitoring</h1>
        <p className="text-white/70">
          Real-time MJPEG camera feeds with AI vehicle detection
        </p>
      </div>

      {/* Controls */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 mb-6 shadow-xl">
        <CardBody className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <Switch
                classNames={{
                  wrapper: "group-data-[selected=true]:bg-white/30",
                }}
                isSelected={showBoxes}
                onValueChange={setShowBoxes}
              >
                <span className="text-white/80">Bounding Boxes</span>
              </Switch>
              <Select
                className="max-w-xs"
                classNames={{
                  label: "text-white/80",
                  trigger: "bg-white/10 border-white/20 text-white",
                }}
                defaultSelectedKeys={["grid"]}
                label="View"
                onChange={(e) => setViewMode(e.target.value)}
              >
                <SelectItem key="grid">Grid (2-up)</SelectItem>
                <SelectItem key="single">Single</SelectItem>
              </Select>
            </div>
            <div className="flex items-center gap-6 text-sm text-white/70">
              <span>
                Session detections:{" "}
                <span className="text-white font-bold">{totalDetections}</span>
              </span>
              {avgSpeedVal !== null && (
                <span>
                  Avg speed:{" "}
                  <span className="text-white font-bold">
                    {avgSpeedVal} km/h
                  </span>
                </span>
              )}
            </div>
          </div>
        </CardBody>
      </Card>

      {/* Camera Feeds */}
      {cameras.length === 0 ? (
        <div className="text-white/50 text-center py-16">Loading cameras…</div>
      ) : (
        <div
          className={`grid gap-6 mb-6 ${viewMode === "grid" ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1 max-w-3xl mx-auto"}`}
        >
          {cameras.map((cam) => (
            <VideoFeed
              key={cam.id}
              boundingBoxes={boxes[cam.id] ?? []}
              cameraId={cam.id}
              cameraName={`${cam.name} — ${cam.location}`}
              fps={cam.fps ?? 30}
              isLive={cam.status === "online"}
              latency={0}
              showBoundingBoxes={showBoxes}
              videoUrl={cam.status === "online" ? streamUrl(cam.id) : undefined}
            />
          ))}
        </div>
      )}

      {/* Live Detection Log */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
        <CardBody className="p-4">
          <h3 className="text-xl font-bold text-white mb-4">
            Live Detection Log
            <span className="ml-3 text-sm font-normal text-white/50">
              (real-time via WebSocket)
            </span>
          </h3>
          <div ref={logRef} className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {log.length === 0 ? (
              <p className="text-white/40 text-center py-8">
                Waiting for detections — camera_sender must be running
              </p>
            ) : (
              log.map((entry) => (
                <div
                  key={entry.key}
                  className="flex justify-between items-center p-3 bg-white/10 rounded-lg border border-white/10"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse flex-shrink-0" />
                    <span className="text-white/60 text-sm truncate max-w-[140px]">
                      {entry.camera_name}
                    </span>
                    <span className="text-white font-semibold capitalize">
                      {entry.vehicle_type}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm flex-shrink-0">
                    {entry.speed != null && (
                      <span
                        className={`font-bold ${entry.speed > 60 ? "text-red-400" : "text-white"}`}
                      >
                        {entry.speed} km/h
                      </span>
                    )}
                    <span className="text-white/60">
                      {(entry.confidence * 100).toFixed(0)}%
                    </span>
                    <span className="text-white/40 text-xs">{entry.time}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
