"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Link } from "@heroui/link";
import { Button } from "@heroui/button";

import { StatCard } from "@/components/stat-card";
import { VideoFeed } from "@/components/video-feed";
import { AlertCard } from "@/components/alert-card";
import { CameraStatus } from "@/components/camera-status";
import { getSocket } from "@/lib/socket";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);

  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);

  if (hrs < 24) return `${hrs}h ago`;

  return new Date(iso).toLocaleDateString();
}

function mapType(t: string): "speeding" | "crash" | "anomaly" | "warning" {
  if (t === "speeding") return "speeding";
  if (t === "crash") return "crash";
  if (
    ["wrong_way", "stopped_vehicle", "congestion", "illegal_parking"].includes(
      t,
    )
  )
    return "warning";

  return "anomaly";
}

interface Summary {
  vehicles_today: number;
  average_speed: number | null;
  incidents_today: number;
  cameras_online: number;
  cameras_total: number;
}
interface Camera {
  id: string;
  name: string;
  location: string;
  status: "online" | "offline" | "error";
  fps: number;
  resolution: string;
}
interface Incident {
  id: number;
  camera_id: string;
  camera_name: string;
  incident_type: string;
  severity: "low" | "medium" | "high" | "critical";
  title: string;
  description: string;
  timestamp: string;
  status: string;
  confidence?: number;
}

export default function Home() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const incidentIds = useRef(new Set<number>());

  const fetchAll = useCallback(async () => {
    try {
      const [sumRes, camRes, incRes] = await Promise.all([
        fetch(`${API}/api/analytics/summary`),
        fetch(`${API}/api/cameras`),
        fetch(`${API}/api/incidents?status=active&limit=5`),
      ]);
      const [sumJson, camJson, incJson] = await Promise.all([
        sumRes.json(),
        camRes.json(),
        incRes.json(),
      ]);

      if (sumJson.success) setSummary(sumJson.data);
      if (camJson.success) setCameras(camJson.data);
      if (incJson.success) {
        setIncidents(incJson.data);
        incidentIds.current = new Set(incJson.data.map((i: Incident) => i.id));
      }
      setError(null);
    } catch {
      setError("Cannot reach Node service — check NEXT_PUBLIC_API_URL");
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch + 30s fallback poll
  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 30000);

    return () => clearInterval(id);
  }, [fetchAll]);

  // WebSocket real-time
  useEffect(() => {
    const socket = getSocket();

    socket.on("connect", () => setWsConnected(true));
    socket.on("disconnect", () => setWsConnected(false));

    socket.emit("subscribe_incidents");

    socket.on("incident", (event: { type: string; data: Incident }) => {
      const inc = event.data;

      if (!incidentIds.current.has(inc.id)) {
        incidentIds.current.add(inc.id);
        setIncidents((prev) => [inc, ...prev].slice(0, 5));
        setSummary((prev) =>
          prev ? { ...prev, incidents_today: prev.incidents_today + 1 } : prev,
        );
      }
    });

    socket.on("detection", (_event: { data: { camera_id: string } }) => {
      setSummary((prev) =>
        prev ? { ...prev, vehicles_today: prev.vehicles_today + 1 } : prev,
      );
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
      socket.off("incident");
      socket.off("detection");
      socket.off("camera_status");
    };
  }, []);

  const camA = cameras.find((c) => c.id.includes("A") || c.name.includes("A"));
  const camB = cameras.find((c) => c.id.includes("B") || c.name.includes("B"));

  const streamUrl = (cameraId: string) =>
    cameraId ? `${API}/api/cameras/${cameraId}/stream` : undefined;

  return (
    <div className="min-h-screen p-6">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-4xl font-bold text-white">Dashboard</h1>
          <div
            className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-400 animate-pulse" : "bg-gray-500"}`}
            title={wsConnected ? "Real-time connected" : "Polling mode"}
          />
        </div>
        <p className="text-white/70">
          Real-time traffic monitoring — Barangay Busay, Cebu
        </p>
        {error && (
          <p className="mt-2 text-sm text-red-400 bg-red-900/30 border border-red-500/30 px-3 py-2 rounded-lg">
            ⚠ {error}
          </p>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
              />
            </svg>
          }
          title="Vehicles Today"
          trend={{ value: "live", isPositive: true }}
          value={
            loading ? "—" : (summary?.vehicles_today ?? 0).toLocaleString()
          }
        />
        <StatCard
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M13 10V3L4 14h7v7l9-11h-7z"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
              />
            </svg>
          }
          title="Average Speed"
          trend={{ value: "live", isPositive: true }}
          value={
            loading
              ? "—"
              : summary?.average_speed
                ? `${summary.average_speed} km/h`
                : "N/A"
          }
        />
        <StatCard
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
              />
            </svg>
          }
          subtitle="Last 24 hours"
          title="Incidents Today"
          value={loading ? "—" : (summary?.incidents_today ?? 0)}
        />
        <StatCard
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
              />
            </svg>
          }
          subtitle={
            cameras.every((c) => c.status === "online")
              ? "All operational"
              : "Check cameras"
          }
          title="Active Cameras"
          value={
            loading
              ? "—"
              : `${summary?.cameras_online ?? 0}/${summary?.cameras_total ?? 2}`
          }
        />
      </div>

      {/* Live Feeds */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-white">Live Camera Feeds</h2>
          <Link href="/monitor">
            <Button className="bg-white text-[#1B1931] font-semibold hover:bg-white/90 shadow-lg">
              View All Feeds
            </Button>
          </Link>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <VideoFeed
            boundingBoxes={[]}
            cameraId={camA?.id ?? "CAM-A-001"}
            cameraName={
              camA
                ? `${camA.name} — ${camA.location}`
                : "Camera A — North Approach"
            }
            fps={camA?.fps ?? 30}
            isLive={camA?.status === "online"}
            latency={42}
            showBoundingBoxes={false}
            videoUrl={camA ? streamUrl(camA.id) : undefined}
          />
          <VideoFeed
            boundingBoxes={[]}
            cameraId={camB?.id ?? "CAM-B-002"}
            cameraName={
              camB
                ? `${camB.name} — ${camB.location}`
                : "Camera B — South Approach"
            }
            fps={camB?.fps ?? 30}
            isLive={camB?.status === "online"}
            latency={38}
            showBoundingBoxes={false}
            videoUrl={camB ? streamUrl(camB.id) : undefined}
          />
        </div>
      </div>

      {/* Camera Status & Incidents */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <h2 className="text-2xl font-bold text-white mb-4">Camera Status</h2>
          <div className="space-y-4">
            {cameras.length > 0 ? (
              cameras.map((cam) => (
                <CameraStatus
                  key={cam.id}
                  detectionRate={cam.status === "online" ? 97 : 0}
                  fps={cam.fps}
                  id={cam.id}
                  isOnline={cam.status === "online"}
                  location={cam.location}
                  name={cam.name}
                  resolution={cam.resolution}
                />
              ))
            ) : (
              <>
                <CameraStatus
                  detectionRate={0}
                  fps={0}
                  id="CAM-A-001"
                  isOnline={false}
                  location="North Approach"
                  name="Camera A"
                  resolution="—"
                />
                <CameraStatus
                  detectionRate={0}
                  fps={0}
                  id="CAM-B-002"
                  isOnline={false}
                  location="South Approach"
                  name="Camera B"
                  resolution="—"
                />
              </>
            )}
          </div>
        </div>

        <div className="lg:col-span-2">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold text-white">Active Incidents</h2>
            <Link href="/incidents">
              <Button className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg">
                View All
              </Button>
            </Link>
          </div>
          <div className="space-y-4">
            {incidents.length === 0 && !loading ? (
              <div className="text-white/50 text-center py-8">
                No active incidents
              </div>
            ) : (
              incidents.map((inc) => (
                <AlertCard
                  key={inc.id}
                  cameraId={inc.camera_name ?? inc.camera_id}
                  description={inc.description ?? ""}
                  details={
                    inc.confidence
                      ? { confidence: `${Math.round(inc.confidence * 100)}%` }
                      : {}
                  }
                  id={String(inc.id)}
                  severity={inc.severity}
                  timestamp={timeAgo(inc.timestamp)}
                  title={inc.title}
                  type={mapType(inc.incident_type)}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
