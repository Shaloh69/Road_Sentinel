"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCard } from "@/components/alert-card";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
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
  if (t === "crash")    return "crash";
  if (["wrong_way", "stopped_vehicle", "congestion", "illegal_parking"].includes(t))
    return "warning";
  return "anomaly";
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
  status: "active" | "resolved" | "false_alarm" | "investigating";
  confidence?: number;
}

type FilterStatus = "all" | "active" | "resolved" | "false_alarm" | "investigating";

export default function IncidentsPage() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [filter,    setFilter]    = useState<FilterStatus>("all");
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState<string | null>(null);
  const incidentIds = useRef(new Set<number>());

  const fetchIncidents = useCallback(async () => {
    try {
      const params = filter !== "all" ? `?status=${filter}&limit=100` : "?limit=100";
      const res  = await fetch(`${API}/api/incidents${params}`);
      const json = await res.json();
      if (json.success) {
        setIncidents(json.data);
        incidentIds.current = new Set(json.data.map((i: Incident) => i.id));
        setError(null);
      }
    } catch {
      setError("Cannot reach server");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchIncidents();
    const id = setInterval(fetchIncidents, 10000);
    return () => clearInterval(id);
  }, [fetchIncidents]);

  // WebSocket — instant updates from Node when a new incident is broadcast
  useEffect(() => {
    const socket = getSocket();
    socket.emit("subscribe_incidents");

    socket.on("incident", (event: { type: string; data: Incident }) => {
      const inc = event.data;
      if (inc.id != null && !incidentIds.current.has(inc.id)) {
        incidentIds.current.add(inc.id);
        // Only prepend if we're viewing "all" or "active" (new incidents start as active)
        if (filter === "all" || filter === "active") {
          setIncidents(prev => [inc, ...prev]);
        }
      }
    });

    return () => { socket.off("incident"); };
  }, [filter]);

  const updateStatus = async (id: number, status: Incident["status"]) => {
    try {
      await fetch(`${API}/api/incidents/${id}/status`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      fetchIncidents();
    } catch {
      // silent — will retry on next poll
    }
  };

  const total    = incidents.length;
  const critical = incidents.filter(i => i.severity === "critical").length;
  const active   = incidents.filter(i => i.status === "active").length;
  const resolved = incidents.filter(i => i.status === "resolved").length;

  const FILTERS: FilterStatus[] = ["all", "active", "resolved", "investigating", "false_alarm"];

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Incidents & Alerts</h1>
        <p className="text-white/70">Real-time incident monitoring and alert management</p>
        {error && (
          <p className="mt-2 text-sm text-red-400 bg-red-900/30 border border-red-500/30 px-3 py-2 rounded-lg">
            ⚠ {error}
          </p>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[
          { label: "Total",    value: total },
          { label: "Critical", value: critical },
          { label: "Active",   value: active },
          { label: "Resolved", value: resolved },
        ].map(s => (
          <Card key={s.label} className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
            <CardBody className="p-4">
              <p className="text-sm text-white/80 font-medium uppercase">{s.label}</p>
              <p className="text-3xl font-bold text-white mt-1">
                {loading ? "—" : s.value}
              </p>
            </CardBody>
          </Card>
        ))}
      </div>

      {/* Filter buttons */}
      <div className="flex flex-wrap gap-2 mb-6">
        {FILTERS.map(f => (
          <Button
            key={f}
            size="sm"
            onClick={() => setFilter(f)}
            className={
              filter === f
                ? "bg-white text-[#1B1931] font-semibold shadow-lg"
                : "bg-white/20 text-white border border-white/20 hover:bg-white/30"
            }
          >
            {f.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase())}
          </Button>
        ))}
      </div>

      {/* Incidents list */}
      <div className="space-y-4">
        {loading ? (
          <div className="text-white/50 text-center py-12">Loading incidents…</div>
        ) : incidents.length === 0 ? (
          <div className="text-white/50 text-center py-12">No incidents found</div>
        ) : incidents.map(inc => (
          <div key={inc.id} className="relative">
            <AlertCard
              id={String(inc.id)}
              type={mapType(inc.incident_type)}
              severity={inc.severity}
              title={inc.title}
              description={inc.description ?? ""}
              timestamp={timeAgo(inc.timestamp)}
              cameraId={inc.camera_name ?? inc.camera_id}
              details={inc.confidence ? { confidence: `${Math.round(inc.confidence * 100)}%` } : {}}
            />
            {inc.status === "active" && (
              <div className="flex gap-2 mt-1 ml-1">
                <Button size="sm"
                  className="bg-green-600/80 text-white text-xs hover:bg-green-600"
                  onClick={() => updateStatus(inc.id, "resolved")}>
                  Resolve
                </Button>
                <Button size="sm"
                  className="bg-white/20 text-white text-xs hover:bg-white/30 border border-white/20"
                  onClick={() => updateStatus(inc.id, "false_alarm")}>
                  False Alarm
                </Button>
                <Button size="sm"
                  className="bg-yellow-600/80 text-white text-xs hover:bg-yellow-600"
                  onClick={() => updateStatus(inc.id, "investigating")}>
                  Investigate
                </Button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
