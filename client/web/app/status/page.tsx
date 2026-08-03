"use client";

import { useCallback, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

// Public, no-login "live status" page for the community near the Busay blind
// curve (Phase 2 new feature). Deliberately minimal: only a safe/caution/
// incident state, no camera feed, no admin surface, no configuration —
// backed by GET /api/public/status, which is itself unauthenticated and
// exposes nothing beyond that same safety-relevant summary.

type RoadState = "clear" | "vehicle_incoming" | "incident";

interface PublicStatus {
  state: RoadState;
  detail: { incident_type?: string; severity?: string; camera_id?: string };
  cameras_online: number;
  cameras_total: number;
  vehicles_today: number;
  incidents_today: number;
  updated_at: string;
}

const STATE_CONFIG: Record<
  RoadState,
  { label: string; sub: string; bg: string; text: string }
> = {
  clear: {
    label: "ROAD CLEAR",
    sub: "No vehicles or incidents detected recently",
    bg: "bg-success",
    text: "text-[#072F20]",
  },
  vehicle_incoming: {
    label: "VEHICLE INCOMING",
    sub: "Slow down and proceed with caution",
    bg: "bg-warning",
    text: "text-[#12151C]",
  },
  incident: {
    label: "INCIDENT AHEAD",
    sub: "An incident was detected near the curve — proceed with extreme caution",
    bg: "bg-danger",
    text: "text-white",
  },
};

export default function PublicStatusPage() {
  const [status, setStatus] = useState<PublicStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/public/status`);
      const json = await res.json();

      if (json.success) {
        setStatus(json.data);
        setError(null);
      }
    } catch {
      setError("Cannot reach the status service");
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 3000);

    return () => clearInterval(id);
  }, [fetchStatus]);

  const cfg = STATE_CONFIG[status?.state ?? "clear"];

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-heading font-bold text-fg">
            Road Sentinel — Busay Blind Curve
          </h1>
          <p className="text-fg-muted text-sm mt-1">
            Live community status — updates every few seconds
          </p>
        </div>

        <div
          className={`rounded-2xl p-12 text-center shadow-2xl transition-colors duration-500 ease-standard ${status ? cfg.bg : "bg-surface-2"}`}
        >
          {status ? (
            <>
              <p
                className={`text-4xl md:text-5xl font-heading font-black ${cfg.text}`}
              >
                {cfg.label}
              </p>
              <p className={`mt-3 text-lg ${cfg.text} opacity-90`}>{cfg.sub}</p>
            </>
          ) : (
            <p className="text-fg-muted text-xl">Loading…</p>
          )}
        </div>

        {error && (
          <p className="mt-4 text-center text-sm text-danger">{error}</p>
        )}

        {status && (
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-surface-2/80 border border-border rounded-xl p-4">
              <p className="text-2xl font-heading font-bold text-fg font-mono">
                {status.cameras_online}/{status.cameras_total}
              </p>
              <p className="text-xs text-fg-muted mt-1">Cameras Online</p>
            </div>
            <div className="bg-surface-2/80 border border-border rounded-xl p-4">
              <p className="text-2xl font-heading font-bold text-fg font-mono">
                {status.vehicles_today.toLocaleString()}
              </p>
              <p className="text-xs text-fg-muted mt-1">Vehicles Today</p>
            </div>
            <div className="bg-surface-2/80 border border-border rounded-xl p-4">
              <p className="text-2xl font-heading font-bold text-fg font-mono">
                {status.incidents_today}
              </p>
              <p className="text-xs text-fg-muted mt-1">Incidents Today</p>
            </div>
            <div className="bg-surface-2/80 border border-border rounded-xl p-4">
              <p className="text-2xl font-heading font-bold text-fg font-mono">
                {new Date(status.updated_at).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })}
              </p>
              <p className="text-xs text-fg-muted mt-1">Last Updated</p>
            </div>
          </div>
        )}

        <p className="mt-8 text-center text-xs text-fg-muted/60">
          This page shows safety status only — no camera footage or admin
          controls are shown here.
        </p>
      </div>
    </div>
  );
}
