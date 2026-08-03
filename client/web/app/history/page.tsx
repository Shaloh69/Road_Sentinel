"use client";

import { useCallback, useEffect, useState } from "react";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Input } from "@heroui/input";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

// Phase 2 note: recording is real now (raspi_scripts/camera/camera_sender.py
// --record, opt-in, off by default) but hasn't been exercised on real camera
// hardware yet — so this page will typically show zero recordings until
// that flag is turned on and verified in Phase 4. The detection log below
// always has real data regardless, so it stays as the primary view.

interface Detection {
  id: number;
  camera_id: string;
  timestamp: string;
  vehicle_type: string;
  speed: number | null;
  confidence: number;
  direction: string | null;
}

interface Recording {
  id: string;
  camera_id: string;
  start_time: string;
  duration_seconds: number | null;
  video_url: string | null;
  status: string;
  vehicle_count: number;
  incident_count: number;
}

export default function HistoryPage() {
  const [selectedDate, setSelectedDate] = useState(
    new Date().toISOString().slice(0, 10),
  );
  const [detections, setDetections] = useState<Detection[]>([]);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selectedRecording, setSelectedRecording] = useState<Recording | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const since = `${selectedDate}T00:00:00`;
      const nextDay = new Date(`${selectedDate}T00:00:00`);

      nextDay.setDate(nextDay.getDate() + 1);
      const until = nextDay.toISOString().slice(0, 19).replace("T", " ");

      const [detRes, recRes] = await Promise.all([
        fetch(
          `${API}/api/detections?since=${encodeURIComponent(since)}&until=${encodeURIComponent(until)}&limit=200`,
        ),
        fetch(`${API}/api/recordings?date=${selectedDate}`),
      ]);
      const [detJson, recJson] = await Promise.all([
        detRes.json(),
        recRes.json(),
      ]);

      if (detJson.success) setDetections(detJson.data);
      if (recJson.success) {
        setRecordings(recJson.data);
        setSelectedRecording((prev) =>
          prev && recJson.data.some((r: Recording) => r.id === prev.id)
            ? prev
            : (recJson.data[0] ?? null),
        );
      }
      setError(null);
    } catch {
      setError("Cannot reach server");
    } finally {
      setLoading(false);
    }
  }, [selectedDate]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-heading font-bold text-fg mb-2">
          History
        </h1>
        <p className="text-fg-muted">
          Recorded footage and logged vehicle detections by date
        </p>
        {error && (
          <p className="mt-2 text-sm text-danger bg-danger/10 border border-danger/30 px-3 py-2 rounded-lg">
            ⚠ {error}
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video playback */}
        <div className="lg:col-span-2">
          <Card className="bg-surface/80 backdrop-blur-md border border-border shadow-xl">
            <CardHeader className="bg-surface-2/60 backdrop-blur-sm px-4 py-3 border-b border-border">
              <h3 className="text-xl font-heading font-bold text-fg">
                Video Playback
              </h3>
            </CardHeader>
            <CardBody className="p-0">
              {selectedRecording?.video_url ? (
                // eslint-disable-next-line jsx-a11y/media-has-caption
                <video
                  controls
                  className="w-full aspect-video bg-black"
                  src={selectedRecording.video_url}
                />
              ) : (
                <div className="aspect-video bg-surface-2/60 backdrop-blur-sm flex items-center justify-center">
                  <div className="text-center max-w-sm px-6">
                    <svg
                      className="w-16 h-16 text-fg mx-auto mb-4 opacity-40"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                      />
                      <path
                        d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                      />
                    </svg>
                    <p className="text-fg-muted font-medium">
                      No recordings for {selectedDate}
                    </p>
                    <p className="text-fg-muted/70 text-sm mt-2">
                      Recording is opt-in (`camera_sender.py --record`) and off
                      by default. The detection log below always reflects real
                      logged data regardless.
                    </p>
                  </div>
                </div>
              )}
            </CardBody>
          </Card>
        </div>

        {/* Date filter */}
        <div className="space-y-4">
          <Card className="bg-surface/80 backdrop-blur-md border border-border shadow-xl">
            <CardHeader className="bg-surface-2/60 backdrop-blur-sm px-4 py-3 border-b border-border">
              <h3 className="text-lg font-heading font-bold text-fg">
                Filter by Date
              </h3>
            </CardHeader>
            <CardBody className="p-4 space-y-4">
              <Input
                classNames={{
                  label: "text-fg-muted",
                  input: "text-fg",
                  inputWrapper: "bg-surface-2 border-border",
                }}
                label="Date"
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
              />
              <p className="text-xs text-fg-muted/70">
                {loading
                  ? "Loading…"
                  : `${detections.length} detection(s), ${recordings.length} recording(s)`}
              </p>
            </CardBody>
          </Card>
        </div>
      </div>

      {/* Recordings list */}
      {recordings.length > 0 && (
        <Card className="bg-surface/80 backdrop-blur-md border border-border shadow-xl mt-6">
          <CardHeader className="bg-surface-2/60 backdrop-blur-sm px-4 py-3 border-b border-border">
            <h3 className="text-xl font-heading font-bold text-fg">
              Recordings on {selectedDate}
            </h3>
          </CardHeader>
          <CardBody className="p-4">
            <div className="space-y-2">
              {recordings.map((r) => (
                <button
                  key={r.id}
                  className={`w-full flex items-center justify-between p-3 rounded-lg border text-left transition-colors duration-150 ease-standard ${
                    selectedRecording?.id === r.id
                      ? "bg-surface-2 border-brand/40"
                      : "bg-surface-2/60 border-border hover:bg-surface-2/80"
                  }`}
                  onClick={() => setSelectedRecording(r)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-fg-muted text-sm">{r.camera_id}</span>
                    <span className="text-fg font-semibold font-mono">
                      {new Date(r.start_time).toLocaleTimeString()}
                    </span>
                    <span className="text-fg-muted/70 text-xs uppercase">
                      {r.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-fg-muted font-mono">
                    <span>{r.duration_seconds ?? "?"}s</span>
                    <span>{r.vehicle_count} vehicle frames</span>
                    <span>{r.incident_count} incidents</span>
                  </div>
                </button>
              ))}
            </div>
          </CardBody>
        </Card>
      )}

      {/* Detection log */}
      <Card className="bg-surface/80 backdrop-blur-md border border-border shadow-xl mt-6">
        <CardHeader className="bg-surface-2/60 backdrop-blur-sm px-4 py-3 border-b border-border">
          <h3 className="text-xl font-heading font-bold text-fg">
            Detections on {selectedDate}
          </h3>
        </CardHeader>
        <CardBody className="p-4">
          {loading ? (
            <div className="text-fg-muted/70 text-center py-8">Loading…</div>
          ) : detections.length === 0 ? (
            <div className="text-fg-muted/70 text-center py-8">
              No detections logged for this date
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {detections.map((d) => (
                <div
                  key={d.id}
                  className="flex items-center justify-between p-3 bg-surface-2/60 rounded-lg border border-border"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-fg-muted text-sm">{d.camera_id}</span>
                    <span className="text-fg font-semibold capitalize">
                      {d.vehicle_type}
                    </span>
                    {d.direction && (
                      <span className="text-fg-muted/70 text-xs">
                        {d.direction}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    {d.speed != null && (
                      <span className="text-fg font-bold font-mono">
                        {d.speed} km/h
                      </span>
                    )}
                    <span className="text-fg-muted font-mono">
                      {(d.confidence * 100).toFixed(0)}%
                    </span>
                    <span className="text-fg-muted/70 text-xs font-mono">
                      {new Date(d.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
