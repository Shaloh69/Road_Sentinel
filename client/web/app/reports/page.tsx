"use client";

import { useState } from "react";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { addToast } from "@heroui/toast";

import { downloadCsv } from "@/lib/export";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

// Phase 1 note: there is no dedicated "saved reports" backend yet (that's a
// Phase 2 feature — a real `recordings`/reports table + generation history).
// Rather than leave the old hardcoded fixture list with a non-functional
// Download button, this generates real reports on demand from data that
// already exists (detections, incidents, hourly analytics).
const REPORT_TYPES = [
  {
    id: "detections",
    name: "Vehicle Detections",
    description: "All logged vehicle detections with speed and confidence",
    endpoint: "/api/detections?limit=1000",
    columns: [
      "timestamp",
      "camera_id",
      "vehicle_type",
      "speed",
      "confidence",
      "direction",
      "lane_number",
    ] as const,
  },
  {
    id: "incidents",
    name: "Incidents & Alerts",
    description: "All logged incidents with severity and status",
    endpoint: "/api/incidents?limit=500",
    columns: [
      "timestamp",
      "camera_name",
      "incident_type",
      "severity",
      "status",
      "title",
      "description",
    ] as const,
  },
  {
    id: "hourly",
    name: "Hourly Traffic Analytics",
    description: "Aggregated per-hour vehicle counts and speeds, today",
    endpoint: `/api/analytics/hourly?date=${new Date().toISOString().slice(0, 10)}`,
    columns: [
      "hour_timestamp",
      "camera_id",
      "total_vehicles",
      "avg_speed",
      "incident_count",
      "speeding_violations",
    ] as const,
  },
  {
    id: "violations",
    name: "Speed Violations by Hour (thesis figure)",
    description:
      "Speed-limit violations bucketed by hour of day, today — all 24 hours included (0 where none)",
    endpoint: `/api/analytics/violations?date=${new Date().toISOString().slice(0, 10)}`,
    columns: ["hour", "violations", "avg_speed", "max_speed"] as const,
  },
];

export default function ReportsPage() {
  const [generating, setGenerating] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const generate = async (report: (typeof REPORT_TYPES)[number]) => {
    setGenerating(report.id);
    setMessage(null);
    try {
      const res = await fetch(`${API}${report.endpoint}`);
      const json = await res.json();

      if (!json.success || !Array.isArray(json.data)) {
        setMessage(`${report.name}: no data available`);

        return;
      }
      if (json.data.length === 0) {
        setMessage(`${report.name}: no records yet — nothing to export`);

        return;
      }
      downloadCsv(
        `road-sentinel-${report.id}-${new Date().toISOString().slice(0, 10)}`,
        [...report.columns],
        json.data.map((row: Record<string, unknown>) =>
          report.columns.map((c) => row[c] as string | number | null),
        ),
      );
      setMessage(`${report.name}: downloaded ${json.data.length} rows`);
      addToast({
        title: "Report downloaded",
        description: `${report.name}: ${json.data.length} rows`,
        color: "success",
      });
    } catch {
      setMessage(`${report.name}: cannot reach server`);
      addToast({
        title: "Report download failed",
        description: `${report.name}: cannot reach server`,
        color: "danger",
      });
    } finally {
      setGenerating(null);
    }
  };

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-heading font-bold text-fg mb-2">
          Reports
        </h1>
        <p className="text-fg-muted">
          Generate CSV reports from real detection, incident, and analytics data
        </p>
      </div>

      <Card className="bg-surface/80 backdrop-blur-md border border-border shadow-xl">
        <CardHeader className="bg-surface-2/60 backdrop-blur-sm px-4 py-3 border-b border-border">
          <h3 className="text-xl font-heading font-bold text-fg">
            Available Reports
          </h3>
        </CardHeader>
        <CardBody className="p-4">
          <div className="space-y-3">
            {REPORT_TYPES.map((report) => (
              <div
                key={report.id}
                className="flex items-center justify-between p-4 bg-surface-2/60 backdrop-blur-sm rounded-lg border border-border"
              >
                <div className="flex items-center gap-4">
                  <div className="bg-brand/15 p-3 rounded-lg">
                    <svg
                      className="w-6 h-6 text-brand"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="text-fg font-semibold">{report.name}</p>
                    <p className="text-fg-muted text-sm">
                      {report.description} • CSV
                    </p>
                  </div>
                </div>
                <Button
                  className="font-semibold"
                  color="primary"
                  isLoading={generating === report.id}
                  size="sm"
                  onClick={() => generate(report)}
                >
                  Download
                </Button>
              </div>
            ))}
          </div>
          {message && <p className="mt-4 text-sm text-brand px-1">{message}</p>}
          <p className="mt-4 text-xs text-fg-muted/70 px-1">
            A saved-reports history with scheduled generation is planned for a
            later pass — these are generated fresh from live data each time.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
