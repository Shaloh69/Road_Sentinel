"use client";

import { useCallback, useEffect, useState } from "react";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { StatCard } from "@/components/stat-card";
import { Select, SelectItem } from "@heroui/select";
import { Button } from "@heroui/button";
import { Progress } from "@heroui/progress";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

interface Summary {
  vehicles_today: number;
  average_speed: number | null;
  incidents_today: number;
  cameras_online: number;
}

interface HourlyRow {
  hour_timestamp: string;
  total_vehicles: number;
  avg_speed: number | null;
  car_count: number;
  truck_count: number;
  bus_count: number;
  motorcycle_count: number;
  bicycle_count: number;
}

interface SpeedBucket {
  speed_bucket: number;
  count: number;
}

export default function AnalyticsPage() {
  const [summary,  setSummary]  = useState<Summary | null>(null);
  const [hourly,   setHourly]   = useState<HourlyRow[]>([]);
  const [speeds,   setSpeeds]   = useState<SpeedBucket[]>([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState("today");

  const hoursForRange: Record<string, number> = {
    today: 24, week: 168, month: 720, year: 8760,
  };

  const fetchData = useCallback(async () => {
    try {
      const hours = hoursForRange[timeRange] ?? 24;
      const today = new Date().toISOString().slice(0, 10);

      const [sumRes, hourlyRes, speedRes] = await Promise.all([
        fetch(`${API}/api/analytics/summary`),
        fetch(`${API}/api/analytics/hourly?date=${today}`),
        fetch(`${API}/api/analytics/speed?hours=${hours}`),
      ]);

      const [sumJson, hourlyJson, speedJson] = await Promise.all([
        sumRes.json(), hourlyRes.json(), speedRes.json(),
      ]);

      if (sumJson.success)    setSummary(sumJson.data);
      if (hourlyJson.success) setHourly(hourlyJson.data);
      if (speedJson.success)  setSpeeds(speedJson.data);
      setError(null);
    } catch {
      setError("Cannot reach server");
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 30000);
    return () => clearInterval(id);
  }, [fetchData]);

  // Build vehicle type breakdown from latest hourly data
  const totals = hourly.reduce(
    (acc, r) => ({
      car:        acc.car        + (r.car_count        ?? 0),
      truck:      acc.truck      + (r.truck_count      ?? 0),
      bus:        acc.bus        + (r.bus_count        ?? 0),
      motorcycle: acc.motorcycle + (r.motorcycle_count ?? 0),
      bicycle:    acc.bicycle    + (r.bicycle_count    ?? 0),
    }),
    { car: 0, truck: 0, bus: 0, motorcycle: 0, bicycle: 0 }
  );
  const totalVehicles = Object.values(totals).reduce((a, b) => a + b, 0) || 1;

  const vehicleTypes = [
    { type: "Car",        count: totals.car,        percentage: Math.round((totals.car        / totalVehicles) * 100) },
    { type: "Motorcycle", count: totals.motorcycle,  percentage: Math.round((totals.motorcycle / totalVehicles) * 100) },
    { type: "Bus",        count: totals.bus,         percentage: Math.round((totals.bus        / totalVehicles) * 100) },
    { type: "Truck",      count: totals.truck,       percentage: Math.round((totals.truck      / totalVehicles) * 100) },
    { type: "Bicycle",    count: totals.bicycle,     percentage: Math.round((totals.bicycle    / totalVehicles) * 100) },
  ];

  // Speed distribution labels from API buckets
  const speedBucketLabel = (bucket: number) =>
    `${bucket}–${bucket + 9} km/h`;
  const speedTotal = speeds.reduce((a, b) => a + b.count, 0) || 1;

  // Find peak hour
  const peakHour = hourly.reduce(
    (best, r) => (r.total_vehicles > (best?.total_vehicles ?? 0) ? r : best),
    null as HourlyRow | null
  );
  const peakLabel = peakHour
    ? new Date(peakHour.hour_timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "N/A";

  const maxHourly = Math.max(...hourly.map(r => r.total_vehicles), 1);

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Analytics</h1>
        <p className="text-white/70">Traffic statistics and trends analysis</p>
        {error && (
          <p className="mt-2 text-sm text-red-400 bg-red-900/30 border border-red-500/30 px-3 py-2 rounded-lg">
            ⚠ {error}
          </p>
        )}
      </div>

      {/* Time Range + Export */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 mb-6 shadow-xl">
        <CardBody className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <Select
              label="Time Range"
              defaultSelectedKeys={["today"]}
              className="max-w-xs"
              onChange={e => setTimeRange(e.target.value)}
              classNames={{
                label: "text-white/80",
                trigger: "bg-white/10 border-white/20 text-white",
              }}
            >
              <SelectItem key="today">Today</SelectItem>
              <SelectItem key="week">This Week</SelectItem>
              <SelectItem key="month">This Month</SelectItem>
              <SelectItem key="year">This Year</SelectItem>
            </Select>
            <div className="flex gap-2">
              <Button className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg">
                Export PDF
              </Button>
              <Button className="bg-white text-[#1B1931] font-semibold hover:bg-white/90 shadow-lg">
                Export CSV
              </Button>
            </div>
          </div>
        </CardBody>
      </Card>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <StatCard
          title="Total Vehicles"
          value={loading ? "—" : (summary?.vehicles_today ?? 0).toLocaleString()}
          trend={{ value: "live", isPositive: true }}
        />
        <StatCard
          title="Avg Speed"
          value={loading ? "—" : (summary?.average_speed ? `${summary.average_speed} km/h` : "N/A")}
          trend={{ value: "live", isPositive: true }}
        />
        <StatCard
          title="Peak Hour"
          value={loading ? "—" : peakLabel}
          subtitle={peakHour ? `${peakHour.total_vehicles} vehicles` : ""}
        />
        <StatCard
          title="Incidents"
          value={loading ? "—" : (summary?.incidents_today ?? 0)}
          subtitle="Today"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Vehicle Types */}
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
          <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
            <h3 className="text-xl font-bold text-white">Vehicle Types Distribution</h3>
          </CardHeader>
          <CardBody className="p-6">
            {loading ? (
              <div className="text-white/50 text-center py-8">Loading…</div>
            ) : (
              <div className="space-y-4">
                {vehicleTypes.map((v, i) => (
                  <div key={i} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-white/80 font-medium">{v.type}</span>
                      <span className="text-white font-bold">{v.count.toLocaleString()}</span>
                    </div>
                    <Progress
                      value={v.percentage}
                      className="h-3"
                      classNames={{ indicator: "bg-white/70", track: "bg-white/20" }}
                    />
                    <span className="text-xs text-white/70">{v.percentage}%</span>
                  </div>
                ))}
              </div>
            )}
          </CardBody>
        </Card>

        {/* Speed Distribution */}
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
          <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
            <h3 className="text-xl font-bold text-white">Speed Distribution</h3>
          </CardHeader>
          <CardBody className="p-6">
            {loading ? (
              <div className="text-white/50 text-center py-8">Loading…</div>
            ) : speeds.length === 0 ? (
              <div className="text-white/50 text-center py-8">No speed data yet</div>
            ) : (
              <div className="space-y-4">
                {speeds.map((s, i) => {
                  const pct = Math.round((s.count / speedTotal) * 100);
                  const color = s.speed_bucket < 30 ? "bg-green-500"
                              : s.speed_bucket < 50 ? "bg-blue-500"
                              : s.speed_bucket < 70 ? "bg-yellow-500"
                              : s.speed_bucket < 90 ? "bg-orange-500"
                              : "bg-red-500";
                  return (
                    <div key={i} className="flex items-center gap-3">
                      <div className={`w-4 h-4 rounded-full flex-shrink-0 ${color}`} />
                      <div className="flex-1 flex justify-between items-center">
                        <span className="text-white/80">{speedBucketLabel(s.speed_bucket)}</span>
                        <span className="text-white font-bold">{s.count} ({pct}%)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardBody>
        </Card>
      </div>

      {/* Hourly Traffic */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
        <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
          <h3 className="text-xl font-bold text-white">Hourly Traffic Flow</h3>
        </CardHeader>
        <CardBody className="p-6">
          {loading ? (
            <div className="text-white/50 text-center py-12">Loading…</div>
          ) : hourly.length === 0 ? (
            <div className="text-white/50 text-center py-12">No hourly data yet</div>
          ) : (
            <div className="flex items-end justify-between h-64 gap-1">
              {hourly.map((row, i) => {
                const heightPct = (row.total_vehicles / maxHourly) * 100;
                const label = new Date(row.hour_timestamp)
                  .toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                return (
                  <div key={i} className="flex-1 flex flex-col items-center gap-2">
                    <div className="relative w-full flex items-end justify-center" style={{ height: "200px" }}>
                      <div
                        className="w-full bg-gradient-to-t from-white/70 to-white/40 rounded-t-lg hover:from-white/80 hover:to-white/50 transition-all cursor-pointer"
                        style={{ height: `${heightPct}%` }}
                      >
                        <span className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-white text-xs font-bold">
                          {row.total_vehicles}
                        </span>
                      </div>
                    </div>
                    <span className="text-xs text-white/80">{label}</span>
                  </div>
                );
              })}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
