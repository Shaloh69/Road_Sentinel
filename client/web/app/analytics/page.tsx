"use client";

import { Card, CardBody, CardHeader } from "@heroui/card";
import { StatCard } from "@/components/stat-card";
import { Select, SelectItem } from "@heroui/select";
import { Button } from "@heroui/button";
import { Progress } from "@heroui/progress";

export default function AnalyticsPage() {
  // Mock data - replace with real API data
  const vehicleTypes = [
    { type: "Car", count: 1847, percentage: 65 },
    { type: "Motorcycle", count: 623, percentage: 22 },
    { type: "Bus", count: 247, percentage: 9 },
    { type: "Truck", count: 98, percentage: 3 },
    { type: "Bicycle", count: 32, percentage: 1 },
  ];

  const hourlyData = [
    { hour: "00:00", vehicles: 45 },
    { hour: "03:00", vehicles: 12 },
    { hour: "06:00", vehicles: 186 },
    { hour: "09:00", vehicles: 342 },
    { hour: "12:00", vehicles: 298 },
    { hour: "15:00", vehicles: 315 },
    { hour: "18:00", vehicles: 387 },
    { hour: "21:00", vehicles: 214 },
  ];

  const speedRanges = [
    { range: "0-30 km/h", count: 234, color: "bg-green-500" },
    { range: "31-50 km/h", count: 1245, color: "bg-blue-500" },
    { range: "51-70 km/h", count: 982, color: "bg-yellow-500" },
    { range: "71-90 km/h", count: 312, color: "bg-orange-500" },
    { range: "91+ km/h", count: 74, color: "bg-red-500" },
  ];

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Analytics</h1>
        <p className="text-white/70">Traffic statistics and trends analysis</p>
      </div>

      {/* Time Range Selector */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 mb-6 shadow-xl">
        <CardBody className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <Select
              label="Time Range"
              placeholder="Select time range"
              className="max-w-xs"
              defaultSelectedKeys={["today"]}
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
          value="2,847"
          trend={{ value: "12%", isPositive: true }}
        />
        <StatCard
          title="Avg Speed"
          value="42 km/h"
          trend={{ value: "3%", isPositive: false }}
        />
        <StatCard
          title="Peak Hour"
          value="6:00 PM"
          subtitle="387 vehicles"
        />
        <StatCard
          title="Most Common"
          value="Car"
          subtitle="65% of traffic"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Vehicle Types Distribution */}
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
          <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
            <h3 className="text-xl font-bold text-white">Vehicle Types Distribution</h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="space-y-4">
              {vehicleTypes.map((vehicle, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-white/80 font-medium">{vehicle.type}</span>
                    <span className="text-white font-bold">{vehicle.count}</span>
                  </div>
                  <Progress
                    value={vehicle.percentage}
                    className="h-3"
                    classNames={{
                      indicator: "bg-white/70",
                      track: "bg-white/20",
                    }}
                  />
                  <span className="text-xs text-white/70">{vehicle.percentage}%</span>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>

        {/* Speed Distribution */}
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
          <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
            <h3 className="text-xl font-bold text-white">Speed Distribution</h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="space-y-4">
              {speedRanges.map((range, index) => (
                <div key={index} className="flex items-center gap-3">
                  <div className={`w-4 h-4 rounded-full ${range.color}`}></div>
                  <div className="flex-1 flex justify-between items-center">
                    <span className="text-white/80">{range.range}</span>
                    <span className="text-white font-bold">{range.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      </div>

      {/* Hourly Traffic Chart */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
        <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
          <h3 className="text-xl font-bold text-white">Hourly Traffic Flow</h3>
        </CardHeader>
        <CardBody className="p-6">
          <div className="flex items-end justify-between h-64 gap-2">
            {hourlyData.map((data, index) => {
              const maxVehicles = Math.max(...hourlyData.map(d => d.vehicles));
              const heightPercent = (data.vehicles / maxVehicles) * 100;

              return (
                <div key={index} className="flex-1 flex flex-col items-center gap-2">
                  <div className="relative w-full flex items-end justify-center" style={{ height: '200px' }}>
                    <div
                      className="w-full bg-gradient-to-t from-white/70 to-white/40 rounded-t-lg hover:from-white/80 hover:to-white/50 transition-all cursor-pointer"
                      style={{ height: `${heightPercent}%` }}
                    >
                      <span className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-white text-sm font-bold">
                        {data.vehicles}
                      </span>
                    </div>
                  </div>
                  <span className="text-xs text-white/80">{data.hour}</span>
                </div>
              );
            })}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
