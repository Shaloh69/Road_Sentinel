"use client";

import { VideoFeed } from "@/components/video-feed";
import { StatCard } from "@/components/stat-card";
import { Button } from "@heroui/button";
import { Switch } from "@heroui/switch";
import { Select, SelectItem } from "@heroui/select";
import { Card, CardBody } from "@heroui/card";
import { useState } from "react";

export default function MonitorPage() {
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [viewMode, setViewMode] = useState("grid");

  // Mock data - replace with real-time data from API
  const cameras = [
    {
      id: "CAM-A-001",
      name: "Camera A - North Approach",
      isLive: true,
      fps: 30,
      latency: 42,
      boundingBoxes: [
        {
          id: "1",
          x: 20,
          y: 30,
          width: 15,
          height: 20,
          label: "Car",
          confidence: 0.95,
          speed: 65,
        },
        {
          id: "2",
          x: 50,
          y: 40,
          width: 12,
          height: 18,
          label: "Motorcycle",
          confidence: 0.89,
          speed: 48,
        },
        {
          id: "3",
          x: 70,
          y: 50,
          width: 14,
          height: 19,
          label: "Car",
          confidence: 0.91,
          speed: 55,
        },
      ],
    },
    {
      id: "CAM-B-002",
      name: "Camera B - South Approach",
      isLive: true,
      fps: 30,
      latency: 38,
      boundingBoxes: [
        {
          id: "4",
          x: 35,
          y: 35,
          width: 18,
          height: 22,
          label: "Bus",
          confidence: 0.92,
          speed: 52,
        },
        {
          id: "5",
          x: 60,
          y: 45,
          width: 11,
          height: 16,
          label: "Motorcycle",
          confidence: 0.87,
          speed: 62,
        },
      ],
    },
  ];

  const totalVehicles = cameras.reduce((sum, cam) => sum + cam.boundingBoxes.length, 0);
  const avgSpeed = Math.round(
    cameras.reduce((sum, cam) =>
      sum + cam.boundingBoxes.reduce((s, box) => s + (box.speed || 0), 0), 0
    ) / totalVehicles
  );

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Live Monitoring</h1>
        <p className="text-white/70">Real-time camera feeds with AI-powered vehicle detection</p>
      </div>

      {/* Controls */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 mb-6 shadow-xl">
        <CardBody className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <Switch
                isSelected={showBoundingBoxes}
                onValueChange={setShowBoundingBoxes}
                classNames={{
                  wrapper: "group-data-[selected=true]:bg-white/30",
                }}
              >
                <span className="text-white/80">Show Bounding Boxes</span>
              </Switch>

              <Select
                label="View Mode"
                placeholder="Select view mode"
                className="max-w-xs"
                defaultSelectedKeys={["grid"]}
                classNames={{
                  label: "text-white/80",
                  trigger: "bg-white/10 border-white/20 text-white",
                }}
              >
                <SelectItem key="grid">Grid View</SelectItem>
                <SelectItem key="single">Single Camera</SelectItem>
                <SelectItem key="split">Split Screen</SelectItem>
              </Select>
            </div>

            <div className="flex items-center gap-3">
              <Button className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Record
              </Button>
              <Button className="bg-white text-[#1B1931] font-semibold hover:bg-white/90 shadow-lg">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Snapshot
              </Button>
            </div>
          </div>
        </CardBody>
      </Card>

      {/* Live Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          title="Total Vehicles"
          value={totalVehicles}
          subtitle="Currently detected"
        />
        <StatCard
          title="Average Speed"
          value={`${avgSpeed} km/h`}
          subtitle="All cameras"
        />
        <StatCard
          title="Camera A Vehicles"
          value={cameras[0].boundingBoxes.length}
          subtitle="North Approach"
        />
        <StatCard
          title="Camera B Vehicles"
          value={cameras[1].boundingBoxes.length}
          subtitle="South Approach"
        />
      </div>

      {/* Camera Feeds */}
      <div className={`grid gap-6 ${viewMode === "grid" ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}>
        {cameras.map((camera) => (
          <VideoFeed
            key={camera.id}
            cameraId={camera.id}
            cameraName={camera.name}
            isLive={camera.isLive}
            fps={camera.fps}
            latency={camera.latency}
            showBoundingBoxes={showBoundingBoxes}
            boundingBoxes={camera.boundingBoxes}
          />
        ))}
      </div>

      {/* Detection Log */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 mt-6 shadow-xl">
        <CardBody className="p-4">
          <h3 className="text-xl font-bold text-white mb-4">Detection Log</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {cameras.flatMap(camera =>
              camera.boundingBoxes.map(box => (
                <div key={box.id} className="flex justify-between items-center p-3 bg-white/10 backdrop-blur-sm rounded-lg border border-white/10">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-white/80">{camera.name}</span>
                    <span className="text-white font-semibold">{box.label}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-white/80 text-sm">{box.speed} km/h</span>
                    <span className="text-white/70 text-sm">{(box.confidence * 100).toFixed(0)}% confidence</span>
                    <span className="text-white/50 text-xs">Just now</span>
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
