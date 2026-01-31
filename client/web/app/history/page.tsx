"use client";

import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import { useState } from "react";

export default function HistoryPage() {
  const [selectedDate, setSelectedDate] = useState("2024-01-31");

  const recordings = [
    { id: "1", camera: "Camera A", timestamp: "14:30:25", duration: "00:15:32", vehicles: 23, incidents: 1 },
    { id: "2", camera: "Camera B", timestamp: "14:00:00", duration: "00:30:00", vehicles: 45, incidents: 0 },
    { id: "3", camera: "Camera A", timestamp: "13:30:15", duration: "00:20:18", vehicles: 31, incidents: 2 },
    { id: "4", camera: "Camera B", timestamp: "13:00:00", duration: "00:25:42", vehicles: 38, incidents: 1 },
  ];

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">History & Playback</h1>
        <p className="text-white/70">Review and playback recorded footage</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Player */}
        <div className="lg:col-span-2">
          <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
            <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
              <h3 className="text-xl font-bold text-white">Video Playback</h3>
            </CardHeader>
            <CardBody className="p-0">
              <div className="aspect-video bg-white/5 backdrop-blur-sm flex items-center justify-center">
                <div className="text-center">
                  <svg className="w-20 h-20 text-white mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-white/70">Select a recording to playback</p>
                </div>
              </div>
              {/* Playback Controls */}
              <div className="p-4 bg-white/10 backdrop-blur-sm border-t border-white/10">
                <div className="flex items-center gap-4">
                  <Button size="sm" className="bg-white text-[#1B1931]">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                    </svg>
                  </Button>
                  <div className="flex-1 h-2 bg-white/20 rounded-full overflow-hidden">
                    <div className="h-full w-1/3 bg-white/70"></div>
                  </div>
                  <span className="text-white/80 text-sm">00:05:20 / 00:15:32</span>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>

        {/* Search & Filters */}
        <div className="space-y-4">
          <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
            <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
              <h3 className="text-lg font-bold text-white">Search Recordings</h3>
            </CardHeader>
            <CardBody className="p-4 space-y-4">
              <Input
                type="date"
                label="Date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                classNames={{
                  label: "text-white/80",
                  input: "text-white",
                  inputWrapper: "bg-white/10 border-white/20",
                }}
              />
              <Input
                type="time"
                label="Time"
                classNames={{
                  label: "text-white/80",
                  input: "text-white",
                  inputWrapper: "bg-white/10 border-white/20",
                }}
              />
              <Button className="w-full bg-white text-[#1B1931] font-semibold hover:bg-white/90">
                Search
              </Button>
            </CardBody>
          </Card>
        </div>
      </div>

      {/* Recordings List */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl mt-6">
        <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
          <h3 className="text-xl font-bold text-white">Available Recordings</h3>
        </CardHeader>
        <CardBody className="p-4">
          <div className="space-y-3">
            {recordings.map((rec) => (
              <div key={rec.id} className="flex items-center justify-between p-4 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/15 transition-colors cursor-pointer border border-white/10">
                <div className="flex items-center gap-4">
                  <div className="bg-white/20 p-3 rounded-lg">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-white font-semibold">{rec.camera} - {rec.timestamp}</p>
                    <p className="text-white/70 text-sm">Duration: {rec.duration} | Vehicles: {rec.vehicles} | Incidents: {rec.incidents}</p>
                  </div>
                </div>
                <Button size="sm" className="bg-white text-[#1B1931] font-semibold hover:bg-white/90">
                  Play
                </Button>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
