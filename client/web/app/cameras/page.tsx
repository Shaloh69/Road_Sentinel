"use client";

import { useCallback, useEffect, useState } from "react";
import { CameraStatus } from "@/components/camera-status";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import { Slider } from "@heroui/slider";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

interface Camera {
  id: string;
  name: string;
  location: string;
  rtsp_url: string;
  status: "online" | "offline" | "error";
  fps: number;
  resolution: string;
  speed_limit: number;
  detection_confidence: number;
  pixels_per_meter: number;
}

export default function CamerasPage() {
  const [cameras,    setCameras]    = useState<Camera[]>([]);
  const [selected,   setSelected]   = useState<Camera | null>(null);
  const [saving,     setSaving]     = useState(false);
  const [saveMsg,    setSaveMsg]    = useState<string | null>(null);
  const [formState,  setFormState]  = useState<Partial<Camera>>({});
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState<string | null>(null);

  const fetchCameras = useCallback(async () => {
    try {
      const res  = await fetch(`${API}/api/cameras`);
      const json = await res.json();
      if (json.success) {
        setCameras(json.data);
        if (!selected && json.data.length > 0) {
          setSelected(json.data[0]);
          setFormState(json.data[0]);
        }
        setError(null);
      }
    } catch {
      setError("Cannot reach server");
    } finally {
      setLoading(false);
    }
  }, [selected]);

  useEffect(() => {
    fetchCameras();
    const id = setInterval(fetchCameras, 10000);
    return () => clearInterval(id);
  }, [fetchCameras]);

  const selectCamera = (cam: Camera) => {
    setSelected(cam);
    setFormState(cam);
    setSaveMsg(null);
  };

  const saveCamera = async () => {
    if (!selected) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await fetch(`${API}/api/cameras/${selected.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formState),
      });
      const json = await res.json();
      setSaveMsg(json.success ? "Saved successfully" : (json.error ?? "Save failed"));
      if (json.success) fetchCameras();
    } catch {
      setSaveMsg("Network error — check server connection");
    } finally {
      setSaving(false);
    }
  };

  const testConnection = async () => {
    if (!selected) return;
    try {
      const res  = await fetch(`${API}/api/cameras/${selected.id}`);
      const json = await res.json();
      setSaveMsg(json.success ? `Connected — status: ${json.data.status}` : "Test failed");
    } catch {
      setSaveMsg("Cannot reach camera");
    }
  };

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Camera Management</h1>
        <p className="text-white/70">Configure and manage traffic cameras</p>
        {error && (
          <p className="mt-2 text-sm text-red-400 bg-red-900/30 border border-red-500/30 px-3 py-2 rounded-lg">
            ⚠ {error}
          </p>
        )}
      </div>

      {/* Camera cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {loading ? (
          <div className="text-white/50 col-span-3 text-center py-8">Loading cameras…</div>
        ) : cameras.map(cam => (
          <div
            key={cam.id}
            onClick={() => selectCamera(cam)}
            className={`cursor-pointer rounded-xl transition-all ${selected?.id === cam.id ? "ring-2 ring-white/60" : ""}`}
          >
            <CameraStatus
              id={cam.id}
              name={cam.name}
              location={cam.location}
              isOnline={cam.status === "online"}
              fps={cam.fps}
              resolution={cam.resolution}
              detectionRate={cam.status === "online" ? 97 : 0}
            />
          </div>
        ))}
        <Card className="bg-white/10 backdrop-blur-md hover:scale-105 transition-transform shadow-lg border-2 border-dashed border-white/30">
          <CardBody className="p-6 flex items-center justify-center">
            <Button className="bg-white text-[#1B1931] font-semibold hover:bg-white/90 shadow-lg">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add New Camera
            </Button>
          </CardBody>
        </Card>
      </div>

      {/* Configuration panel */}
      {selected && (
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
          <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
            <h3 className="text-xl font-bold text-white">
              Camera Configuration — {selected.name}
            </h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-white">Basic Settings</h4>
                <Input
                  label="Camera Name"
                  value={formState.name ?? ""}
                  onChange={e => setFormState(s => ({ ...s, name: e.target.value }))}
                  classNames={{ label: "text-white/80", input: "text-white", inputWrapper: "bg-white/10 border-white/20" }}
                />
                <Input
                  label="Location"
                  value={formState.location ?? ""}
                  onChange={e => setFormState(s => ({ ...s, location: e.target.value }))}
                  classNames={{ label: "text-white/80", input: "text-white", inputWrapper: "bg-white/10 border-white/20" }}
                />
                <Input
                  label="RTSP URL"
                  type="password"
                  value={formState.rtsp_url ?? ""}
                  onChange={e => setFormState(s => ({ ...s, rtsp_url: e.target.value }))}
                  classNames={{ label: "text-white/80", input: "text-white", inputWrapper: "bg-white/10 border-white/20" }}
                />
              </div>

              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-white">Detection Settings</h4>
                <Slider
                  label="Detection Confidence Threshold"
                  step={0.01}
                  minValue={0}
                  maxValue={1}
                  value={formState.detection_confidence ?? 0.75}
                  onChange={v => setFormState(s => ({ ...s, detection_confidence: v as number }))}
                  className="max-w-md"
                  classNames={{ label: "text-white/80", value: "text-white", track: "bg-white/20", filler: "bg-white/70" }}
                />
                <Slider
                  label="Speed Limit (km/h)"
                  step={5}
                  minValue={30}
                  maxValue={120}
                  value={formState.speed_limit ?? 60}
                  onChange={v => setFormState(s => ({ ...s, speed_limit: v as number }))}
                  className="max-w-md"
                  classNames={{ label: "text-white/80", value: "text-white", track: "bg-white/20", filler: "bg-white/70" }}
                />
                <Input
                  label="Pixels Per Meter (PPM)"
                  type="number"
                  value={String(formState.pixels_per_meter ?? 25.5)}
                  onChange={e => setFormState(s => ({ ...s, pixels_per_meter: parseFloat(e.target.value) }))}
                  description="For speed calculation calibration"
                  classNames={{ label: "text-white/80", input: "text-white", inputWrapper: "bg-white/10 border-white/20", description: "text-white/70" }}
                />
              </div>
            </div>

            {saveMsg && (
              <p className={`mt-4 text-sm px-3 py-2 rounded-lg ${saveMsg.includes("success") || saveMsg.includes("Connected") ? "text-green-400 bg-green-900/30 border border-green-500/30" : "text-red-400 bg-red-900/30 border border-red-500/30"}`}>
                {saveMsg}
              </p>
            )}

            <div className="flex justify-end gap-3 mt-6 pt-6 border-t border-white/10">
              <Button
                className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg"
                onClick={testConnection}
              >
                Test Connection
              </Button>
              <Button
                className="bg-white text-[#1B1931] font-semibold hover:bg-white/90 shadow-lg"
                onClick={saveCamera}
                isLoading={saving}
              >
                Save Changes
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* Calibration Tool */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl mt-6">
        <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
          <h3 className="text-xl font-bold text-white">Camera Calibration</h3>
        </CardHeader>
        <CardBody className="p-6">
          <p className="text-white/80 mb-4">
            Use this tool to calibrate camera perspective and set reference points for accurate speed detection.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Button className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg">
              Open Calibration Tool
            </Button>
            <Button className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg">
              View Calibration Guide
            </Button>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
