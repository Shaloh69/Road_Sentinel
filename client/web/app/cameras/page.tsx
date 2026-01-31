import { CameraStatus } from "@/components/camera-status";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import { Slider } from "@heroui/slider";

export default function CamerasPage() {
  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Camera Management</h1>
        <p className="text-white/70">Configure and manage traffic cameras</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <CameraStatus
          id="CAM-A-001"
          name="Camera A"
          location="North Approach - Blind Curve"
          isOnline={true}
          fps={30}
          resolution="1920x1080"
          detectionRate={98}
        />
        <CameraStatus
          id="CAM-B-002"
          name="Camera B"
          location="South Approach - Blind Curve"
          isOnline={true}
          fps={30}
          resolution="1920x1080"
          detectionRate={96}
        />
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

      {/* Camera Configuration */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
        <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
          <h3 className="text-xl font-bold text-white">Camera Configuration - Camera A</h3>
        </CardHeader>
        <CardBody className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white">Basic Settings</h4>
              <Input
                label="Camera Name"
                defaultValue="Camera A"
                classNames={{
                  label: "text-white/80",
                  input: "text-white",
                  inputWrapper: "bg-white/10 border-white/20",
                }}
              />
              <Input
                label="Location"
                defaultValue="North Approach - Blind Curve"
                classNames={{
                  label: "text-white/80",
                  input: "text-white",
                  inputWrapper: "bg-white/10 border-white/20",
                }}
              />
              <Input
                label="RTSP URL"
                type="password"
                defaultValue="rtsp://192.168.1.100:554/stream"
                classNames={{
                  label: "text-white/80",
                  input: "text-white",
                  inputWrapper: "bg-white/10 border-white/20",
                }}
              />
            </div>

            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white">Detection Settings</h4>
              <Slider
                label="Detection Confidence Threshold"
                step={0.01}
                minValue={0}
                maxValue={1}
                defaultValue={0.75}
                className="max-w-md"
                classNames={{
                  label: "text-white/80",
                  value: "text-white",
                  track: "bg-white/20",
                  filler: "bg-white/70",
                }}
              />
              <Slider
                label="Speed Limit (km/h)"
                step={5}
                minValue={30}
                maxValue={120}
                defaultValue={60}
                className="max-w-md"
                classNames={{
                  label: "text-white/80",
                  value: "text-white",
                  track: "bg-white/20",
                  filler: "bg-white/70",
                }}
              />
              <Input
                label="Pixels Per Meter (PPM)"
                type="number"
                defaultValue="25.5"
                description="For speed calculation calibration"
                classNames={{
                  label: "text-white/80",
                  input: "text-white",
                  inputWrapper: "bg-white/10 border-white/20",
                  description: "text-white/70",
                }}
              />
            </div>
          </div>

          <div className="flex justify-end gap-3 mt-6 pt-6 border-t border-white/10">
            <Button className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg">
              Test Connection
            </Button>
            <Button className="bg-white text-[#1B1931] font-semibold hover:bg-white/90 shadow-lg">
              Save Changes
            </Button>
          </div>
        </CardBody>
      </Card>

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
