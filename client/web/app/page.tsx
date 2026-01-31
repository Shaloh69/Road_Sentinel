import { Link } from "@heroui/link";
import { Button } from "@heroui/button";
import { StatCard } from "@/components/stat-card";
import { VideoFeed } from "@/components/video-feed";
import { AlertCard } from "@/components/alert-card";
import { CameraStatus } from "@/components/camera-status";

export default function Home() {
  // Mock data - replace with real data from API
  const stats = {
    vehiclesToday: 2847,
    averageSpeed: 42,
    incidents: 3,
    activeCameras: 2,
  };

  const recentAlerts: Array<{
    id: string;
    type: "speeding" | "crash" | "anomaly" | "warning";
    severity: "low" | "medium" | "high" | "critical";
    title: string;
    description: string;
    timestamp: string;
    cameraId: string;
    details: Record<string, string | number>;
  }> = [
    {
      id: "1",
      type: "speeding" as const,
      severity: "high" as const,
      title: "Speed Violation Detected",
      description: "Vehicle exceeded speed limit by 25 km/h",
      timestamp: "2 minutes ago",
      cameraId: "Camera A",
      details: {
        speed: "85 km/h",
        limit: "60 km/h",
        vehicle: "Sedan",
      },
    },
    {
      id: "2",
      type: "warning" as const,
      severity: "medium" as const,
      title: "Heavy Traffic Detected",
      description: "Multiple vehicles approaching simultaneously",
      timestamp: "15 minutes ago",
      cameraId: "Camera B",
      details: {
        vehicles: "8",
        "avg speed": "45 km/h",
      },
    },
  ];

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-[#ED9E59] mb-2">Dashboard</h1>
        <p className="text-[#E8BCB8]">Real-time traffic monitoring system overview</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Vehicles Today"
          value={stats.vehiclesToday.toLocaleString()}
          trend={{ value: "12%", isPositive: true }}
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
        />
        <StatCard
          title="Average Speed"
          value={`${stats.averageSpeed} km/h`}
          trend={{ value: "3%", isPositive: false }}
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
        />
        <StatCard
          title="Incidents"
          value={stats.incidents}
          subtitle="Last 24 hours"
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
        />
        <StatCard
          title="Active Cameras"
          value={`${stats.activeCameras}/2`}
          subtitle="All systems operational"
          icon={
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          }
        />
      </div>

      {/* Live Feeds */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-[#ED9E59]">Live Camera Feeds</h2>
          <Link href="/monitor">
            <Button className="bg-[#ED9E59] text-[#1B1931] font-semibold hover:bg-[#A34054]">
              View All Feeds
            </Button>
          </Link>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <VideoFeed
            cameraId="CAM-A-001"
            cameraName="Camera A - North Approach"
            isLive={true}
            fps={30}
            latency={42}
            showBoundingBoxes={true}
            boundingBoxes={[
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
            ]}
          />
          <VideoFeed
            cameraId="CAM-B-002"
            cameraName="Camera B - South Approach"
            isLive={true}
            fps={30}
            latency={38}
            showBoundingBoxes={true}
            boundingBoxes={[
              {
                id: "3",
                x: 35,
                y: 35,
                width: 18,
                height: 22,
                label: "Bus",
                confidence: 0.92,
                speed: 52,
              },
            ]}
          />
        </div>
      </div>

      {/* Camera Status & Recent Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Camera Status */}
        <div className="lg:col-span-1">
          <h2 className="text-2xl font-bold text-[#ED9E59] mb-4">Camera Status</h2>
          <div className="space-y-4">
            <CameraStatus
              id="CAM-A-001"
              name="Camera A"
              location="North Approach"
              isOnline={true}
              fps={30}
              resolution="1920x1080"
              detectionRate={98}
            />
            <CameraStatus
              id="CAM-B-002"
              name="Camera B"
              location="South Approach"
              isOnline={true}
              fps={30}
              resolution="1920x1080"
              detectionRate={96}
            />
          </div>
        </div>

        {/* Recent Alerts */}
        <div className="lg:col-span-2">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold text-[#ED9E59]">Recent Alerts</h2>
            <Link href="/incidents">
              <Button className="bg-[#862249] text-[#E8BCB8] font-semibold hover:bg-[#A34054]">
                View All Alerts
              </Button>
            </Link>
          </div>
          <div className="space-y-4">
            {recentAlerts.map((alert) => (
              <AlertCard key={alert.id} {...alert} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
