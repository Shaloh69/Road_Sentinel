import { AlertCard } from "@/components/alert-card";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";

export default function IncidentsPage() {
  const incidents: Array<{
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
      id: "INC-001",
      type: "crash" as const,
      severity: "critical" as const,
      title: "Collision Detected",
      description: "Two vehicles involved in accident at blind curve",
      timestamp: "10 minutes ago",
      cameraId: "Camera A",
      details: {
        vehicles: "2",
        location: "North Approach",
        status: "Active",
      },
    },
    {
      id: "INC-002",
      type: "speeding" as const,
      severity: "high" as const,
      title: "Excessive Speed Violation",
      description: "Motorcycle exceeded speed limit by 35 km/h",
      timestamp: "25 minutes ago",
      cameraId: "Camera B",
      details: {
        speed: "95 km/h",
        limit: "60 km/h",
        vehicle: "Motorcycle",
      },
    },
  ];

  const stats = {
    total: incidents.length,
    critical: incidents.filter(i => i.severity === "critical").length,
    resolved: 12,
    active: incidents.length,
  };

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-[#ED9E59] mb-2">Incidents & Alerts</h1>
        <p className="text-[#E8BCB8]">Real-time incident monitoring and alert management</p>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card className="bg-gradient-to-br from-[#44174E] to-[#862249] border-none">
          <CardBody className="p-4">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-sm text-[#E8BCB8] font-medium uppercase">Total Incidents</p>
                <p className="text-3xl font-bold text-[#ED9E59] mt-1">{stats.total}</p>
              </div>
            </div>
          </CardBody>
        </Card>
      </div>

      {/* Incidents List */}
      <div className="space-y-4">
        {incidents.map((incident) => (
          <AlertCard key={incident.id} {...incident} />
        ))}
      </div>
    </div>
  );
}
