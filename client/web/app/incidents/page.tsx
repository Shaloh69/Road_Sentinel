import { AlertCard } from "@/components/alert-card";
import { Card, CardBody } from "@heroui/card";
import { Select, SelectItem } from "@heroui/select";
import { Input } from "@heroui/input";
import { Button } from "@heroui/button";
import { Chip } from "@heroui/chip";

export default function IncidentsPage() {
  // Mock data - replace with real API data
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
    {
      id: "INC-003",
      type: "speeding" as const,
      severity: "medium" as const,
      title: "Speed Violation",
      description: "Vehicle exceeded speed limit by 18 km/h",
      timestamp: "1 hour ago",
      cameraId: "Camera A",
      details: {
        speed: "78 km/h",
        limit: "60 km/h",
        vehicle: "Car",
      },
    },
    {
      id: "INC-004",
      type: "warning" as const,
      severity: "medium" as const,
      title: "Traffic Congestion",
      description: "Heavy traffic detected - 12 vehicles simultaneously",
      timestamp: "2 hours ago",
      cameraId: "Camera B",
      details: {
        vehicles: "12",
        duration: "15 min",
      },
    },
    {
      id: "INC-005",
      type: "anomaly" as const,
      severity: "low" as const,
      title: "Unusual Activity",
      description: "Stopped vehicle detected for extended period",
      timestamp: "3 hours ago",
      cameraId: "Camera A",
      details: {
        duration: "8 min",
        vehicle: "Bus",
      },
    },
  ];

  const stats = {
    total: incidents.length,
    critical: incidents.filter(i => i.severity === "critical").length,
    resolved: 12,
    active: incidents.length - 12,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#1B1931] via-[#44174E] to-[#862249] p-6">
      {/* Header */}
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
              <div className="bg-[#ED9E59] bg-opacity-20 p-3 rounded-lg">
                <svg className="w-6 h-6 text-[#ED9E59]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card className="bg-gradient-to-br from-red-900 to-red-700 border-none">
          <CardBody className="p-4">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-sm text-white font-medium uppercase">Critical</p>
                <p className="text-3xl font-bold text-white mt-1">{stats.critical}</p>
              </div>
              <div className="bg-white bg-opacity-20 p-3 rounded-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card className="bg-gradient-to-br from-green-900 to-green-700 border-none">
          <CardBody className="p-4">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-sm text-white font-medium uppercase">Resolved</p>
                <p className="text-3xl font-bold text-white mt-1">{stats.resolved}</p>
              </div>
              <div className="bg-white bg-opacity-20 p-3 rounded-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card className="bg-gradient-to-br from-yellow-900 to-yellow-700 border-none">
          <CardBody className="p-4">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-sm text-white font-medium uppercase">Active</p>
                <p className="text-3xl font-bold text-white mt-1">{stats.active}</p>
              </div>
              <div className="bg-white bg-opacity-20 p-3 rounded-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </CardBody>
        </Card>
      </div>

      {/* Filters */}
      <Card className="bg-[#1B1931] border-2 border-[#44174E] mb-6">
        <CardBody className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Select
              label="Severity"
              placeholder="All severities"
              classNames={{
                label: "text-[#E8BCB8]",
                trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
              }}
            >
              <SelectItem key="all">All</SelectItem>
              <SelectItem key="critical">Critical</SelectItem>
              <SelectItem key="high">High</SelectItem>
              <SelectItem key="medium">Medium</SelectItem>
              <SelectItem key="low">Low</SelectItem>
            </Select>

            <Select
              label="Type"
              placeholder="All types"
              classNames={{
                label: "text-[#E8BCB8]",
                trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
              }}
            >
              <SelectItem key="all">All</SelectItem>
              <SelectItem key="crash">Crash</SelectItem>
              <SelectItem key="speeding">Speeding</SelectItem>
              <SelectItem key="anomaly">Anomaly</SelectItem>
              <SelectItem key="warning">Warning</SelectItem>
            </Select>

            <Select
              label="Camera"
              placeholder="All cameras"
              classNames={{
                label: "text-[#E8BCB8]",
                trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
              }}
            >
              <SelectItem key="all">All Cameras</SelectItem>
              <SelectItem key="cam-a">Camera A</SelectItem>
              <SelectItem key="cam-b">Camera B</SelectItem>
            </Select>

            <Input
              label="Search"
              placeholder="Search incidents..."
              classNames={{
                label: "text-[#E8BCB8]",
                input: "text-[#E8BCB8]",
                inputWrapper: "bg-[#44174E] border-[#862249]",
              }}
            />
          </div>
        </CardBody>
      </Card>

      {/* Incidents List */}
      <div className="space-y-4">
        {incidents.map((incident) => (
          <AlertCard key={incident.id} {...incident} />
        ))}
      </div>

      {/* Load More */}
      <div className="flex justify-center mt-6">
        <Button className="bg-[#862249] text-[#E8BCB8] font-semibold hover:bg-[#A34054] px-8">
          Load More Incidents
        </Button>
      </div>
    </div>
  );
}
