import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";

export default function ReportsPage() {
  const recentReports = [
    { id: "1", name: "Daily Summary - Jan 31", date: "2024-01-31", type: "PDF", size: "2.4 MB" },
    { id: "2", name: "Weekly Incidents - Week 4", date: "2024-01-30", type: "CSV", size: "1.2 MB" },
    { id: "3", name: "Speed Violations - January", date: "2024-01-29", type: "PDF", size: "3.1 MB" },
  ];

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Reports</h1>
        <p className="text-white/70">Generate and download traffic analysis reports</p>
      </div>

      {/* Recent Reports */}
      <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
        <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
          <h3 className="text-xl font-bold text-white">Recent Reports</h3>
        </CardHeader>
        <CardBody className="p-4">
          <div className="space-y-3">
            {recentReports.map((report) => (
              <div key={report.id} className="flex items-center justify-between p-4 bg-white/10 backdrop-blur-sm rounded-lg border border-white/10">
                <div className="flex items-center gap-4">
                  <div className="bg-white/20 p-3 rounded-lg">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-white font-semibold">{report.name}</p>
                    <p className="text-white/70 text-sm">{report.date} • {report.type} • {report.size}</p>
                  </div>
                </div>
                <Button size="sm" className="bg-white text-[#1B1931] font-semibold hover:bg-white/90">
                  Download
                </Button>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
