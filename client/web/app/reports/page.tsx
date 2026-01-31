import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Select, SelectItem } from "@heroui/select";
import { Input } from "@heroui/input";
import { Checkbox } from "@heroui/checkbox";

export default function ReportsPage() {
  const templates = [
    { id: "1", name: "Daily Traffic Summary", description: "Overview of daily traffic patterns" },
    { id: "2", name: "Weekly Incident Report", description: "All incidents from the past week" },
    { id: "3", name: "Speed Violation Report", description: "Detailed speed violation analysis" },
    { id: "4", name: "Vehicle Count Report", description: "Vehicle counts by type and time" },
  ];

  const recentReports = [
    { id: "1", name: "Daily Summary - Jan 31", date: "2024-01-31", type: "PDF", size: "2.4 MB" },
    { id: "2", name: "Weekly Incidents - Week 4", date: "2024-01-30", type: "CSV", size: "1.2 MB" },
    { id: "3", name: "Speed Violations - January", date: "2024-01-29", type: "PDF", size: "3.1 MB" },
  ];

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-[#ED9E59] mb-2">Reports</h1>
        <p className="text-[#E8BCB8]">Generate and download traffic analysis reports</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Report Generator */}
        <Card className="bg-[#1B1931] border-2 border-[#44174E]">
          <CardHeader className="bg-gradient-to-r from-[#44174E] to-[#862249] px-4 py-3">
            <h3 className="text-xl font-bold text-[#ED9E59]">Generate New Report</h3>
          </CardHeader>
          <CardBody className="p-6 space-y-4">
            <Select
              label="Report Template"
              placeholder="Select a template"
              classNames={{
                label: "text-[#E8BCB8]",
                trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
              }}
            >
              {templates.map((template) => (
                <SelectItem key={template.id} >
                  {template.name}
                </SelectItem>
              ))}
            </Select>

            <Input
              type="date"
              label="Start Date"
              classNames={{
                label: "text-[#E8BCB8]",
                input: "text-[#E8BCB8]",
                inputWrapper: "bg-[#44174E] border-[#862249]",
              }}
            />

            <Input
              type="date"
              label="End Date"
              classNames={{
                label: "text-[#E8BCB8]",
                input: "text-[#E8BCB8]",
                inputWrapper: "bg-[#44174E] border-[#862249]",
              }}
            />

            <div className="space-y-2">
              <p className="text-[#E8BCB8] text-sm font-medium">Include Data From:</p>
              <Checkbox defaultSelected classNames={{ label: "text-[#E8BCB8]" }}>Camera A</Checkbox>
              <Checkbox defaultSelected classNames={{ label: "text-[#E8BCB8]" }}>Camera B</Checkbox>
            </div>

            <Select
              label="Export Format"
              placeholder="Select format"
              defaultSelectedKeys={["pdf"]}
              classNames={{
                label: "text-[#E8BCB8]",
                trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
              }}
            >
              <SelectItem key="pdf">PDF</SelectItem>
              <SelectItem key="csv">CSV</SelectItem>
              <SelectItem key="xlsx">Excel (XLSX)</SelectItem>
            </Select>

            <Button className="w-full bg-[#ED9E59] text-[#1B1931] font-semibold hover:bg-[#A34054]">
              Generate Report
            </Button>
          </CardBody>
        </Card>

        {/* Report Templates */}
        <Card className="bg-[#1B1931] border-2 border-[#44174E]">
          <CardHeader className="bg-gradient-to-r from-[#44174E] to-[#862249] px-4 py-3">
            <h3 className="text-xl font-bold text-[#ED9E59]">Report Templates</h3>
          </CardHeader>
          <CardBody className="p-4">
            <div className="space-y-3">
              {templates.map((template) => (
                <div key={template.id} className="p-4 bg-[#44174E] rounded-lg hover:bg-[#862249] transition-colors">
                  <h4 className="text-[#ED9E59] font-semibold mb-1">{template.name}</h4>
                  <p className="text-[#E8BCB8] text-sm mb-3">{template.description}</p>
                  <Button size="sm" className="bg-[#ED9E59] text-[#1B1931] font-semibold">
                    Use Template
                  </Button>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      </div>

      {/* Recent Reports */}
      <Card className="bg-[#1B1931] border-2 border-[#44174E] mt-6">
        <CardHeader className="bg-gradient-to-r from-[#44174E] to-[#862249] px-4 py-3">
          <h3 className="text-xl font-bold text-[#ED9E59]">Recent Reports</h3>
        </CardHeader>
        <CardBody className="p-4">
          <div className="space-y-3">
            {recentReports.map((report) => (
              <div key={report.id} className="flex items-center justify-between p-4 bg-[#44174E] rounded-lg">
                <div className="flex items-center gap-4">
                  <div className="bg-[#ED9E59] bg-opacity-20 p-3 rounded-lg">
                    <svg className="w-6 h-6 text-[#ED9E59]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-[#ED9E59] font-semibold">{report.name}</p>
                    <p className="text-[#E8BCB8] text-sm">{report.date} • {report.type} • {report.size}</p>
                  </div>
                </div>
                <Button size="sm" className="bg-[#ED9E59] text-[#1B1931] font-semibold">
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
