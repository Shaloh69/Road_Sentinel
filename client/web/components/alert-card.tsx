import { Card, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";

interface AlertCardProps {
  id: string;
  type: "speeding" | "crash" | "anomaly" | "warning";
  severity: "low" | "medium" | "high" | "critical";
  title: string;
  description: string;
  timestamp: string;
  cameraId: string;
  details?: Record<string, string | number>;
  imageUrl?: string;
}

const severityColors = {
  low: "bg-blue-500",
  medium: "bg-yellow-500",
  high: "bg-orange-500",
  critical: "bg-red-500",
};

const severityTextColors = {
  low: "text-blue-400",
  medium: "text-yellow-400",
  high: "text-orange-400",
  critical: "text-red-400",
};

const typeIcons = {
  speeding: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  crash: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  anomaly: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
    </svg>
  ),
  warning: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
};

export const AlertCard = ({
  id,
  type,
  severity,
  title,
  description,
  timestamp,
  cameraId,
  details,
  imageUrl,
}: AlertCardProps) => {
  return (
    <Card className="bg-gradient-to-br from-[#1B1931] to-[#44174E] border-l-4 hover:border-l-[#ED9E59] transition-all shadow-lg">
      <CardBody className="p-4">
        <div className="flex gap-4">
          {/* Alert Icon */}
          <div className={`${severityColors[severity]} p-3 rounded-lg h-fit`}>
            <div className="text-white">
              {typeIcons[type]}
            </div>
          </div>

          {/* Alert Content */}
          <div className="flex-1">
            <div className="flex justify-between items-start mb-2">
              <div>
                <h3 className="text-lg font-bold text-[#ED9E59]">{title}</h3>
                <p className="text-sm text-[#E8BCB8] mt-1">{description}</p>
              </div>
              <Chip
                className={`${severityColors[severity]} text-white text-xs font-semibold uppercase`}
                size="sm"
                variant="solid"
              >
                {severity}
              </Chip>
            </div>

            {/* Details */}
            {details && (
              <div className="grid grid-cols-2 gap-2 mt-3 p-3 bg-[#1B1931] bg-opacity-50 rounded-lg">
                {Object.entries(details).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-xs text-[#E8BCB8] opacity-70 capitalize">{key}:</span>
                    <span className="text-xs text-[#ED9E59] font-semibold">{value}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Footer */}
            <div className="flex justify-between items-center mt-3 pt-3 border-t border-[#44174E]">
              <div className="flex items-center gap-2 text-xs text-[#E8BCB8]">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>{cameraId}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-[#E8BCB8]">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>{timestamp}</span>
              </div>
            </div>
          </div>
        </div>
      </CardBody>
    </Card>
  );
};
