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
  isHeuristic?: boolean;
}

// RAG-safe severity mapping: red is reserved for "critical" only, so it
// never gets diluted into ordinary chrome elsewhere in the app.
const severityColors: Record<AlertCardProps["severity"], string> = {
  low: "bg-info text-white",
  medium: "bg-brand text-brand-foreground",
  high: "bg-warning text-brand-foreground",
  critical: "bg-danger text-white",
};

const typeIcons = {
  speeding: (
    <svg
      className="w-5 h-5"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M13 10V3L4 14h7v7l9-11h-7z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  ),
  crash: (
    <svg
      className="w-5 h-5"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  ),
  anomaly: (
    <svg
      className="w-5 h-5"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  ),
  warning: (
    <svg
      className="w-5 h-5"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  ),
};

export const AlertCard = ({
  id: _id,
  type,
  severity,
  title,
  description,
  timestamp,
  cameraId,
  details,
  imageUrl: _imageUrl,
  isHeuristic = false,
}: AlertCardProps) => {
  return (
    <Card className="bg-surface/80 backdrop-blur-md border-l-4 border-l-border hover:border-l-brand transition-colors duration-150 ease-standard shadow-lg hover:bg-surface-2/80">
      <CardBody className="p-4">
        <div className="flex gap-4">
          {/* Alert Icon */}
          <div
            className={`${severityColors[severity]} p-3 rounded-lg h-fit shadow-lg`}
          >
            {typeIcons[type]}
          </div>

          {/* Alert Content */}
          <div className="flex-1">
            <div className="flex justify-between items-start mb-2">
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-lg font-heading font-bold text-fg">
                    {title}
                  </h3>
                  {isHeuristic && (
                    <Chip
                      className="bg-surface-2 text-fg-muted text-[10px] border border-border"
                      size="sm"
                      title="No trained crash/incident model is deployed yet — this came from a brightness-based heuristic placeholder, not a real detection."
                      variant="flat"
                    >
                      ESTIMATED — no model trained
                    </Chip>
                  )}
                </div>
                <p className="text-sm text-fg-muted mt-1">{description}</p>
              </div>
              <Chip
                className={`${severityColors[severity]} text-xs font-semibold uppercase`}
                size="sm"
                variant="solid"
              >
                {severity}
              </Chip>
            </div>

            {/* Details */}
            {details && (
              <div className="grid grid-cols-2 gap-2 mt-3 p-3 bg-surface-2/60 rounded-lg backdrop-blur-sm">
                {Object.entries(details).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-xs text-fg-muted capitalize">
                      {key}:
                    </span>
                    <span className="text-xs text-fg font-semibold font-mono">
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Footer */}
            <div className="flex justify-between items-center mt-3 pt-3 border-t border-border">
              <div className="flex items-center gap-2 text-xs text-fg-muted">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                  />
                </svg>
                <span>{cameraId}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-fg-muted font-mono">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                  />
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
