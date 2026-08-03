import { Card, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";

interface CameraStatusProps {
  id: string;
  name: string;
  location: string;
  isOnline: boolean;
  lastSeen?: string;
  fps?: number;
  resolution?: string;
  detectionRate?: number;
}

export const CameraStatus = ({
  id,
  name,
  location,
  isOnline,
  lastSeen,
  fps = 0,
  resolution = "1920x1080",
  detectionRate = 0,
}: CameraStatusProps) => {
  return (
    <Card className="bg-surface/80 backdrop-blur-md border border-border hover:scale-[1.02] transition-transform duration-150 ease-standard shadow-lg hover:bg-surface-2/80">
      <CardBody className="p-5">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-lg font-heading font-bold text-fg">{name}</h3>
            <p className="text-xs text-fg-muted mt-1">{location}</p>
            <p className="text-xs text-fg-muted/70 font-mono">ID: {id}</p>
          </div>
          <Chip
            color={isOnline ? "success" : "danger"}
            size="sm"
            variant="solid"
          >
            {isOnline ? "Online" : "Offline"}
          </Chip>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-fg-muted">FPS</span>
            <span className="text-sm font-bold text-fg font-mono">{fps}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-fg-muted">Resolution</span>
            <span className="text-sm font-bold text-fg font-mono">
              {resolution}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-fg-muted">Detection Rate</span>
            <span className="text-sm font-bold text-fg font-mono">
              {detectionRate}%
            </span>
          </div>
          {!isOnline && lastSeen && (
            <div className="flex justify-between items-center pt-2 border-t border-border">
              <span className="text-xs text-fg-muted">Last Seen</span>
              <span className="text-xs text-danger">{lastSeen}</span>
            </div>
          )}
        </div>

        <div className="mt-4 pt-3 border-t border-border">
          <div className="w-full h-2 bg-surface-2 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ease-standard ${isOnline ? "bg-success shadow-lg shadow-success/50" : "bg-danger"}`}
              style={{ width: `${isOnline ? 100 : 0}%` }}
            />
          </div>
          <p className="text-xs text-fg-muted text-center mt-2">
            {isOnline ? "Operating Normally" : "Connection Lost"}
          </p>
        </div>
      </CardBody>
    </Card>
  );
};
