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
    <Card className="bg-white/10 backdrop-blur-md border border-white/20 hover:scale-105 transition-transform shadow-lg hover:bg-white/15">
      <CardBody className="p-5">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-lg font-bold text-white">{name}</h3>
            <p className="text-xs text-white/80 mt-1">{location}</p>
            <p className="text-xs text-white/60">ID: {id}</p>
          </div>
          <Chip
            className={`${isOnline ? 'bg-green-500' : 'bg-red-500'} text-white text-xs font-semibold shadow-lg`}
            size="sm"
            variant="solid"
          >
            {isOnline ? "Online" : "Offline"}
          </Chip>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-white/70">FPS</span>
            <span className="text-sm font-bold text-white">{fps}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-white/70">Resolution</span>
            <span className="text-sm font-bold text-white">{resolution}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-white/70">Detection Rate</span>
            <span className="text-sm font-bold text-white">{detectionRate}%</span>
          </div>
          {!isOnline && lastSeen && (
            <div className="flex justify-between items-center pt-2 border-t border-white/10">
              <span className="text-xs text-white/70">Last Seen</span>
              <span className="text-xs text-red-400">{lastSeen}</span>
            </div>
          )}
        </div>

        <div className="mt-4 pt-3 border-t border-white/10">
          <div className={`w-full h-2 bg-white/10 rounded-full overflow-hidden`}>
            <div
              className={`h-full ${isOnline ? 'bg-green-400 shadow-lg shadow-green-400/50' : 'bg-red-400'} transition-all`}
              style={{ width: `${isOnline ? 100 : 0}%` }}
            />
          </div>
          <p className="text-xs text-white/70 text-center mt-2">
            {isOnline ? "Operating Normally" : "Connection Lost"}
          </p>
        </div>
      </CardBody>
    </Card>
  );
};
