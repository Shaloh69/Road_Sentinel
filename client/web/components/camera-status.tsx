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
    <Card className="bg-gradient-to-br from-[#44174E] to-[#862249] hover:scale-105 transition-transform shadow-lg">
      <CardBody className="p-5">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-lg font-bold text-[#ED9E59]">{name}</h3>
            <p className="text-xs text-[#E8BCB8] mt-1">{location}</p>
            <p className="text-xs text-[#E8BCB8] opacity-70">ID: {id}</p>
          </div>
          <Chip
            className={`${isOnline ? 'bg-green-500' : 'bg-red-500'} text-white text-xs font-semibold`}
            size="sm"
            variant="solid"
          >
            {isOnline ? "Online" : "Offline"}
          </Chip>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[#E8BCB8]">FPS</span>
            <span className="text-sm font-bold text-[#ED9E59]">{fps}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[#E8BCB8]">Resolution</span>
            <span className="text-sm font-bold text-[#ED9E59]">{resolution}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[#E8BCB8]">Detection Rate</span>
            <span className="text-sm font-bold text-[#ED9E59]">{detectionRate}%</span>
          </div>
          {!isOnline && lastSeen && (
            <div className="flex justify-between items-center pt-2 border-t border-[#A34054]">
              <span className="text-xs text-[#E8BCB8]">Last Seen</span>
              <span className="text-xs text-red-400">{lastSeen}</span>
            </div>
          )}
        </div>

        <div className="mt-4 pt-3 border-t border-[#A34054]">
          <div className={`w-full h-2 bg-[#1B1931] rounded-full overflow-hidden`}>
            <div
              className={`h-full ${isOnline ? 'bg-green-400' : 'bg-red-400'} transition-all`}
              style={{ width: `${isOnline ? 100 : 0}%` }}
            />
          </div>
          <p className="text-xs text-[#E8BCB8] text-center mt-2">
            {isOnline ? "Operating Normally" : "Connection Lost"}
          </p>
        </div>
      </CardBody>
    </Card>
  );
};
