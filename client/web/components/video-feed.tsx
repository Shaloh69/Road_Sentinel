"use client";

import { Card, CardBody, CardHeader } from "@heroui/card";
import { Chip } from "@heroui/chip";
import { useState } from "react";

interface BoundingBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
  speed?: number;
}

interface VideoFeedProps {
  cameraId: string;
  cameraName: string;
  isLive?: boolean;
  showBoundingBoxes?: boolean;
  boundingBoxes?: BoundingBox[];
  videoUrl?: string;
  fps?: number;
  latency?: number;
}

export const VideoFeed = ({
  cameraId,
  cameraName,
  isLive = true,
  showBoundingBoxes = true,
  boundingBoxes = [],
  videoUrl,
  fps = 30,
  latency = 45,
}: VideoFeedProps) => {
  const [isFullscreen, setIsFullscreen] = useState(false);

  return (
    <Card className="bg-[#1B1931] border-2 border-[#44174E] shadow-xl">
      <CardHeader className="flex justify-between items-center bg-gradient-to-r from-[#44174E] to-[#862249] px-4 py-3">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}></div>
          <div>
            <h3 className="text-lg font-bold text-[#ED9E59]">{cameraName}</h3>
            <p className="text-xs text-[#E8BCB8]">ID: {cameraId}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isLive && (
            <Chip
              className="bg-red-500 text-white text-xs font-semibold"
              size="sm"
              variant="solid"
            >
              LIVE
            </Chip>
          )}
          <div className="flex flex-col text-right">
            <span className="text-xs text-[#E8BCB8]">{fps} FPS</span>
            <span className="text-xs text-[#E8BCB8]">{latency}ms</span>
          </div>
        </div>
      </CardHeader>
      <CardBody className="p-0">
        <div className="relative aspect-video bg-[#1B1931] overflow-hidden">
          {/* Video placeholder - replace with actual video stream */}
          <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-[#1B1931] via-[#44174E] to-[#862249]">
            <div className="text-center">
              <svg
                className="w-20 h-20 text-[#ED9E59] mx-auto mb-4 opacity-50"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <p className="text-[#E8BCB8] text-sm">
                {isLive ? "Camera Feed" : "No Signal"}
              </p>
            </div>
          </div>

          {/* Bounding boxes overlay */}
          {showBoundingBoxes && boundingBoxes.map((box) => (
            <div
              key={box.id}
              className="absolute border-2 border-[#ED9E59] rounded"
              style={{
                left: `${box.x}%`,
                top: `${box.y}%`,
                width: `${box.width}%`,
                height: `${box.height}%`,
              }}
            >
              <div className="absolute -top-6 left-0 bg-[#ED9E59] text-[#1B1931] px-2 py-0.5 rounded text-xs font-bold">
                {box.label} {(box.confidence * 100).toFixed(0)}%
                {box.speed && ` • ${box.speed} km/h`}
              </div>
            </div>
          ))}

          {/* Detection count overlay */}
          {showBoundingBoxes && boundingBoxes.length > 0 && (
            <div className="absolute top-4 left-4 bg-[#1B1931] bg-opacity-80 px-3 py-2 rounded-lg border border-[#ED9E59]">
              <p className="text-[#ED9E59] text-sm font-bold">
                {boundingBoxes.length} Vehicle{boundingBoxes.length !== 1 ? 's' : ''} Detected
              </p>
            </div>
          )}

          {/* Fullscreen toggle */}
          <button
            className="absolute bottom-4 right-4 bg-[#1B1931] bg-opacity-80 p-2 rounded-lg hover:bg-opacity-100 transition-all border border-[#44174E] hover:border-[#ED9E59]"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            <svg
              className="w-5 h-5 text-[#E8BCB8]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
              />
            </svg>
          </button>
        </div>
      </CardBody>
    </Card>
  );
};
