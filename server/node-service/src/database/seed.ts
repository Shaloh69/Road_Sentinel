import { query } from "../config/database";
import { logger } from "../config/logger";

interface CameraRow {
  id: string;
  name: string;
  location: string;
  rtsp_url: string;
  fps: number;
  resolution: string;
  pixels_per_meter: number;
  speed_limit: number;
  detection_confidence: number;
}

const DEFAULT_CAMERAS: CameraRow[] = [
  {
    id: "CAM-A-001",
    name: "Camera A",
    location: "Busay Blind Curve — Approach",
    rtsp_url:
      process.env.CAM_A_RTSP ?? "rtsp://192.168.8.104:554/cam/realmonitor",
    fps: 15,
    resolution: "640x480",
    pixels_per_meter: 8.0,
    speed_limit: 40.0,
    detection_confidence: 0.5,
  },
  {
    id: "CAM-B-002",
    name: "Camera B",
    location: "Busay Blind Curve — Exit",
    rtsp_url:
      process.env.CAM_B_RTSP ?? "rtsp://192.168.8.102:554/cam/realmonitor",
    fps: 15,
    resolution: "640x480",
    pixels_per_meter: 8.0,
    speed_limit: 40.0,
    detection_confidence: 0.5,
  },
];

export async function seedCameras(): Promise<void> {
  // Remove legacy IDs if they exist with no data
  for (const oldId of ["cam_a", "cam_b"]) {
    try {
      await query("DELETE FROM cameras WHERE id = ?", [oldId]);
    } catch {
      // ignore — may have FK constraints if data exists
    }
  }

  for (const cam of DEFAULT_CAMERAS) {
    await query(
      `INSERT INTO cameras
         (id, name, location, rtsp_url, fps, resolution, pixels_per_meter, speed_limit, detection_confidence)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON DUPLICATE KEY UPDATE updated_at = updated_at`,
      [
        cam.id,
        cam.name,
        cam.location,
        cam.rtsp_url,
        cam.fps,
        cam.resolution,
        cam.pixels_per_meter,
        cam.speed_limit,
        cam.detection_confidence,
      ],
    );
  }
  logger.info("Camera seed complete (CAM-A-001, CAM-B-002)");
}
