import { Router, Request, Response } from "express";
import { query } from "../config/database";

const router = Router();

// GET /api/public/status — minimal, unauthenticated road-status summary for
// the community "live status" page (Phase 2 new feature). Deliberately
// exposes only a safety-relevant state (safe / vehicle-incoming / incident),
// camera online counts, and a same-day vehicle/incident tally — no camera
// feeds, no admin surface, no RTSP URLs or other configuration. Mirrors the
// same VEHICLE_ALERT_SECS convention raspi_scripts/display_manager.py's
// SystemState uses, so the LED sign and this page agree on "current state."

const VEHICLE_ALERT_SECS = 8;

router.get("/", async (req: Request, res: Response) => {
  try {
    const [activeIncidentRows, recentDetectionRows, cameraRows, todayRows] =
      await Promise.all([
        query<
          {
            incident_type: string;
            severity: string;
            camera_id: string;
            timestamp: string;
          }[]
        >(
          `SELECT incident_type, severity, camera_id, timestamp FROM incidents
           WHERE status = 'active' ORDER BY timestamp DESC LIMIT 1`,
        ),
        query<{ camera_id: string; timestamp: string }[]>(
          `SELECT camera_id, timestamp FROM detections
           WHERE timestamp >= NOW() - INTERVAL ? SECOND
           ORDER BY timestamp DESC LIMIT 1`,
          [VEHICLE_ALERT_SECS],
        ),
        query<{ online: number; total: number }[]>(
          `SELECT SUM(status = 'online') AS online, COUNT(*) AS total FROM cameras`,
        ),
        query<{ vehicles: number; incidents: number }[]>(
          `SELECT
             (SELECT COUNT(*) FROM detections WHERE timestamp >= CURDATE()) AS vehicles,
             (SELECT COUNT(*) FROM incidents  WHERE DATE(timestamp) = CURDATE()) AS incidents`,
        ),
      ]);

    let state: "incident" | "vehicle_incoming" | "clear" = "clear";
    let detail: Record<string, unknown> = {};

    if (activeIncidentRows.length > 0) {
      state = "incident";
      const inc = activeIncidentRows[0];
      detail = {
        incident_type: inc.incident_type,
        severity: inc.severity,
        camera_id: inc.camera_id,
      };
    } else if (recentDetectionRows.length > 0) {
      state = "vehicle_incoming";
      detail = { camera_id: recentDetectionRows[0].camera_id };
    }

    res.json({
      success: true,
      data: {
        state, // "clear" | "vehicle_incoming" | "incident"
        detail,
        cameras_online: cameraRows[0]?.online ?? 0,
        cameras_total: cameraRows[0]?.total ?? 0,
        vehicles_today: todayRows[0]?.vehicles ?? 0,
        incidents_today: todayRows[0]?.incidents ?? 0,
        updated_at: new Date().toISOString(),
      },
    });
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch public status" });
  }
});

export default router;
