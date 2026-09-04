import { Router, Request, Response } from "express";
import { query } from "../config/database";
import { activeSignMode } from "./sign-mode";

const router = Router();

// GET /api/public/status — minimal, unauthenticated road-status summary for
// the community "live status" page (Phase 2 new feature). Deliberately
// exposes only a safety-relevant state (safe / vehicle-incoming / incident),
// camera online counts, and a same-day vehicle/incident tally — no camera
// feeds, no admin surface, no RTSP URLs or other configuration. Mirrors the
// same VEHICLE_ALERT_SECS convention raspi_scripts/led_sign_bridge.py's
// SystemState uses, so the LED sign and this page agree on "current state."

const VEHICLE_ALERT_SECS = 8;

// How long an incident keeps driving the physical signs and this page.
//
// WHY THIS EXISTS: the incident query used to filter on `status = 'active'`
// with no time bound at all, so ANY unresolved incident pinned the state to
// "incident" indefinitely. A speeding record from 2026-08-20 was still putting
// both roadside signs into flashing STOP two weeks later, with zero cameras
// online. For a safety sign that is worse than showing nothing: a warning that
// is always on is one drivers learn to ignore.
//
// Five minutes is long enough for an incident to be seen and acted on, short
// enough that a forgotten row cannot hold the sign hostage. Unresolved older
// incidents still appear in the dashboard and the incidents API — they just
// stop driving the sign.
const INCIDENT_ALERT_SECS = Number(process.env.INCIDENT_ALERT_SECS || 300);

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
           WHERE status = 'active'
             AND timestamp >= NOW() - INTERVAL ? SECOND
           ORDER BY timestamp DESC LIMIT 1`,
          [INCIDENT_ALERT_SECS],
        ),
        // VEHICLE INCOMING requires a recent detection on BOTH approaches,
        // not either. On a blind curve a single vehicle is the normal case and
        // warning on it would leave the sign lit almost continuously, which
        // costs the warning its meaning. Two vehicles converging from opposite
        // sides is the situation neither driver can see, and the one the sign
        // exists for.
        //
        // GROUP BY camera_id so this counts DISTINCT approaches — twenty
        // detections from one camera must not look like two cameras agreeing.
        query<{ camera_id: string; last_seen: string }[]>(
          `SELECT camera_id, MAX(timestamp) AS last_seen FROM detections
           WHERE timestamp >= NOW() - INTERVAL ? SECOND
           GROUP BY camera_id`,
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
    } else if (recentDetectionRows.length >= 2) {
      state = "vehicle_incoming";
      detail = {
        camera_ids: recentDetectionRows.map((r) => r.camera_id),
        approaches: recentDetectionRows.length,
      };
    }

    res.json({
      success: true,
      data: {
        state, // "clear" | "vehicle_incoming" | "incident"
        // Transient display-mode override. Null in normal operation.
        //
        // Takes precedence over the road state, by request: activating it
        // overwrites whatever the sign is currently showing.
        //
        // Two things keep that safe rather than merely brief. It self-expires
        // server-side, and the animation itself is one-shot — the firmware
        // returns the panel to status duty when the sequence ends, so the sign
        // resumes warning even if this endpoint were never called again.
        // Worth knowing: while it runs, a genuine incident will not be shown.
        sign_mode: activeSignMode(),
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
