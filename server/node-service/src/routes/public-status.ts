import { Router, Request, Response } from "express";
import { query } from "../config/database";
import { activeSignMode } from "./sign-mode";

const router = Router();

// GET /api/public/status — road-status summary for the community "live status"
// page AND the per-approach state each physical sign follows.
//
// ── PHYSICAL LAYOUT, which drives everything below ────────────────────────
//
// Two cameras face OUTWARD from the blind curve, one on each approach, with an
// LED sign mounted directly beneath each camera. A sign therefore faces the
// traffic its own camera is watching.
//
//        approach A                 approach B
//     [cam A] --> ) ) )  curve  ( ( ( <-- [cam B]
//     [sign A]                          [sign B]
//
// That makes state PER-APPROACH, not global:
//
//   * An incident seen by camera A concerns the driver in front of sign A, so
//     only sign A reacts. Sign B has nothing to warn about and stays SAFE.
//     Lighting both would train drivers to ignore a sign that is often lit for
//     something happening where they cannot see it.
//
//   * A vehicle on BOTH approaches at once is the case neither driver can see
//     around the curve, and the reason this system exists. Both signs show
//     VEHICLE INCOMING together.
//
// `state` remains the overall figure for the public web page; `signs` carries
// the per-approach state the bridges follow.

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

// incidents.incident_type (an ENUM in migrate.ts) -> the wire word the sign
// firmware understands. Anything unmapped falls back to a generic STOP, which
// is the safe direction to fail: an unrecognised incident still warns.
const INCIDENT_SIGN_STATE: Record<string, string> = {
  crash: "crash", // CRASH / AHEAD      red
  stopped_vehicle: "stopped", // STOPPED / VEHICLE  red
  congestion: "congestion", // TRAFFIC / AHEAD    yellow
  speeding: "speeding", // SLOW / DOWN        yellow
  // wrong_way, illegal_parking, other -> "incident" (generic STOP)
};

router.get("/", async (_req: Request, res: Response) => {
  try {
    const [incidentRows, detectionRows, cameraRows, todayRows] =
      await Promise.all([
        // Most recent qualifying incident per approach.
        query<
          {
            camera_id: string;
            incident_type: string;
            severity: string;
            timestamp: string;
          }[]
        >(
          `SELECT i.camera_id, i.incident_type, i.severity, i.timestamp
             FROM incidents i
             JOIN (
               SELECT camera_id, MAX(timestamp) AS mt FROM incidents
                WHERE status = 'active'
                  AND timestamp >= NOW() - INTERVAL ? SECOND
                GROUP BY camera_id
             ) latest
               ON latest.camera_id = i.camera_id AND latest.mt = i.timestamp
            WHERE i.status = 'active'`,
          [INCIDENT_ALERT_SECS],
        ),
        // Grouped by camera so this counts DISTINCT approaches — twenty
        // detections from one camera must not look like two cameras agreeing.
        query<{ camera_id: string; last_seen: string }[]>(
          `SELECT camera_id, MAX(timestamp) AS last_seen FROM detections
           WHERE timestamp >= NOW() - INTERVAL ? SECOND
           GROUP BY camera_id`,
          [VEHICLE_ALERT_SECS],
        ),
        query<{ id: string; online: number }[]>(
          `SELECT id, (status = 'online') AS online FROM cameras`,
        ),
        query<{ vehicles: number; incidents: number }[]>(
          `SELECT
             (SELECT COUNT(*) FROM detections WHERE timestamp >= CURDATE()) AS vehicles,
             (SELECT COUNT(*) FROM incidents  WHERE DATE(timestamp) = CURDATE()) AS incidents`,
        ),
      ]);

    const camerasWithVehicles = new Set(detectionRows.map((d) => d.camera_id));
    // Both approaches occupied is what makes this a blind-curve conflict.
    const bothApproaches = camerasWithVehicles.size >= 2;

    const incidentByCamera = new Map(incidentRows.map((i) => [i.camera_id, i]));

    // ── Per-approach state ────────────────────────────────────────────────
    const signs: Record<string, Record<string, unknown>> = {};

    for (const cam of cameraRows) {
      const inc = incidentByCamera.get(cam.id);
      let signState = "clear";
      const signDetail: Record<string, unknown> = {};

      if (inc) {
        // An incident on THIS approach outranks the converging-vehicle case:
        // it is more specific, and more urgent to the driver in front of it.
        signState = INCIDENT_SIGN_STATE[inc.incident_type] ?? "incident";
        signDetail.incident_type = inc.incident_type;
        signDetail.severity = inc.severity;
      } else if (bothApproaches) {
        signState = "vehicle";
        signDetail.approaches = camerasWithVehicles.size;
      }

      signs[cam.id] = { state: signState, detail: signDetail };
    }

    // ── Overall state, for the public page ────────────────────────────────
    let state: "incident" | "vehicle_incoming" | "clear" = "clear";
    let detail: Record<string, unknown> = {};

    if (incidentRows.length > 0) {
      state = "incident";
      const first = incidentRows[0];

      detail = {
        incident_type: first.incident_type,
        severity: first.severity,
        camera_id: first.camera_id,
      };
    } else if (bothApproaches) {
      state = "vehicle_incoming";
      detail = { approaches: camerasWithVehicles.size };
    }

    res.json({
      success: true,
      data: {
        state, // "clear" | "vehicle_incoming" | "incident"
        detail,
        // Per-approach state, keyed by camera id. Each bridge reads its own
        // entry; a bridge with no camera id configured falls back to `state`.
        signs,
        // Transient display-mode override. Null in normal operation.
        //
        // Takes precedence over road state, by request: activating it
        // overwrites whatever the sign is showing. Two things keep that safe
        // rather than merely brief — it self-expires server-side, and the
        // animation is one-shot, so the firmware returns the panel to status
        // duty when the sequence ends. Worth knowing: while it runs, a genuine
        // incident will not be displayed.
        sign_mode: activeSignMode(),
        cameras_online: cameraRows.filter((c) => c.online).length,
        cameras_total: cameraRows.length,
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
