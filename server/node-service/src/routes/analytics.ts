import { Router, Request, Response } from "express";
import { query } from "../config/database";
import { HourlyAnalytics, ApiResponse } from "../types";

const router = Router();

// GET /api/analytics/summary — live summary stats for the dashboard
// Returns totals for today across all cameras
router.get("/summary", async (req: Request, res: Response) => {
  try {
    const [traffic, incidents, cameras] = await Promise.all([
      query<{ total: number; avg_speed: number | null }[]>(`
        SELECT COUNT(*) AS total, AVG(speed) AS avg_speed
        FROM detections
        WHERE timestamp >= CURDATE()
      `),
      query<{ total: number }[]>(`
        SELECT COUNT(*) AS total
        FROM incidents
        WHERE DATE(timestamp) = CURDATE()
      `),
      query<{ online: number; total: number }[]>(`
        SELECT
          SUM(status = 'online') AS online,
          COUNT(*) AS total
        FROM cameras
      `),
    ]);

    res.json({
      success: true,
      data: {
        vehicles_today: traffic[0].total,
        average_speed: traffic[0].avg_speed
          ? Math.round(traffic[0].avg_speed)
          : null,
        incidents_today: incidents[0].total,
        cameras_online: cameras[0].online,
        cameras_total: cameras[0].total,
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: "Failed to fetch summary" });
  }
});

// GET /api/analytics/hourly?camera_id=&date=YYYY-MM-DD
//
// Aggregated live from `detections` and `incidents`. It used to read the
// `hourly_analytics` table, which migrate.ts creates but nothing ever writes,
// so this always returned [] — leaving the hourly chart, the vehicle-type
// cards (summed from these rows), the peak hour and the CSV exports empty
// while detections held 1,500+ rows for the day (found 2026-10-27).
//
// Rows are per hour across the selected cameras, in the HourlyAnalytics shape
// the clients already read. Timestamps are stored in server-local time, so
// HOUR() buckets are local hours. Counts are CAST because MySQL returns SUM()
// as DECIMAL, which mysql2 hands over as a string ("3"), and the client adds
// them up.
router.get("/hourly", async (req: Request, res: Response) => {
  try {
    const { camera_id, date } = req.query as Record<string, string>;
    const day = date || new Date().toLocaleDateString("en-CA");
    const camFilter = camera_id ? " AND camera_id = ?" : "";
    const params: string[] = camera_id ? [day, day, camera_id] : [day, day];
    const range =
      "timestamp >= ? AND timestamp < DATE_ADD(?, INTERVAL 1 DAY)" + camFilter;
    const hourOf =
      "TIMESTAMP(DATE(timestamp), MAKETIME(HOUR(timestamp), 0, 0))";

    const [det, inc] = await Promise.all([
      query<HourlyAnalytics[]>(
        `SELECT ${hourOf} AS hour_timestamp,
                COUNT(*) AS total_vehicles,
                AVG(speed) AS avg_speed,
                MAX(speed) AS max_speed,
                MIN(speed) AS min_speed,
                CAST(SUM(vehicle_type = 'car')        AS SIGNED) AS car_count,
                CAST(SUM(vehicle_type = 'truck')      AS SIGNED) AS truck_count,
                CAST(SUM(vehicle_type = 'bus')        AS SIGNED) AS bus_count,
                CAST(SUM(vehicle_type = 'motorcycle') AS SIGNED) AS motorcycle_count,
                CAST(SUM(vehicle_type = 'bicycle')    AS SIGNED) AS bicycle_count
           FROM detections
          WHERE ${range}
          GROUP BY hour_timestamp
          ORDER BY hour_timestamp ASC`,
        params,
      ),
      query<
        { hour_timestamp: Date; incident_count: number; speeding: number }[]
      >(
        `SELECT ${hourOf} AS hour_timestamp,
                COUNT(*) AS incident_count,
                CAST(SUM(incident_type = 'speeding') AS SIGNED) AS speeding
           FROM incidents
          WHERE ${range}
          GROUP BY hour_timestamp`,
        params,
      ),
    ]);

    const incByHour = new Map(
      inc.map((r) => [new Date(r.hour_timestamp).getTime(), r]),
    );
    const rows: HourlyAnalytics[] = det.map((r) => {
      const i = incByHour.get(new Date(r.hour_timestamp).getTime());

      return {
        ...r,
        camera_id: camera_id || "all",
        incident_count: i?.incident_count ?? 0,
        speeding_violations: i?.speeding ?? 0,
      };
    });

    res.json({ success: true, data: rows } as ApiResponse<HourlyAnalytics[]>);
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch hourly analytics" });
  }
});

// GET /api/analytics/speed?camera_id=&hours=24
// Returns speed histogram buckets for charts
router.get("/speed", async (req: Request, res: Response) => {
  try {
    const { camera_id, hours = "24" } = req.query as Record<string, string>;

    let sql = `
      SELECT
        FLOOR(speed / 10) * 10 AS speed_bucket,
        COUNT(*) AS count
      FROM detections
      WHERE speed IS NOT NULL
        AND timestamp >= NOW() - INTERVAL ? HOUR
    `;
    const params: (string | number)[] = [parseInt(hours)];

    if (camera_id) {
      sql += " AND camera_id = ?";
      params.push(camera_id);
    }

    sql += " GROUP BY speed_bucket ORDER BY speed_bucket ASC";

    const rows = await query<{ speed_bucket: number; count: number }[]>(
      sql,
      params,
    );
    res.json({ success: true, data: rows });
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch speed distribution" });
  }
});

// GET /api/analytics/violations?date=YYYY-MM-DD&camera_id=
// Speed violations by hour-of-day bucket (0-23) — built for thesis write-up
// figures (Phase 2 new feature): "when during the day do speed violations
// happen" is a more useful chart for a blind-curve safety thesis than a
// flat speed histogram alone.
router.get("/violations", async (req: Request, res: Response) => {
  try {
    const { date, camera_id } = req.query as Record<string, string>;
    const day = date || new Date().toISOString().slice(0, 10);

    let sql = `
      SELECT
        HOUR(d.timestamp) AS hour,
        COUNT(*) AS violations,
        ROUND(AVG(d.speed), 1) AS avg_speed,
        ROUND(MAX(d.speed), 1) AS max_speed,
        c.speed_limit AS speed_limit
      FROM detections d
      JOIN cameras c ON d.camera_id = c.id
      WHERE DATE(d.timestamp) = ?
        AND d.speed IS NOT NULL
        AND d.speed > c.speed_limit
    `;
    const params: (string | number)[] = [day];

    if (camera_id) {
      sql += " AND d.camera_id = ?";
      params.push(camera_id);
    }

    sql += " GROUP BY HOUR(d.timestamp), c.speed_limit ORDER BY hour ASC";

    const rows = await query<
      {
        hour: number;
        violations: number;
        avg_speed: number;
        max_speed: number;
        speed_limit: number;
      }[]
    >(sql, params);

    // Fill in all 24 hours (0 violations for hours with no rows) so charts
    // don't have to special-case missing buckets.
    const byHour = new Map(rows.map((r) => [r.hour, r]));
    const filled = Array.from({ length: 24 }, (_, hour) => {
      const r = byHour.get(hour);
      return {
        hour,
        violations: r?.violations ?? 0,
        avg_speed: r?.avg_speed ?? null,
        max_speed: r?.max_speed ?? null,
      };
    });

    res.json({ success: true, data: filled, date: day });
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch speed violations" });
  }
});

export default router;
