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
router.get("/hourly", async (req: Request, res: Response) => {
  try {
    const { camera_id, date } = req.query as Record<string, string>;

    let sql = `
      SELECT * FROM hourly_analytics
      WHERE DATE(hour_timestamp) = ?
    `;
    const params: string[] = [date || new Date().toISOString().slice(0, 10)];

    if (camera_id) {
      sql += " AND camera_id = ?";
      params.push(camera_id);
    }

    sql += " ORDER BY hour_timestamp ASC";

    const rows = await query<HourlyAnalytics[]>(sql, params);
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
