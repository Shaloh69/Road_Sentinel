import { Router, Request, Response } from 'express';
import { query } from '../config/database';
import { HourlyAnalytics, ApiResponse } from '../types';

const router = Router();

// GET /api/analytics/summary — live summary stats for the dashboard
// Returns totals for today across all cameras
router.get('/summary', async (req: Request, res: Response) => {
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
        average_speed: traffic[0].avg_speed ? Math.round(traffic[0].avg_speed) : null,
        incidents_today: incidents[0].total,
        cameras_online: cameras[0].online,
        cameras_total: cameras[0].total,
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch summary' });
  }
});

// GET /api/analytics/hourly?camera_id=&date=YYYY-MM-DD
router.get('/hourly', async (req: Request, res: Response) => {
  try {
    const { camera_id, date } = req.query as Record<string, string>;

    let sql = `
      SELECT * FROM analytics_hourly
      WHERE DATE(hour_timestamp) = ?
    `;
    const params: string[] = [date || new Date().toISOString().slice(0, 10)];

    if (camera_id) {
      sql += ' AND camera_id = ?';
      params.push(camera_id);
    }

    sql += ' ORDER BY hour_timestamp ASC';

    const rows = await query<HourlyAnalytics[]>(sql, params);
    res.json({ success: true, data: rows } as ApiResponse<HourlyAnalytics[]>);
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch hourly analytics' });
  }
});

// GET /api/analytics/speed?camera_id=&hours=24
// Returns speed histogram buckets for charts
router.get('/speed', async (req: Request, res: Response) => {
  try {
    const { camera_id, hours = '24' } = req.query as Record<string, string>;

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
      sql += ' AND camera_id = ?';
      params.push(camera_id);
    }

    sql += ' GROUP BY speed_bucket ORDER BY speed_bucket ASC';

    const rows = await query<{ speed_bucket: number; count: number }[]>(sql, params);
    res.json({ success: true, data: rows });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch speed distribution' });
  }
});

export default router;
