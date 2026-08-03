import { Router, Request, Response } from "express";
import { query } from "../config/database";
import { logger } from "../config/logger";
import { Recording, ApiResponse } from "../types";

const router = Router();

// GET /api/recordings?camera_id=&date=YYYY-MM-DD&limit=
router.get("/", async (req: Request, res: Response) => {
  try {
    const {
      camera_id,
      date,
      limit = "50",
    } = req.query as Record<string, string>;

    let sql = "SELECT * FROM recordings WHERE 1=1";
    const params: (string | number)[] = [];

    if (camera_id) {
      sql += " AND camera_id = ?";
      params.push(camera_id);
    }
    if (date) {
      sql += " AND DATE(start_time) = ?";
      params.push(date);
    }

    const limitNum = Math.min(parseInt(limit, 10) || 50, 500);
    sql += ` ORDER BY start_time DESC LIMIT ${limitNum}`;

    const rows = await query<Recording[]>(sql, params);
    res.json({ success: true, data: rows } as ApiResponse<Recording[]>);
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch recordings" });
  }
});

// GET /api/recordings/:id
router.get("/:id", async (req: Request, res: Response) => {
  try {
    const rows = await query<Recording[]>(
      "SELECT * FROM recordings WHERE id = ?",
      [req.params.id],
    );
    if (rows.length === 0) {
      return res
        .status(404)
        .json({ success: false, error: "Recording not found" });
    }
    res.json({ success: true, data: rows[0] } as ApiResponse<Recording>);
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch recording" });
  }
});

// POST /api/recordings — called by camera_sender.py once a segment is uploaded
router.post("/", async (req: Request, res: Response) => {
  try {
    const r: Recording = req.body;

    const sql = `
      INSERT INTO recordings
        (id, camera_id, start_time, end_time, duration_seconds, video_url,
         thumbnail_url, file_size_mb, format, resolution, fps, status,
         error_message, vehicle_count, incident_count)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    await query(sql, [
      r.id,
      r.camera_id,
      r.start_time,
      r.end_time ?? null,
      r.duration_seconds ?? null,
      r.video_url ?? null,
      r.thumbnail_url ?? null,
      r.file_size_mb ?? null,
      r.format || "mp4",
      r.resolution ?? null,
      r.fps ?? null,
      r.status || "completed",
      r.error_message ?? null,
      r.vehicle_count ?? 0,
      r.incident_count ?? 0,
    ]);

    logger.info(
      `🎬 Recording registered [${r.camera_id}] ${r.id} — ${r.duration_seconds ?? "?"}s, ` +
        `${r.vehicle_count ?? 0} vehicle frames, ${r.incident_count ?? 0} incidents`,
    );

    res.status(201).json({ success: true, data: r } as ApiResponse<Recording>);
  } catch (err) {
    logger.error("Failed to save recording:", err);
    res.status(500).json({ success: false, error: "Failed to save recording" });
  }
});

export default router;
