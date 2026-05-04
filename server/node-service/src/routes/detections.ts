import { Router, Request, Response } from "express";
import { query } from "../config/database";
import { logger } from "../config/logger";
import { Detection, ApiResponse } from "../types";
import { io } from "../server";

const router = Router();

// GET /api/detections?camera_id=&limit=&since=
router.get("/", async (req: Request, res: Response) => {
  try {
    const {
      camera_id,
      limit = "100",
      since,
    } = req.query as Record<string, string>;

    let sql = "SELECT * FROM detections WHERE 1=1";
    const params: (string | number)[] = [];

    if (camera_id) {
      sql += " AND camera_id = ?";
      params.push(camera_id);
    }
    if (since) {
      sql += " AND timestamp >= ?";
      params.push(since);
    }

    sql += " ORDER BY timestamp DESC LIMIT ?";
    params.push(Math.min(parseInt(limit, 10) || 100, 1000));

    const rows = await query<Detection[]>(sql, params);
    res.json({ success: true, data: rows } as ApiResponse<Detection[]>);
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch detections" });
  }
});

// POST /api/detections — log a single detection and broadcast via Socket.IO
router.post("/", async (req: Request, res: Response) => {
  try {
    const d: Detection = req.body;

    const sql = `
      INSERT INTO detections
        (camera_id, vehicle_type, speed, confidence, bbox_x, bbox_y, bbox_width, bbox_height, direction, lane_number)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    const result = await query<{ insertId: number }>(sql, [
      d.camera_id,
      d.vehicle_type,
      d.speed ?? null,
      d.confidence,
      d.bbox_x,
      d.bbox_y,
      d.bbox_width,
      d.bbox_height,
      d.direction ?? null,
      d.lane_number ?? null,
    ]);

    const saved: Detection = {
      ...d,
      id: result.insertId,
      timestamp: new Date(),
    };

    // Broadcast to clients subscribed to this camera
    io.to(`camera:${d.camera_id}`).emit("detection", {
      type: "detection",
      data: saved,
    });

    const speedStr = d.speed != null ? ` @ ${Math.round(d.speed)} km/h` : "";
    logger.info(
      `🚗 Detection [${d.camera_id}] ${d.vehicle_type ?? "vehicle"}${speedStr}  conf=${((d.confidence ?? 0) * 100).toFixed(0)}%`,
    );

    res
      .status(201)
      .json({ success: true, data: saved } as ApiResponse<Detection>);
  } catch (err) {
    res.status(500).json({ success: false, error: "Failed to save detection" });
  }
});

export default router;
