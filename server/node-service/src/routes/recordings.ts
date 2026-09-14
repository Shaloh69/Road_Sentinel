import { Router, Request, Response } from "express";
import { query } from "../config/database";
import { logger } from "../config/logger";
import { Recording, ApiResponse } from "../types";
import { requireAuth } from "../middleware/auth";

const router = Router();

// ── On-demand clip requests ──────────────────────────────────────────────────
//
// The admin presses "clip the next 15/30 min" and this holds the request until
// the camera's own sender claims it. The sender already has the decoded RTSP
// stream in hand, so it does the recording itself (a second RTSP connection
// just to record would double the load on the camera and risk its connection
// limit) — this is only the hand-off point.
//
// In-memory and self-expiring: a request the sender never claims (its Pi is
// offline) must not sit around and fire hours later when the Pi returns. One
// pending request per camera — pressing again replaces the last.
const CLIP_REQUEST_TTL_MS = 2 * 60_000; // unclaimed requests lapse after 2 min
const ALLOWED_CLIP_MINUTES = new Set([15, 30]);

interface ClipRequest {
  id: string;
  camera_id: string;
  minutes: number;
  requested_at: number;
}

const pendingClips = new Map<string, ClipRequest>();

function freshPending(cameraId: string): ClipRequest | null {
  const req = pendingClips.get(cameraId);
  if (!req) return null;
  if (Date.now() - req.requested_at > CLIP_REQUEST_TTL_MS) {
    pendingClips.delete(cameraId);
    return null;
  }
  return req;
}

// POST /api/recordings/request — admin asks a camera to clip the next N minutes.
router.post("/request", requireAuth, (req: Request, res: Response) => {
  const camera_id = String(req.body?.camera_id ?? "").trim();
  const minutes = Number(req.body?.minutes);

  if (!camera_id) {
    return res
      .status(400)
      .json({ success: false, error: "camera_id is required" });
  }
  if (!ALLOWED_CLIP_MINUTES.has(minutes)) {
    return res.status(400).json({
      success: false,
      error: `minutes must be one of ${[...ALLOWED_CLIP_MINUTES].join(", ")}`,
    });
  }

  const request: ClipRequest = {
    id: `clip_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    camera_id,
    minutes,
    requested_at: Date.now(),
  };
  pendingClips.set(camera_id, request);
  logger.info(
    `🎬 Clip requested [${camera_id}] ${minutes} min (${request.id})`,
  );

  res.status(201).json({ success: true, data: request });
});

// GET /api/recordings/pending — admin UI: what is queued but not yet claimed.
router.get("/pending", requireAuth, (_req: Request, res: Response) => {
  const pending = [...pendingClips.keys()]
    .map((cam) => freshPending(cam))
    .filter((r): r is ClipRequest => r !== null);
  res.json({ success: true, data: { pending } });
});

// GET /api/recordings/pending/:cameraId — the camera_sender claims its request.
//
// Claim-on-read: returning it also clears it, so the same clip cannot start
// twice. There is exactly one sender per camera, so no two callers race for it.
// Unauthenticated to match the sender's other Node calls (it posts detections
// and recordings without a token); the request it claims was itself created by
// an authenticated admin, so nothing unprivileged is exposed here.
router.get("/pending/:cameraId", (req: Request, res: Response) => {
  const req0 = freshPending(req.params.cameraId);
  if (req0) pendingClips.delete(req.params.cameraId);
  res.json({ success: true, data: { pending: req0 ?? null } });
});

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
