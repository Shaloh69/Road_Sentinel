import express, { Router, Request, Response } from "express";
import { EventEmitter } from "events";
import { query } from "../config/database";
import { logger } from "../config/logger";
import { Camera, ApiResponse } from "../types";
import { io } from "../server";

const router = Router();

// ── In-memory MJPEG frame buffer ──────────────────────────────────────────────
const frameBuffer  = new Map<string, Buffer>();
const frameEmitters = new Map<string, EventEmitter>();
const frameTimers  = new Map<string, ReturnType<typeof setTimeout>>();

const FRAME_TIMEOUT_MS = 15_000; // mark offline if no frame for 15s

function getEmitter(cameraId: string): EventEmitter {
  if (!frameEmitters.has(cameraId)) {
    const emitter = new EventEmitter();
    emitter.setMaxListeners(100);
    frameEmitters.set(cameraId, emitter);
  }
  return frameEmitters.get(cameraId)!;
}

async function setCameraOnline(cameraId: string, isFirst: boolean) {
  await query("UPDATE cameras SET status = 'online', updated_at = NOW() WHERE id = ?", [cameraId]);
  if (isFirst) {
    io.emit("camera_status", { type: "camera_status", data: { camera_id: cameraId, status: "online" } });
    logger.info(`🟢 Camera ${cameraId} — status changed to ONLINE`);
  }
}

async function setCameraOffline(cameraId: string) {
  await query("UPDATE cameras SET status = 'offline', updated_at = NOW() WHERE id = ?", [cameraId]);
  io.emit("camera_status", { type: "camera_status", data: { camera_id: cameraId, status: "offline" } });
  logger.warn(`🔴 Camera ${cameraId} — status changed to OFFLINE (no frames for ${FRAME_TIMEOUT_MS / 1000}s)`);
}

function resetFrameTimeout(cameraId: string) {
  const existing = frameTimers.get(cameraId);
  if (existing) clearTimeout(existing);
  frameTimers.set(
    cameraId,
    setTimeout(() => setCameraOffline(cameraId).catch(() => {}), FRAME_TIMEOUT_MS),
  );
}

// GET /api/cameras — list all cameras
router.get("/", async (req: Request, res: Response) => {
  try {
    const cameras = await query<Camera[]>(
      "SELECT * FROM cameras ORDER BY name",
    );
    res.json({ success: true, data: cameras } as ApiResponse<Camera[]>);
  } catch (err) {
    res.status(500).json({ success: false, error: "Failed to fetch cameras" });
  }
});

// GET /api/cameras/:id — single camera
router.get("/:id", async (req: Request, res: Response) => {
  try {
    const rows = await query<Camera[]>("SELECT * FROM cameras WHERE id = ?", [
      req.params.id,
    ]);
    if (rows.length === 0) {
      return res
        .status(404)
        .json({ success: false, error: "Camera not found" });
    }
    res.json({ success: true, data: rows[0] } as ApiResponse<Camera>);
  } catch (err) {
    res.status(500).json({ success: false, error: "Failed to fetch camera" });
  }
});

// PUT /api/cameras/:id — full settings update (name, location, rtsp_url, limits, calibration)
router.put("/:id", async (req: Request, res: Response) => {
  const {
    name,
    location,
    rtsp_url,
    speed_limit,
    detection_confidence,
    pixels_per_meter,
  } = req.body;
  try {
    await query(
      `UPDATE cameras SET
        name                 = COALESCE(?, name),
        location             = COALESCE(?, location),
        rtsp_url             = COALESCE(?, rtsp_url),
        speed_limit          = COALESCE(?, speed_limit),
        detection_confidence = COALESCE(?, detection_confidence),
        pixels_per_meter     = COALESCE(?, pixels_per_meter),
        updated_at           = NOW()
       WHERE id = ?`,
      [
        name ?? null,
        location ?? null,
        rtsp_url ?? null,
        speed_limit ?? null,
        detection_confidence ?? null,
        pixels_per_meter ?? null,
        req.params.id,
      ],
    );
    res.json({ success: true, message: `Camera ${req.params.id} updated` });
  } catch (err) {
    res.status(500).json({ success: false, error: "Failed to update camera" });
  }
});

// PUT /api/cameras/:id/status — update camera online/offline/error status
router.put("/:id/status", async (req: Request, res: Response) => {
  const { status } = req.body as { status: Camera["status"] };
  const allowed: Camera["status"][] = ["online", "offline", "error"];
  if (!allowed.includes(status)) {
    return res
      .status(400)
      .json({ success: false, error: "Invalid status value" });
  }
  try {
    await query(
      "UPDATE cameras SET status = ?, updated_at = NOW() WHERE id = ?",
      [status, req.params.id],
    );
    io.emit("camera_status", {
      type: "camera_status",
      data: { camera_id: req.params.id, status },
    });
    const emoji =
      status === "online" ? "🟢" : status === "offline" ? "🔴" : "🟡";
    logger.info(
      `${emoji} Camera ${req.params.id} — status changed to ${status.toUpperCase()}`,
    );
    res.json({
      success: true,
      message: `Camera ${req.params.id} status set to ${status}`,
    });
  } catch (err) {
    res
      .status(500)
      .json({ success: false, error: "Failed to update camera status" });
  }
});

// PUT /api/cameras/:id/frame — receive raw JPEG from camera_sender (best-effort)
router.put(
  "/:id/frame",
  express.raw({
    type: ["image/jpeg", "application/octet-stream"],
    limit: "2mb",
  }),
  (req: Request, res: Response) => {
    const jpeg = req.body as Buffer;
    if (!Buffer.isBuffer(jpeg) || jpeg.length === 0) {
      return res
        .status(400)
        .json({ success: false, error: "Expected raw JPEG body" });
    }
    const { id } = req.params;
    const isFirst = !frameBuffer.has(id);
    frameBuffer.set(id, jpeg);
    getEmitter(id).emit("frame", jpeg);
    // Broadcast binary frame to WebSocket stream subscribers (low-latency path)
    io.to(`stream:${id}`).emit(`camera_frame:${id}`, jpeg);
    if (isFirst)
      logger.info(`📹 Camera ${id} — first frame received (MJPEG + WS stream live)`);
    setCameraOnline(id, isFirst).catch(() => {});
    resetFrameTimeout(id);
    res.status(204).end();
  },
);

// GET /api/cameras/:id/stream — MJPEG multipart stream
router.get("/:id/stream", (req: Request, res: Response) => {
  const { id } = req.params;
  const boundary = "rsframe";
  logger.info(`📺 Camera ${id} — new viewer connected to MJPEG stream`);

  res.writeHead(200, {
    "Content-Type": `multipart/x-mixed-replace; boundary=${boundary}`,
    "Cache-Control": "no-cache, no-store",
    Connection: "keep-alive",
    Pragma: "no-cache",
    "Access-Control-Allow-Origin": "*",
  });

  const sendFrame = (jpeg: Buffer) => {
    try {
      res.write(
        `--${boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\n\r\n`,
      );
      res.write(jpeg);
      res.write("\r\n");
    } catch {
      // client disconnected — cleanup happens in 'close' handler
    }
  };

  // Immediately send the latest buffered frame if available
  const current = frameBuffer.get(id);
  if (current) sendFrame(current);

  const emitter = getEmitter(id);
  emitter.on("frame", sendFrame);

  req.on("close", () => {
    emitter.off("frame", sendFrame);
    logger.info(`📺 Camera ${id} — viewer disconnected from MJPEG stream`);
  });
});

export default router;
