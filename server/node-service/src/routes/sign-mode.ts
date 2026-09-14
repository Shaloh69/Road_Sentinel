import { Router, Request, Response } from "express";
import { requireAuth } from "../middleware/auth";

const router = Router();

// ── Transient display-mode override (the sand attract sequence) ──────────────
//
// Held in memory rather than the database on purpose: it is momentary, it must
// not survive a restart, and it must never end up in the incident history that
// the dashboard and reports read from.
//
// The override always expires on its own. A sign that could be put into a
// non-status mode indefinitely by an HTTP call would be a safety problem, so
// the timeout is the mechanism rather than a courtesy — nothing has to remember
// to clear it.
const ATTRACT_DURATION_MS = 45_000;

let attractUntil = 0;

export function activeSignMode(): string | null {
  return Date.now() < attractUntil ? "sand" : null;
}

// ── Per-sign disable (admin "turn this LED off for now") ─────────────────────
//
// An operator can blank a physical sign without touching its camera: the Pi's
// camera_sender keeps detecting and the dashboard keeps updating; only the LED
// bridge, which reads this flag out of /api/public/status, stops drawing and
// clears the panel. Used when a sign needs to go dark for maintenance, at a
// resident's request, or while something is being worked on beneath it.
//
// Deliberately in-memory, NOT persisted. A blanked SAFETY sign is not a state
// that should quietly survive a server restart — the safe default on reboot is
// every sign back in service. If it turns out signs need to stay disabled
// across restarts, that becomes a conscious DB-backed change, not an accident.
const disabledSigns = new Set<string>();

export function isSignDisabled(cameraId: string): boolean {
  return disabledSigns.has(cameraId);
}

router.post("/attract", (_req: Request, res: Response) => {
  attractUntil = Date.now() + ATTRACT_DURATION_MS;
  res.json({
    success: true,
    data: { mode: "sand", expires_in_ms: ATTRACT_DURATION_MS },
  });
});

router.post("/attract/cancel", (_req: Request, res: Response) => {
  attractUntil = 0;
  res.json({ success: true, data: { mode: null } });
});

// GET /api/sign/disabled — which signs are currently blanked (admin UI state).
router.get("/disabled", requireAuth, (_req: Request, res: Response) => {
  res.json({ success: true, data: { disabled: [...disabledSigns] } });
});

// POST /api/sign/:cameraId/disable — blank this sign. Camera keeps running.
router.post(
  "/:cameraId/disable",
  requireAuth,
  (req: Request, res: Response) => {
    const { cameraId } = req.params;
    disabledSigns.add(cameraId);
    res.json({ success: true, data: { camera_id: cameraId, disabled: true } });
  },
);

// POST /api/sign/:cameraId/enable — return this sign to normal status duty.
router.post("/:cameraId/enable", requireAuth, (req: Request, res: Response) => {
  const { cameraId } = req.params;
  disabledSigns.delete(cameraId);
  res.json({ success: true, data: { camera_id: cameraId, disabled: false } });
});

export default router;
