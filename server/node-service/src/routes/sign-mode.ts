import { Router, Request, Response } from "express";

const router = Router();

// Transient display-mode override for the physical signs.
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

router.post("/attract", (_req: Request, res: Response) => {
  attractUntil = Date.now() + ATTRACT_DURATION_MS;
  res.json({ success: true, data: { mode: "sand", expires_in_ms: ATTRACT_DURATION_MS } });
});

router.post("/attract/cancel", (_req: Request, res: Response) => {
  attractUntil = 0;
  res.json({ success: true, data: { mode: null } });
});

export default router;
