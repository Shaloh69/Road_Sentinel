import { Router, Request, Response } from "express";
import { logger } from "../config/logger";
import { signAdminToken, timingSafeStringEqual } from "../middleware/auth";

const router = Router();

// ── Simple in-memory brute-force limiter ──────────────────────────────────────
// One shared admin password, no user table — a fixed attempt cap per source IP
// is enough to make guessing impractical without adding a dependency.
const MAX_ATTEMPTS = 5;
const WINDOW_MS = 15 * 60 * 1000; // 15 minutes
const attempts = new Map<string, { count: number; resetAt: number }>();

function isRateLimited(key: string): boolean {
  const now = Date.now();
  const entry = attempts.get(key);
  if (!entry || now > entry.resetAt) {
    attempts.set(key, { count: 0, resetAt: now + WINDOW_MS });
    return false;
  }
  return entry.count >= MAX_ATTEMPTS;
}

function recordFailure(key: string): void {
  const entry = attempts.get(key);
  if (entry) entry.count += 1;
}

// POST /api/auth/login — { password } → { token }
router.post("/login", (req: Request, res: Response) => {
  const key = req.ip || "unknown";
  const adminPassword = process.env.ADMIN_PASSWORD || "";

  if (!adminPassword) {
    logger.error("Login attempted but ADMIN_PASSWORD is not set in .env");
    res
      .status(500)
      .json({ success: false, error: "Admin login is not configured" });
    return;
  }

  if (isRateLimited(key)) {
    res
      .status(429)
      .json({ success: false, error: "Too many attempts — try again later" });
    return;
  }

  const { password } = req.body as { password?: string };
  if (!password || !timingSafeStringEqual(password, adminPassword)) {
    recordFailure(key);
    logger.warn(`Failed admin login attempt from ${key}`);
    res.status(401).json({ success: false, error: "Invalid password" });
    return;
  }

  const token = signAdminToken();
  logger.info(`Admin login succeeded from ${key}`);
  res.json({
    success: true,
    token,
    expiresIn: process.env.JWT_EXPIRES_IN || "12h",
  });
});

export default router;
