import jwt from "jsonwebtoken";
import crypto from "crypto";
import { Request, Response, NextFunction } from "express";
import { Socket } from "socket.io";

const JWT_SECRET = process.env.JWT_SECRET || "";
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || "12h";

if (!JWT_SECRET) {
  // Fail loud at boot rather than silently issuing/accepting unsigned tokens.
  throw new Error(
    "JWT_SECRET is not set. Generate one and add it to server/node-service/.env " +
      "before starting the server (see .env.example).",
  );
}

export type TokenRole = "admin" | "pi-agent";

export interface AuthTokenPayload {
  role: TokenRole;
  piId?: string;
}

export function signAdminToken(): string {
  const payload: AuthTokenPayload = { role: "admin" };
  return jwt.sign(payload, JWT_SECRET, {
    expiresIn: JWT_EXPIRES_IN,
  } as jwt.SignOptions);
}

export function verifyToken(token: string): AuthTokenPayload | null {
  try {
    return jwt.verify(token, JWT_SECRET) as AuthTokenPayload;
  } catch {
    return null;
  }
}

/** Express middleware — requires a valid `Authorization: Bearer <token>` header with role "admin". */
export function requireAuth(
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  const payload = token ? verifyToken(token) : null;

  if (!payload || payload.role !== "admin") {
    res.status(401).json({ success: false, error: "Unauthorized" });
    return;
  }

  next();
}

/**
 * Socket.IO namespace middleware for the `/admin` namespace.
 * Accepts either:
 *   - a short-lived admin JWT (browser admin-terminal sessions), or
 *   - the static PI_AGENT_TOKEN (Raspberry Pi `pi_agent.py` service connections).
 * Rejects the connection outright on failure — no admin/pi-agent event handler
 * ever runs for an unauthenticated socket.
 */
export function adminNamespaceAuth(
  socket: Socket,
  next: (err?: Error) => void,
): void {
  const token = socket.handshake.auth?.token as string | undefined;

  if (!token) {
    next(new Error("Authentication required"));
    return;
  }

  const piAgentToken = process.env.PI_AGENT_TOKEN || "";
  if (piAgentToken && timingSafeStringEqual(token, piAgentToken)) {
    socket.data.role = "pi-agent" as TokenRole;
    next();
    return;
  }

  const payload = verifyToken(token);
  if (payload && payload.role === "admin") {
    socket.data.role = "admin" as TokenRole;
    next();
    return;
  }

  next(new Error("Invalid or expired token"));
}

/** Constant-time string comparison (avoids leaking length/prefix via timing). */
export function timingSafeStringEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) {
    // Still run a comparison of equal-length buffers so failure timing
    // doesn't trivially reveal the length mismatch.
    crypto.timingSafeEqual(bufA, Buffer.alloc(bufA.length));
    return false;
  }
  return crypto.timingSafeEqual(bufA, bufB);
}
