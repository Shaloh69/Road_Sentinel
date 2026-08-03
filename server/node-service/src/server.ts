import express, { Application, Request, Response, NextFunction } from "express";
import cors from "cors";
import dotenv from "dotenv";
import { Server, Socket } from "socket.io";
import http from "http";
import { spawn } from "child_process";
import os from "os";
import { logger } from "./config/logger";
import { testConnection, closePool } from "./config/database";
import { initializeStorageBuckets } from "./config/supabase";
import { aiService } from "./services/ai.service";
import { runMigrations } from "./database/migrate";
import { seedCameras } from "./database/seed";
import { adminNamespaceAuth } from "./middleware/auth";

// Routes
import cameraRoutes, { handlePiFrame } from "./routes/cameras";
import detectionRoutes from "./routes/detections";
import incidentRoutes from "./routes/incidents";
import analyticsRoutes from "./routes/analytics";
import authRoutes from "./routes/auth";
import recordingRoutes from "./routes/recordings";
import publicStatusRoutes from "./routes/public-status";

// Load environment variables
dotenv.config();

const app: Application = express();
const server = http.createServer(app);
const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || "0.0.0.0";

// ── CORS allowlist ────────────────────────────────────────────────────────────
// CORS_ORIGIN is a comma-separated list of allowed origins, e.g.
//   CORS_ORIGIN=http://localhost:3000,https://your-tunnel.trycloudflare.com
// Falls back to localhost:3000 (dev) if unset, rather than allowing "*".
const allowedOrigins = (process.env.CORS_ORIGIN || "http://localhost:3000")
  .split(",")
  .map((o) => o.trim())
  .filter(Boolean);

function isAllowedOrigin(origin: string | undefined): boolean {
  // Same-origin / non-browser requests (curl, server-to-server) send no Origin header.
  if (!origin) return true;
  return allowedOrigins.includes(origin);
}

const corsOptionsDelegate = (
  req: Request,
  callback: (err: Error | null, options?: { origin: boolean }) => void,
) => {
  const origin = req.header("Origin");
  callback(null, { origin: isAllowedOrigin(origin) });
};

// Initialize Socket.IO
const io = new Server(server, {
  cors: {
    origin: (origin, callback) => {
      callback(null, isAllowedOrigin(origin));
    },
    methods: ["GET", "POST"],
  },
});

// Middleware
app.use(cors(corsOptionsDelegate));
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Request logging middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  logger.info(`${req.method} ${req.path}`);
  next();
});

// ── Health & status ────────────────────────────────────────────────────────

app.get("/health", (req: Request, res: Response) => {
  res.status(200).json({
    success: true,
    message: "Road Sentinel Node Service is running",
    timestamp: new Date().toISOString(),
  });
});

app.get("/api/status", async (req: Request, res: Response) => {
  const dbHealthy = await testConnection();
  const aiHealthy = await aiService.healthCheck();

  res.status(200).json({
    success: true,
    data: {
      service: "road-sentinel-node",
      database: dbHealthy ? "connected" : "disconnected",
      ai_service: aiHealthy ? "connected" : "disconnected",
      timestamp: new Date().toISOString(),
    },
  });
});

// ── API routes ─────────────────────────────────────────────────────────────

app.use("/api/auth", authRoutes);
app.use("/api/cameras", cameraRoutes);
app.use("/api/detections", detectionRoutes);
app.use("/api/incidents", incidentRoutes);
app.use("/api/analytics", analyticsRoutes);
app.use("/api/recordings", recordingRoutes);
app.use("/api/public/status", publicStatusRoutes);

// ── Socket.IO — default namespace (public: camera streams, incidents feed) ───
// Nothing here executes a command or reaches admin/Pi-control surfaces.
// That is exclusively handled by the authenticated `/admin` namespace below.

io.on("connection", (socket) => {
  logger.info(
    `🔌 WebSocket client connected: ${socket.id}  (total: ${io.engine.clientsCount})`,
  );

  socket.on("disconnect", () => {
    logger.info(
      `🔌 WebSocket client disconnected: ${socket.id}  (total: ${io.engine.clientsCount})`,
    );
  });

  // ── Camera subscriptions ──────────────────────────────────────────────────

  socket.on("subscribe_camera", (cameraId: string) => {
    socket.join(`camera:${cameraId}`);
    logger.info(`📷 Client subscribed to camera:${cameraId}`);
  });

  socket.on("unsubscribe_camera", (cameraId: string) => {
    socket.leave(`camera:${cameraId}`);
  });

  // WebSocket binary stream — browser subscribes to get JPEG frames directly
  socket.on("subscribe_stream", (cameraId: string) => {
    socket.join(`stream:${cameraId}`);
    logger.info(`📺 Client subscribed to stream:${cameraId}`);
  });

  socket.on("unsubscribe_stream", (cameraId: string) => {
    socket.leave(`stream:${cameraId}`);
    logger.info(`📺 Client unsubscribed from stream:${cameraId}`);
  });

  // Pi camera frame via Socket.IO — zero RTT, no HTTP PUT overhead
  socket.on("pi_frame", (payload: { camera_id: string; data: Buffer }) => {
    if (!payload?.camera_id || !Buffer.isBuffer(payload.data)) return;
    handlePiFrame(payload.camera_id, payload.data);
  });

  socket.on("subscribe_incidents", () => {
    socket.join("incidents");
    logger.info(`Client ${socket.id} subscribed to incidents`);
  });

  socket.on("unsubscribe_incidents", () => {
    socket.leave("incidents");
  });
});

// ── Socket.IO — `/admin` namespace (authenticated: terminal + Pi relay) ──────
// Every connection here must present either a valid short-lived admin JWT
// (browser admin-terminal sessions, issued by POST /api/auth/login) or the
// static PI_AGENT_TOKEN (Raspberry Pi `pi_agent.py` service connections).
// adminNamespaceAuth rejects the connection outright on failure — none of
// the event handlers below ever run for an unauthenticated socket.

const adminIo = io.of("/admin");
adminIo.use(adminNamespaceAuth);

// Track running server-side terminal processes per socket
const terminalProcesses = new Map<string, ReturnType<typeof spawn>>();

// Track connected Pi agents: piId (e.g. 'pi4') → socket.id
const piAgents = new Map<string, string>();

function isAdminUser(socket: Socket): boolean {
  return socket.data.role === "admin";
}

function isPiAgent(socket: Socket): boolean {
  return socket.data.role === "pi-agent";
}

adminIo.on("connection", (socket) => {
  logger.info(
    `🔐 Admin namespace connection: ${socket.id}  role=${socket.data.role}`,
  );

  if (isAdminUser(socket)) {
    // Send current Pi agent status snapshot to this new admin session
    const snapshot: Record<string, boolean> = {};
    for (const [piId] of piAgents.entries()) {
      snapshot[piId] = true;
    }
    socket.emit("pi_status_all", snapshot);
  }

  socket.on("disconnect", () => {
    // Kill any server-side process this socket owned
    const proc = terminalProcesses.get(socket.id);
    if (proc) {
      proc.kill();
      terminalProcesses.delete(socket.id);
    }
    // If this was a Pi agent, notify admins it went offline
    for (const [piId, sid] of piAgents.entries()) {
      if (sid === socket.id) {
        piAgents.delete(piId);
        adminIo.emit("pi_status", { piId, online: false });
        logger.warn(`🔴 Pi agent OFFLINE: ${piId}`);
        break;
      }
    }
    logger.info(`🔐 Admin namespace disconnect: ${socket.id}`);
  });

  // ── Pi agent registration/output relay (pi-agent role only) ──────────────

  socket.on("pi_register", (piId: string) => {
    if (!isPiAgent(socket)) return;
    piAgents.set(piId, socket.id);
    adminIo.emit("pi_status", { piId, online: true });
    logger.info(`🟢 Pi agent ONLINE: ${piId}  socket=${socket.id}`);
  });

  // Pi streams command output → route back to the requesting admin socket
  socket.on(
    "pi_output",
    (payload: { type: string; data: string; requesterId: string }) => {
      if (!isPiAgent(socket)) return;
      const { type, data, requesterId } = payload;
      adminIo.to(requesterId).emit("terminal_output", { type, data });
    },
  );

  // ── Terminal command (admin role only; target: 'server' | 'pi4' | 'pi5') ──

  socket.on("terminal_command", (data: { command: string; target: string }) => {
    if (!isAdminUser(socket)) return;
    const { command, target } = data;
    if (!command?.trim()) return;

    if (target !== "server") {
      // Route command to the Pi agent
      const piSocketId = piAgents.get(target);
      if (!piSocketId) {
        socket.emit("terminal_output", {
          type: "stderr",
          data: `\n[${target} is not connected — start roadsentinel-agent service on that Pi]\n`,
        });
        socket.emit("terminal_output", { type: "exit", data: "" });
        return;
      }
      adminIo.to(piSocketId).emit("pi_command", {
        command,
        requesterId: socket.id,
      });
      return;
    }

    // Server-side: kill previous process for this socket first
    const existing = terminalProcesses.get(socket.id);
    if (existing) {
      existing.kill();
      terminalProcesses.delete(socket.id);
    }

    logger.info(`🖥️  Terminal [server]: ${command}`);

    const isWindows = os.platform() === "win32";
    const shell = isWindows ? "cmd" : "sh";
    const flag = isWindows ? "/c" : "-c";

    const child = spawn(shell, [flag, command], {
      cwd: process.cwd(),
      env: process.env,
    });

    terminalProcesses.set(socket.id, child);

    child.stdout.on("data", (chunk: Buffer) => {
      socket.emit("terminal_output", {
        type: "stdout",
        data: chunk.toString(),
      });
    });

    child.stderr.on("data", (chunk: Buffer) => {
      socket.emit("terminal_output", {
        type: "stderr",
        data: chunk.toString(),
      });
    });

    child.on("close", (code: number | null) => {
      terminalProcesses.delete(socket.id);
      socket.emit("terminal_output", {
        type: "exit",
        data: `\n[Process exited with code ${code ?? "?"}]\n`,
      });
    });

    child.on("error", (err: Error) => {
      terminalProcesses.delete(socket.id);
      socket.emit("terminal_output", {
        type: "stderr",
        data: `\n[Spawn error: ${err.message}]\n`,
      });
    });
  });

  socket.on("terminal_kill", (target: string) => {
    if (!isAdminUser(socket)) return;
    if (target !== "server") {
      const piSocketId = piAgents.get(target);
      if (piSocketId) adminIo.to(piSocketId).emit("pi_kill");
      socket.emit("terminal_output", { type: "stderr", data: "\n^C\n" });
      return;
    }
    const proc = terminalProcesses.get(socket.id);
    if (proc) {
      proc.kill("SIGINT");
      terminalProcesses.delete(socket.id);
      socket.emit("terminal_output", { type: "stderr", data: "\n^C\n" });
    }
  });
});

// Export io so route handlers can broadcast
export { io };

// ── Error handling ─────────────────────────────────────────────────────────

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error("Unhandled error:", err);
  res.status(500).json({
    success: false,
    error: "Internal server error",
    message: process.env.NODE_ENV === "development" ? err.message : undefined,
  });
});

app.use((req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    error: "Not found",
    message: `Route ${req.method} ${req.path} not found`,
  });
});

// ── Graceful shutdown ──────────────────────────────────────────────────────

async function shutdown() {
  logger.info("Shutting down server...");
  server.close(async () => {
    await closePool();
    process.exit(0);
  });
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);

// ── Start ──────────────────────────────────────────────────────────────────

async function startServer() {
  try {
    const dbConnected = await testConnection();
    if (!dbConnected) {
      logger.warn(
        "Database connection failed — server will start but some features may not work",
      );
    } else {
      await runMigrations();
      await seedCameras();
    }

    await initializeStorageBuckets();

    const aiHealthy = await aiService.healthCheck();
    if (!aiHealthy) {
      logger.warn(
        "AI service is not available — detection features will not work",
      );
    }

    server.listen(PORT, () => {
      logger.info(`🚀 Server running on http://${HOST}:${PORT}`);
      logger.info(`📊 Environment: ${process.env.NODE_ENV || "development"}`);
      logger.info(
        `🔌 WebSocket server ready (admin namespace: /admin, authenticated)`,
      );
      logger.info(`💾 Database: ${dbConnected ? "Connected" : "Disconnected"}`);
      logger.info(`🤖 AI Service: ${aiHealthy ? "Connected" : "Disconnected"}`);
    });
  } catch (error) {
    logger.error("Failed to start server:", error);
    process.exit(1);
  }
}

startServer();
