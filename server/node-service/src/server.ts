import express, { Application, Request, Response, NextFunction } from "express";
import cors from "cors";
import dotenv from "dotenv";
import { Server } from "socket.io";
import http from "http";
import { spawn } from "child_process";
import os from "os";
import { logger } from "./config/logger";
import { testConnection, closePool } from "./config/database";
import { initializeStorageBuckets } from "./config/supabase";
import { aiService } from "./services/ai.service";
import { runMigrations } from "./database/migrate";
import { seedCameras } from "./database/seed";

// Routes
import cameraRoutes from "./routes/cameras";
import detectionRoutes from "./routes/detections";
import incidentRoutes from "./routes/incidents";
import analyticsRoutes from "./routes/analytics";

// Load environment variables
dotenv.config();

const app: Application = express();
const server = http.createServer(app);
const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || "0.0.0.0";

// Initialize Socket.IO
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
  },
});

// Middleware
app.use(cors({ origin: "*" }));
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

app.use("/api/cameras", cameraRoutes);
app.use("/api/detections", detectionRoutes);
app.use("/api/incidents", incidentRoutes);
app.use("/api/analytics", analyticsRoutes);

// ── Socket.IO ──────────────────────────────────────────────────────────────

// Track running server-side terminal processes per socket
const terminalProcesses = new Map<string, ReturnType<typeof spawn>>();

// Track connected Pi agents: piId (e.g. 'pi4') → socket.id
const piAgents = new Map<string, string>();

io.on("connection", (socket) => {
  logger.info(
    `🔌 WebSocket client connected: ${socket.id}  (total: ${io.engine.clientsCount})`,
  );

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
        io.to("admin").emit("pi_status", { piId, online: false });
        logger.warn(`🔴 Pi agent OFFLINE: ${piId}`);
        break;
      }
    }
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

  socket.on("subscribe_incidents", () => {
    socket.join("incidents");
    logger.info(`Client ${socket.id} subscribed to incidents`);
  });

  socket.on("unsubscribe_incidents", () => {
    socket.leave("incidents");
  });

  // ── Admin terminal ────────────────────────────────────────────────────────

  socket.on("subscribe_admin", () => {
    socket.join("admin");
    logger.info(`🖥️  Admin terminal session opened: ${socket.id}`);
    // Send current Pi agent status snapshot to this new admin client
    const snapshot: Record<string, boolean> = {};
    for (const [piId] of piAgents.entries()) {
      snapshot[piId] = true;
    }
    socket.emit("pi_status_all", snapshot);
  });

  socket.on("unsubscribe_admin", () => {
    socket.leave("admin");
  });

  // ── Pi agent ─────────────────────────────────────────────────────────────

  socket.on("pi_register", (piId: string) => {
    piAgents.set(piId, socket.id);
    io.to("admin").emit("pi_status", { piId, online: true });
    logger.info(`🟢 Pi agent ONLINE: ${piId}  socket=${socket.id}`);
  });

  // Pi streams command output → route back to the requesting admin socket
  socket.on(
    "pi_output",
    (payload: { type: string; data: string; requesterId: string }) => {
      const { type, data, requesterId } = payload;
      io.to(requesterId).emit("terminal_output", { type, data });
    },
  );

  // ── Terminal command (target: 'server' | 'pi4' | 'pi5') ──────────────────

  socket.on("terminal_command", (data: { command: string; target: string }) => {
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
      io.to(piSocketId).emit("pi_command", {
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
    if (target !== "server") {
      const piSocketId = piAgents.get(target);
      if (piSocketId) io.to(piSocketId).emit("pi_kill");
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
      logger.info(`🔌 WebSocket server ready`);
      logger.info(`💾 Database: ${dbConnected ? "Connected" : "Disconnected"}`);
      logger.info(`🤖 AI Service: ${aiHealthy ? "Connected" : "Disconnected"}`);
    });
  } catch (error) {
    logger.error("Failed to start server:", error);
    process.exit(1);
  }
}

startServer();
