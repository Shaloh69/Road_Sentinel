import express, { Application, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { Server } from 'socket.io';
import http from 'http';
import { logger } from './config/logger';
import { testConnection, closePool } from './config/database';
import { initializeStorageBuckets } from './config/supabase';
import { aiService } from './services/ai.service';

// Load environment variables
dotenv.config();

const app: Application = express();
const server = http.createServer(app);
const PORT = process.env.PORT || 3001;
const HOST = process.env.HOST || '0.0.0.0';

// Initialize Socket.IO
const io = new Server(server, {
  cors: {
    origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
    methods: ['GET', 'POST'],
  },
});

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  credentials: true,
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request logging middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  logger.info(`${req.method} ${req.path}`);
  next();
});

// Health check endpoint
app.get('/health', (req: Request, res: Response) => {
  res.status(200).json({
    success: true,
    message: 'Road Sentinel Node Service is running',
    timestamp: new Date().toISOString(),
  });
});

// API status endpoint
app.get('/api/status', async (req: Request, res: Response) => {
  const dbHealthy = await testConnection();
  const aiHealthy = await aiService.healthCheck();

  res.status(200).json({
    success: true,
    data: {
      service: 'road-sentinel-node',
      database: dbHealthy ? 'connected' : 'disconnected',
      ai_service: aiHealthy ? 'connected' : 'disconnected',
      timestamp: new Date().toISOString(),
    },
  });
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  logger.info(`Client connected: ${socket.id}`);

  socket.on('disconnect', () => {
    logger.info(`Client disconnected: ${socket.id}`);
  });

  socket.on('subscribe_camera', (cameraId: string) => {
    socket.join(`camera:${cameraId}`);
    logger.info(`Client ${socket.id} subscribed to camera ${cameraId}`);
  });

  socket.on('unsubscribe_camera', (cameraId: string) => {
    socket.leave(`camera:${cameraId}`);
    logger.info(`Client ${socket.id} unsubscribed from camera ${cameraId}`);
  });
});

// Export io for use in other modules
export { io };

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined,
  });
});

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`,
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM signal received: closing HTTP server');
  server.close(async () => {
    logger.info('HTTP server closed');
    await closePool();
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  logger.info('SIGINT signal received: closing HTTP server');
  server.close(async () => {
    logger.info('HTTP server closed');
    await closePool();
    process.exit(0);
  });
});

// Start server
async function startServer() {
  try {
    // Test database connection
    const dbConnected = await testConnection();
    if (!dbConnected) {
      logger.warn('Database connection failed - server will start but some features may not work');
    }

    // Initialize Supabase storage
    await initializeStorageBuckets();

    // Check AI service
    const aiHealthy = await aiService.healthCheck();
    if (!aiHealthy) {
      logger.warn('AI service is not available - detection features will not work');
    }

    // Start listening
    server.listen(PORT, () => {
      logger.info(`🚀 Server running on http://${HOST}:${PORT}`);
      logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info(`🔌 WebSocket server ready`);
      logger.info(`💾 Database: ${dbConnected ? 'Connected' : 'Disconnected'}`);
      logger.info(`🤖 AI Service: ${aiHealthy ? 'Connected' : 'Disconnected'}`);
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
startServer();
