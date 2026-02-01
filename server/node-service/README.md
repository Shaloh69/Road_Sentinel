# Road Sentinel Node Service

Main backend service for Road Sentinel traffic monitoring system.

## Features

- **RTSP Stream Management**: Pull and process video streams from Raspberry Pi cameras
- **AI Integration**: Communicate with Python AI service for vehicle and incident detection
- **Database**: MySQL (Aiven) for storing detections, incidents, and analytics
- **Storage**: Supabase Storage for images and video recordings
- **Real-time Updates**: WebSocket (Socket.IO) for live frontend updates
- **RESTful API**: Express.js endpoints for frontend communication

## Tech Stack

- **Runtime**: Node.js 18+
- **Framework**: Express.js
- **Language**: TypeScript
- **Database**: MySQL (Aiven Cloud)
- **Storage**: Supabase Storage
- **Real-time**: Socket.IO
- **Video**: fluent-ffmpeg, node-rtsp-stream
- **Logging**: Winston

---

## 🚀 Complete Installation Guide

### Prerequisites

- **Node.js 18+** ([Download](https://nodejs.org/))
- **MySQL Database** (Aiven Cloud account - [Sign up](https://aiven.io/))
- **Supabase Account** ([Sign up](https://supabase.com/))
- **Python AI Service** running on port 8000

---

## Windows Installation

### Step 1: Navigate to Node Service Directory

```powershell
cd C:\Projects\Thesis\2026\RoadSentinel\server\node-service
```

### Step 2: Install Node.js Dependencies

```powershell
npm install
```

This will install all required packages from `package.json`.

### Step 3: Configure Environment Variables

```powershell
# Copy example env file
copy .env.example .env
```

Edit `.env` file with your credentials:

```env
# Server Configuration
NODE_ENV=development
PORT=3001
HOST=0.0.0.0

# MySQL Database (Aiven)
DB_HOST=your-aiven-mysql-host.aivencloud.com
DB_PORT=3306
DB_USER=avnadmin
DB_PASSWORD=your-password
DB_NAME=road_sentinel
DB_SSL=true

# Supabase Storage
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_BUCKET_INCIDENTS=incidents
SUPABASE_BUCKET_RECORDINGS=recordings

# Python AI Service
AI_SERVICE_URL=http://localhost:8000
AI_SERVICE_TIMEOUT=30000

# RTSP Stream Configuration
RTSP_BUFFER_SIZE=1024000
FRAME_PROCESSING_RATE=5
VIDEO_RECORDING_ENABLED=true
MAX_RECONNECT_ATTEMPTS=5

# WebSocket
WEBSOCKET_PORT=3002
CORS_ORIGIN=http://localhost:3000

# Logging
LOG_LEVEL=info
LOG_FILE=./logs/app.log
```

### Step 4: Setup MySQL Database (Aiven)

1. **Create Aiven MySQL Database:**
   - Go to [Aiven Console](https://console.aiven.io/)
   - Create a new MySQL service
   - Note down the connection details

2. **Run Database Schema:**

```powershell
# Option 1: Using MySQL client (if installed)
mysql -h your-host.aivencloud.com -u avnadmin -p -D road_sentinel < ..\database\mysql_schema.sql

# Option 2: Using Aiven web console
# Copy contents of ../database/mysql_schema.sql and run in Aiven SQL editor
```

### Step 5: Setup Supabase Storage

1. **Create Supabase Project:**
   - Go to [Supabase](https://supabase.com/)
   - Create a new project
   - Get your URL and keys from Project Settings > API

2. **Create Storage Buckets:**
   - Go to Storage in Supabase dashboard
   - Create bucket: `incidents` (public)
   - Create bucket: `recordings` (public)

### Step 6: Create Logs Directory

```powershell
mkdir logs
```

### Step 7: Build TypeScript

```powershell
npm run build
```

### Step 8: Start the Service

**Development mode (with hot reload):**
```powershell
npm run dev
```

**Production mode:**
```powershell
npm start
```

You should see:
```
🚀 Server running on http://0.0.0.0:3001
📊 Environment: development
🔌 WebSocket server ready
💾 Database: Connected
🤖 AI Service: Connected
```

**✅ Node Service is now running!**

---

## Linux/macOS Installation

### Step 1: Navigate to Node Service Directory

```bash
cd /home/user/Road_Sentinel/server/node-service
```

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Step 4: Setup Database

```bash
# Run MySQL schema on Aiven
mysql -h your-host.aivencloud.com -u avnadmin -p road_sentinel < ../database/mysql_schema.sql
```

### Step 5: Create Directories

```bash
mkdir -p logs
```

### Step 6: Build and Run

```bash
# Development
npm run dev

# Production
npm run build
npm start
```

---

## 🧪 Testing the Node Service

### Check Service Status

```powershell
# Open browser or use curl
curl http://localhost:3001/health
```

Response:
```json
{
  "success": true,
  "message": "Road Sentinel Node Service is running",
  "timestamp": "2024-01-31T10:30:00.000Z"
}
```

### Check System Status

```powershell
curl http://localhost:3001/api/status
```

Response:
```json
{
  "success": true,
  "data": {
    "service": "road-sentinel-node",
    "database": "connected",
    "ai_service": "connected",
    "timestamp": "2024-01-31T10:30:00.000Z"
  }
}
```

---

## API Endpoints

### Health & Status

- `GET /health` - Service health check
- `GET /api/status` - Detailed system status (DB, AI service)

### Cameras

- `GET /api/cameras` - List all cameras
- `GET /api/cameras/:id` - Get camera details
- `POST /api/cameras` - Add new camera
- `PUT /api/cameras/:id` - Update camera
- `DELETE /api/cameras/:id` - Delete camera

### Detections

- `GET /api/detections` - Get recent detections
- `GET /api/detections/camera/:id` - Get detections by camera

### Incidents

- `GET /api/incidents` - Get incidents
- `GET /api/incidents/:id` - Get incident details
- `PUT /api/incidents/:id` - Update incident status

### Analytics

- `GET /api/analytics/hourly` - Get hourly statistics
- `GET /api/analytics/daily` - Get daily statistics

---

## WebSocket Events

### Client → Server

- `subscribe_camera` - Subscribe to camera updates
- `unsubscribe_camera` - Unsubscribe from camera

### Server → Client

- `detection` - New vehicle detection
- `incident` - New incident detected
- `camera_status` - Camera status change

---

## Project Structure

```
src/
├── config/           # Configuration files
│   ├── database.ts   # MySQL connection
│   ├── supabase.ts   # Supabase client
│   └── logger.ts     # Winston logger
├── services/         # Business logic services
│   ├── ai.service.ts       # AI service client
│   └── storage.service.ts  # Supabase storage
├── types/            # TypeScript types
│   └── index.ts
└── server.ts         # Main server file

logs/                 # Log files
dist/                 # Compiled JavaScript (after build)
```

---

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_HOST` | Aiven MySQL host | `mysql-xxx.aivencloud.com` |
| `DB_USER` | Database user | `avnadmin` |
| `DB_PASSWORD` | Database password | `your-password` |
| `DB_NAME` | Database name | `road_sentinel` |
| `SUPABASE_URL` | Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service key | `eyJhbGc...` |
| `AI_SERVICE_URL` | Python AI service URL | `http://localhost:8000` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3001` |
| `LOG_LEVEL` | Logging level | `info` |
| `CORS_ORIGIN` | Frontend URL | `http://localhost:3000` |
| `FRAME_PROCESSING_RATE` | Process every Nth frame | `5` |

---

## Troubleshooting

### "Cannot connect to MySQL database"

**Check:**
1. Aiven MySQL service is running
2. Firewall rules allow connection
3. Credentials are correct in `.env`
4. SSL is enabled (`DB_SSL=true`)

**Solution:**
```powershell
# Test connection manually
mysql -h your-host.aivencloud.com -u avnadmin -p
```

### "AI service not available"

Make sure Python AI service is running:
```powershell
cd ..\ai-service
.\venv\Scripts\Activate.ps1
python -m app.main
```

### "Module not found" errors

Reinstall dependencies:
```powershell
rm -rf node_modules
rm package-lock.json
npm install
```

### Port 3001 already in use

Change port in `.env`:
```env
PORT=3002
```

### TypeScript compilation errors

Make sure TypeScript is installed:
```powershell
npm install -D typescript ts-node @types/node
npm run build
```

---

## Development Scripts

```powershell
# Install dependencies
npm install

# Run in development mode (hot reload)
npm run dev

# Build TypeScript to JavaScript
npm run build

# Run production server
npm start

# Run linter
npm run lint

# Format code
npm run format
```

---

## Integration with AI Service

The Node service communicates with the Python AI service:

```typescript
// Example: Send frame for detection
const result = await aiService.detectObjects(
  frameBuffer,    // Image buffer
  'CAM-A-001',   // Camera ID
  0.75           // Confidence threshold
);

// Result contains:
// - detections: Vehicle detections
// - incidents: Incident detections
// - processing_time_ms: Inference time
```

---

## Database Schema

See `../database/mysql_schema.sql` for complete schema.

**Main Tables:**
- `cameras` - Camera configuration
- `detections` - Vehicle detections
- `incidents` - Traffic incidents
- `analytics_hourly` - Hourly statistics
- `recordings` - Video recordings

---

## Deployment

### Using PM2 (Process Manager)

```powershell
# Install PM2 globally
npm install -g pm2

# Build the project
npm run build

# Start with PM2
pm2 start dist/server.js --name road-sentinel-node

# Monitor
pm2 logs road-sentinel-node

# Stop
pm2 stop road-sentinel-node
```

---

## Next Steps

1. **Start AI Service** (port 8000)
2. **Start Node Service** (port 3001)
3. **Start Frontend** (port 3000)
4. **Setup Raspberry Pi** with RTSP streaming
5. **Configure cameras** in database

---

## License

MIT
