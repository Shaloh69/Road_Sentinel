# Road Sentinel Server

Backend services for the Road Sentinel traffic monitoring system.

## Architecture

```
┌─────────────────┐
│  Raspberry Pi   │  Camera + RTSP Streaming
└────────┬────────┘
         │ RTSP Stream (rtsp://ip:8554/camera1)
         ▼
┌──────────────────────────────────────────────┐
│         Node.js Main Service                 │
│  - Express.js REST API                       │
│  - RTSP Stream Management                    │
│  - MySQL Database (Aiven)                    │
│  - Supabase Storage (Images/Videos)          │
│  - WebSocket (Real-time updates)             │
└────────┬─────────────────────────────────────┘
         │ HTTP API Calls
         ▼
┌──────────────────────────────────────────────┐
│      Python AI Microservice                  │
│  - FastAPI                                   │
│  - YOLOv8 Traffic Detection                  │
│  - YOLOv8 Incident Detection                 │
│  - GPU Accelerated Inference                 │
└──────────────────────────────────────────────┘
```

## Services

### 1. Node.js Service (`node-service/`)

Main backend service handling:
- RTSP video stream processing
- Database operations (MySQL on Aiven)
- File storage (Supabase Storage)
- WebSocket real-time updates
- RESTful API for frontend

**Tech Stack:**
- Node.js + Express + TypeScript
- MySQL (Aiven Cloud)
- Supabase Storage
- Socket.IO

[View Node Service README](./node-service/README.md)

### 2. Python AI Service (`ai-service/`)

AI/ML microservice for:
- Vehicle detection and classification
- Speed estimation
- Incident detection (crashes, violations)
- Real-time inference

**Tech Stack:**
- Python + FastAPI
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV

[View AI Service README](./ai-service/README.md)

## Database Schema

MySQL schema is located in `database/mysql_schema.sql`

**Tables:**
- `cameras` - Camera configuration and metadata
- `detections` - Vehicle detection records
- `incidents` - Traffic incidents and alerts
- `analytics_hourly` - Aggregated hourly statistics
- `recordings` - Video recording metadata

## Quick Start

### Prerequisites

- **Node.js** 18+ (for node-service)
- **Python** 3.10+ (for ai-service)
- **MySQL** (Aiven Cloud account)
- **Supabase** account for storage
- **NVIDIA GPU** (optional, for faster AI inference)

### 1. Setup MySQL Database

```bash
# Connect to your Aiven MySQL instance and run the schema
mysql -h your-host.aivencloud.com -u avnadmin -p road_sentinel < database/mysql_schema.sql
```

### 2. Setup Node.js Service

```bash
cd node-service
npm install
cp .env.example .env
# Edit .env with your credentials
npm run dev
```

### 3. Setup Python AI Service

```bash
cd ai-service
pip install -r requirements.txt
cp .env.example .env
# Place your YOLOv8 models in models/ directory
python -m app.main
```

### 4. Verify Services

- Node Service: http://localhost:3001/health
- AI Service: http://localhost:8000/health

## Environment Variables

### Node Service

```env
# Database
DB_HOST=your-aiven-host.aivencloud.com
DB_USER=avnadmin
DB_PASSWORD=your-password
DB_NAME=road_sentinel

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key

# AI Service
AI_SERVICE_URL=http://localhost:8000
```

### AI Service

```env
TRAFFIC_MODEL_PATH=./models/traffic.pt
INCIDENT_MODEL_PATH=./models/incident.pt
DEVICE=cuda  # or 'cpu'
```

## Data Flow

1. **Raspberry Pi** streams video via RTSP
2. **Node Service** pulls RTSP stream, extracts frames
3. **Node Service** sends frames to **AI Service** for inference
4. **AI Service** returns detections and incidents
5. **Node Service** stores results in MySQL and Supabase
6. **Node Service** broadcasts updates via WebSocket to frontend
7. **Frontend** displays live detections and statistics

## API Endpoints

### Node Service (Port 3001)

- `GET /health` - Health check
- `GET /api/status` - System status
- `GET /api/cameras` - List cameras
- `GET /api/detections` - Get detections
- `GET /api/incidents` - Get incidents
- `GET /api/analytics/hourly` - Hourly statistics

### AI Service (Port 8000)

- `GET /health` - Health check
- `POST /api/detect` - Combined detection
- `POST /api/detect/traffic` - Traffic only
- `POST /api/detect/incidents` - Incidents only

## WebSocket Events

**Client → Server:**
- `subscribe_camera` - Subscribe to camera updates
- `unsubscribe_camera` - Unsubscribe

**Server → Client:**
- `detection` - New vehicle detection
- `incident` - New incident
- `camera_status` - Camera status update

## Docker Deployment (Coming Soon)

```bash
docker-compose up -d
```

## Production Considerations

1. **Scaling**: Use multiple AI service instances for handling multiple cameras
2. **Load Balancing**: Nginx reverse proxy for Node service
3. **Monitoring**: Prometheus + Grafana for metrics
4. **Logging**: Centralized logging (ELK stack)
5. **Security**:
   - Use SSL/TLS for all connections
   - Secure RTSP streams
   - Implement API authentication
   - Enable Supabase Row Level Security

## Performance Optimization

- **Frame Processing Rate**: Process 1-5 FPS instead of 30 FPS
- **Batch Processing**: Process multiple frames together
- **Model Optimization**: Use YOLOv8n (nano) for faster inference
- **Caching**: Cache analytics data
- **Database Indexing**: Proper indexes on timestamp columns

## Troubleshooting

### Node Service won't connect to MySQL
- Check Aiven firewall rules
- Verify SSL certificate
- Check credentials in .env

### AI Service slow inference
- Ensure CUDA is properly installed
- Check GPU availability
- Consider using smaller YOLOv8 model (v8n instead of v8x)

### RTSP stream fails
- Verify Raspberry Pi IP and port
- Check network connectivity
- Ensure MediaMTX is running on Raspberry Pi

## Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## License

MIT
