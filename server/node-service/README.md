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

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

- MySQL (Aiven) connection details
- Supabase URL and keys
- Python AI service URL
- CORS origin (frontend URL)

### 3. Database Setup

Run the MySQL schema on your Aiven database:

```bash
mysql -h your-host.aivencloud.com -u avnadmin -p road_sentinel < ../database/mysql_schema.sql
```

### 4. Create Required Directories

```bash
mkdir -p logs
```

## Development

Start development server with hot reload:

```bash
npm run dev
```

## Production

Build TypeScript to JavaScript:

```bash
npm run build
```

Start production server:

```bash
npm start
```

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

## WebSocket Events

### Client → Server

- `subscribe_camera` - Subscribe to camera updates
- `unsubscribe_camera` - Unsubscribe from camera

### Server → Client

- `detection` - New vehicle detection
- `incident` - New incident detected
- `camera_status` - Camera status change

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
```

## Environment Variables

See `.env.example` for all required variables.

## License

MIT
