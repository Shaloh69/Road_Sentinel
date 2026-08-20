# Road Sentinel Server

Backend services for the Road Sentinel traffic monitoring system.

## Architecture

```
┌──────────────────┐
│  Raspberry Pi     │  camera_sender.py: RTSP capture, pushes frames to
│  (Camera A or B)  │  the AI service and streams to Node over Socket.IO
└────────┬──────────┘
         │ POST /api/detect (+homography) · Socket.IO pi_frame
         ▼
┌──────────────────────────────────────────────┐      ┌─────────────────────┐
│         Node.js Service (:3001)               │◄────►│  MySQL (self-hosted,│
│  - Express REST API                            │      │  local-only)        │
│  - JWT auth + authenticated /admin namespace   │      └─────────────────────┘
│  - Socket.IO (public ns + /admin ns)           │
│  - Webhook alerts on critical incidents        │
└────────┬───────────────────────────────────────┘
         │ HTTP API calls
         ▼
┌──────────────────────────────────────────────┐
│      Python AI Service (:8000)                │
│  - FastAPI                                     │
│  - YOLO26 vehicle detection & tracking         │
│  - Heuristic incident detection (no trained    │
│    crash model yet — clearly labeled as such)  │
│  - Homography-corrected speed estimation       │
│  - Local disk media storage                    │
└──────────────────────────────────────────────┘
```

## Services

### 1. Node service (`node-service/`)

Express + TypeScript + Socket.IO. REST API, MySQL persistence, JWT admin auth, real-time updates, admin terminal (relays shell commands to itself or either Raspberry Pi over an authenticated Socket.IO namespace — no SSH or open ports needed on the Pis), webhook alerting.

[View Node Service README](./node-service/README.md)

### 2. Python AI service (`ai-service/`)

FastAPI. Vehicle detection/classification/tracking, homography-corrected speed estimation, incident detection (heuristic until a crash model is trained), local media storage.

[View AI Service README](./ai-service/README.md)

## Database

MySQL, self-hosted and local-only — bound to `localhost`, never exposed publicly (not even through Tailscale). For local development, `docker-compose.yml` at the repo root brings up a MySQL 8.0 container plus Adminer (a web DB browser at `http://localhost:8080`); in production it runs the same way directly on `irm-pc`. Aiven was used earlier and dropped entirely in Phase 0.5 after its hostname went NXDOMAIN — there's no migration path from it, just a fresh schema.

Schema is defined in `database/mysql_schema.sql` (a generated reference) and applied by `node-service/src/database/migrate.ts`, which is the **authoritative** source and idempotent — safe to run against a database that already has some or all of the tables.

**Tables:** `cameras`, `detections`, `incidents`, `hourly_analytics`, `recordings`.

## Quick start

The easiest path is `start.bat` at the repo root (Windows) — brings up local MySQL, both services, and the client together, with sensible local-dev defaults.

To run the services individually:

```bash
# MySQL (local Docker, from repo root)
docker compose up -d

# Node service
cd node-service
npm install
cp .env.example .env   # see docker-compose.yml for the matching local DB credentials
npm run dev             # http://localhost:3001

# AI service
cd ai-service
python -m venv venv && venv/Scripts/activate   # or source venv/bin/activate on Linux/macOS
pip install -r requirements.txt   # or requirements-cpu.txt on CPU-only machines
cp .env.example .env   # set TRAFFIC_MODEL_PATH to your trained weight
python -m app.main      # http://localhost:8000
```

Verify: `GET http://localhost:3001/health` and `GET http://localhost:8000/health`.

## Data flow

1. Each Raspberry Pi's `camera_sender.py` captures its RTSP stream and POSTs frames to the AI service for inference (`/api/detect`), while separately streaming raw frames to the Node service over Socket.IO (`pi_frame`) for the live dashboard view.
2. The AI service returns detections (and, once a crash model exists, real incidents — currently heuristic) to the Pi script, which forwards them to Node.
3. Node stores results in MySQL and broadcasts them over Socket.IO to connected dashboard clients.
4. Critical incidents optionally fire a webhook (`ALERT_WEBHOOK_URL`).
5. The dashboard (`client/web`) and the public `/status` page both read live state from Node's REST API and WebSocket events.

## Security notes

- JWT-authenticated admin actions (login rate-limited); a separate, non-authenticated public Socket.IO namespace exposes only read-only live status.
- CORS is an explicit allowlist, not a wildcard.
- MySQL is never exposed outside `localhost`.
- Raspberry Pi agents authenticate with a shared `PI_AGENT_TOKEN`, not SSH/open ports.

## Troubleshooting

**Node service can't connect to MySQL** — check `DB_HOST`/`DB_PORT`/credentials in `node-service/.env`; if using the local Docker container, confirm it's running (`docker compose ps`) and healthy.

**AI service falls back to the stock model** — check `TRAFFIC_MODEL_PATH` in `ai-service/.env` (resolves relative to `server/ai-service/`) and the startup log line `Traffic detector ready — custom_model=...`.

**RTSP stream fails on a Pi** — `camera_sender.py` has ONVIF/subnet-scan auto-discovery that persists a working IP back to Node; a hardcoded IP going stale is expected occasionally since Camera B's address is DHCP-assigned, not static.
