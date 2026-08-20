# Road Sentinel Node Service

Express + Socket.IO backend: REST API, MySQL persistence, AI-service orchestration, real-time updates, and an authenticated admin terminal.

## Stack

- Node.js + Express + TypeScript
- MySQL 8.0, self-hosted (local Docker for development, local-only on `irm-pc` in production — **not** Aiven, which was dropped entirely in Phase 0.5 after its hostname went NXDOMAIN)
- Socket.IO — a public default namespace (live feeds, incidents, detections) and a separate, JWT-authenticated `/admin` namespace (admin terminal)
- Storage: the AI service's local disk (`server/ai-service/media/`), **not** Supabase — Supabase was evaluated early on and never actually wired up; there's no Supabase code path left to configure

## Setup

```bash
npm install
cp .env.example .env
# Fill in DB_* (see docker-compose.yml at the repo root for local defaults),
# then generate JWT_SECRET / ADMIN_PASSWORD / PI_AGENT_TOKEN.
npm run dev       # http://localhost:3001, auto-reloads via nodemon
```

The easiest path is `start.bat` at the repo root, which brings up local MySQL (Docker), this service, the AI service, and the client together.

On startup this service checks the database connection; if MySQL is reachable it runs migrations (`src/database/migrate.ts` — idempotent, safe to run against an existing database) and seeds the two Busay cameras. If MySQL is unreachable, the server still starts (degraded — DB-backed routes will 500) rather than crashing, so you can at least confirm the process itself boots.

## Authentication

`POST /api/auth/login` (rate-limited) exchanges `ADMIN_PASSWORD` for a JWT. That token is required to connect to the `/admin` Socket.IO namespace (`adminNamespaceAuth` middleware, `src/middleware/auth.ts`) — the admin terminal, and the channel Raspberry Pi agents (`pi_agent.py`) use to relay shell commands. The public namespace (live camera feeds, incidents, detections) requires no auth by design — it's read-only, no camera credentials or config are exposed through it.

Raspberry Pi agents authenticate to the same `/admin` namespace using `PI_AGENT_TOKEN` (must match exactly between this service's `.env` and each Pi's environment).

CORS is an explicit allowlist (`CORS_ORIGIN`, comma-separated) — no wildcard.

## API Endpoints

| Route | Purpose |
|---|---|
| `POST /api/auth/login` | Admin login → JWT |
| `GET/POST/PUT/DELETE /api/cameras` | Camera CRUD, including `homography_points` for calibrated speed |
| `GET /api/detections` | Vehicle detections, filterable by camera/date range (`until` param) |
| `GET/POST/PUT /api/incidents` | Incidents — creating a `critical`-severity one also fires the webhook alert (below) |
| `GET /api/analytics/summary` \| `/hourly` \| `/speed` \| `/violations` | Dashboard stats, hourly traffic, speed histogram, speed violations by hour-of-day (thesis figure) |
| `GET/POST /api/recordings` | Recorded video segments (opt-in, Pi-side `--record`) |
| `GET /api/public/status` | **Unauthenticated.** Clear/vehicle-incoming/incident state + today's tallies — backs the public `/status` page, exposes no camera feeds or config |

## WebSocket events

**Public namespace (`/`)**

| Direction | Event | Purpose |
|---|---|---|
| Client → Server | `subscribe_camera` / `unsubscribe_camera` | Camera status updates |
| Client → Server | `subscribe_stream` / `unsubscribe_stream` | Binary JPEG frame push for a camera's live view |
| Client → Server | `subscribe_incidents` / `unsubscribe_incidents` | Live incident feed |
| Pi → Server | `pi_frame` | Camera frame upload, relayed to subscribed clients |
| Server → Client | `detection`, `incident`, `camera_status` | Real-time updates |

**Admin namespace (`/admin`, JWT or `PI_AGENT_TOKEN` required)**

| Direction | Event | Purpose |
|---|---|---|
| Pi → Server | `pi_register` | Pi agent announces itself online |
| Pi → Server | `pi_output` | Streams shell command output back to the requesting admin |
| Admin → Server | `terminal_command` / `terminal_kill` | Run/kill a shell command on the server or a specific Pi (`target: 'server' \| 'pi4' \| 'pi5'`) |
| Server → Admin | `terminal_output`, `pi_status`, `pi_status_all` | Command output, Pi online/offline state |

## Webhook alerts

Set `ALERT_WEBHOOK_URL` (and optionally `ALERT_WEBHOOK_MIN_SEVERITY`, default `critical`) to POST a Slack/Discord/Zapier-compatible JSON payload whenever a matching-severity incident is created (`src/services/alert.service.ts`). Unset by default — silent no-op, not an error, if not configured.

## Environment variables

See `.env.example` for the full annotated list. The load-bearing ones:

```env
DB_HOST=localhost           # local MySQL — see docker-compose.yml
DB_PORT=3307                # 3307 for the local dev Docker container; irm-pc production uses the real 3306
DB_USER=roadsentinel
DB_PASSWORD=...
DB_NAME=roadsentinel
DB_SSL=false                 # always false for a local/loopback connection

AI_SERVICE_URL=http://localhost:8000

JWT_SECRET=...                # node -e "console.log(require('crypto').randomBytes(48).toString('hex'))"
ADMIN_PASSWORD=...
PI_AGENT_TOKEN=...            # must match each Pi's PI_AGENT_TOKEN

CORS_ORIGIN=http://localhost:3000

# Optional
ALERT_WEBHOOK_URL=
ALERT_WEBHOOK_MIN_SEVERITY=critical
```

## Troubleshooting

**"Database connection failed" on startup** — the server still starts, but every DB-backed route 500s. Check `DB_HOST`/`DB_PORT`/credentials and that the host actually resolves/is reachable (`docker compose ps` if using the local Docker MySQL).

**AI service unreachable** — detection features won't work, but the rest of the API still functions. Check `AI_SERVICE_URL` and that `server/ai-service` is actually running.

**Admin terminal connection rejected** — the token sent from the client doesn't match a currently-valid JWT, or a Pi's `PI_AGENT_TOKEN` doesn't match this service's. Re-login (client) or re-check the shared token (Pi).
