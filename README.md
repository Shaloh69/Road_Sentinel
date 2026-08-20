# 🚗 Road Sentinel — AI Blind-Curve Warning System

An AI-powered dual-camera vehicle detection, speed-estimation, and incident-warning system, built for a real blind curve in Barangay Busay, Cebu, Philippines. Real-time YOLO26 vehicle detection, perspective-corrected speed estimation, physical LED warning signs on both approaches, and a full monitoring/analytics web dashboard.

## 🎯 Overview

This is a **thesis project**. It uses a trained YOLO26 model to run a dual-camera safety system that:

- **Detects vehicles** approaching from both sides of the blind curve (car, motorcycle, bicycle, bus, truck)
- **Measures vehicle speed** — perspective-corrected via a per-camera calibration tool, or a flat pixels-per-meter fallback when uncalibrated
- **Flags incidents** — crash/anomaly detection is currently a heuristic placeholder, clearly labeled as such (`isHeuristic`) in the UI and API, until a real trained model ships (dataset is ready; training itself is a deliberately separate, GPU-heavy step — see Troubleshooting/Training below)
- **Displays LED warnings** to approaching drivers on **both** approaches (Raspberry Pi 4 + Pi 5, each with its own HUB75 matrix)
- **Records** short video segments around detections/incidents, retrievable from the dashboard's History page
- **Alerts** an external webhook (Slack/Discord/Zapier/etc.) on critical incidents, and exposes a no-login public status page for the community
- Logs everything to MySQL and shows it live on a Next.js dashboard styled with a purpose-built "Night Watch" design system (see `client/web/DESIGN.md`)

### System Architecture

```
[Camera A — Pi 4 + LED matrix]              [Camera B — Pi 5 + LED matrix]
       │  RTSP (auto-discovered)                    │  RTSP (auto-discovered)
       ▼                                             ▼
  camera_sender.py ──POST /api/detect (+homography)──▶  AI Service (FastAPI, :8000)
       │  Socket.IO pi_frame · recordings upload            │  YOLO26 vehicle model
       │  roadsentinel-agent (admin terminal relay)          │  heuristic incident model
       ▼                                             ▼
              Node Service (Express + Socket.IO, :3001)
                     │  MySQL              │  WebSocket: public ns (live feeds/incidents)
                     ▼                     │             /admin ns (JWT-authenticated terminal)
      cameras / detections / incidents /   ▼
      hourly_analytics / recordings    Next.js client (:3000)
      tables                           dashboard · monitor · analytics · incidents
                                        history · reports · cameras · settings
                                        admin terminal · /status (public, no login)
```

## ✨ Features

### Vehicle Detection & Tracking
- **Real-time detection** using a trained YOLO26n model (`models/runs/vehicle/…/best.pt`)
- **Multi-class detection**: car, motorcycle, bicycle, bus, truck
- **Dual-camera** coordination, symmetric hardware (Pi 4 = Camera A, Pi 5 = Camera B — both drive their own LED matrix)
- **Adaptive AI-sampling**: full-rate sampling near a live detection, tiered backoff during quiet stretches to cut AI-service load without touching the live-view frame rate
- **Auto-discovery**: RTSP IP isn't hardcoded — `camera_sender.py` discovers and persists the real camera address, since it isn't guaranteed static
- **IR/night-vision auto-switching** via ONVIF (opt-in, `--ir-auto`)

### Speed & Safety Monitoring
- **Perspective-corrected speed** via the Cameras page's Calibration Tool (click 4 known points, solves a homography matrix); falls back to a flat pixels-per-meter estimate when uncalibrated
- **Incident detection**: heuristic placeholder (brightness-variance) until a crash/anomaly model is trained — always labeled `isHeuristic: true` in both the API and the dashboard, never presented as a real detection
- **Recordings**: opt-in local video segment capture (`--record`) around detection activity, uploaded and registered automatically, playable from the History page
- **Webhook alerts**: critical incidents POST to any Slack/Discord/Zapier-compatible webhook URL (`ALERT_WEBHOOK_URL`) — opt-in, silent no-op if unconfigured
- **Public status page** (`/status`, no login) — a phone-friendly, auto-refreshing clear/vehicle-incoming/incident indicator for the community, backed by the same state logic the physical LED signs use
- **Database logging** (MySQL) for every detection, incident, and recording
- **Speed-violation-by-hour export** — a thesis-figure-ready CSV report (Reports page)

### Admin & Operations
- Admin terminal (sidebar → Admin Terminal) — run shell commands on the Node server or either Pi, behind real JWT login (`/api/auth/login`, rate-limited) and a dedicated authenticated Socket.IO `/admin` namespace. Each Pi relays commands via `roadsentinel-agent` — no SSH or open ports required on the Pi itself.
- CORS is allowlisted (`CORS_ORIGIN`), not wildcard.

### Training
- **`training/train.py`** — the real, current trainer. `python train.py --dataset {vehicle,accident,both} --model-size {n,s,m,l,x}`, YOLO26 base weights.
- Merge your own Roboflow datasets first with `training/run_merge_busay.py`.

## 🎨 Design System

The dashboard runs on **"Night Watch"** — a deliberate design system (design tokens, font pairing, motion, toasts) built for this revamp, replacing an unstyled HeroUI starter template. Full rationale, before/after, and the color/typography reasoning are in **`client/web/DESIGN.md`** — written to be directly citable in the thesis write-up.

## 📁 Project Structure

```
RoadSentinel/
├── docs/                     # Audit, revamp record, and supplementary guides
├── training/                 # YOLO26 training — train.py, validate.py, merge/convert/analyze scripts
├── testing/                  # HTTP integration tests against the deployed AI service
├── inference/                # Standalone offline utilities (speed_detection.py, extract_frames.py)
├── server/
│   ├── ai-service/           # FastAPI — traffic + incident detection, homography speed, media storage
│   ├── node-service/         # Express + Socket.IO — REST API, MySQL, auth, admin terminal, webhook alerts
│   └── database/             # mysql_schema.sql (generated reference — migrate.ts is authoritative)
├── client/web/               # Next.js dashboard — see client/web/DESIGN.md for the design system
├── raspi_scripts/            # Runs on the Pis: camera_sender.py, display_manager.py, pi_agent.py, setup scripts
├── datasets/                 # downloaded/ (raw), processed/ (merged, ready to train)
├── models/                   # models/runs/<dataset>/<run>/weights/{best,last}.pt — real trained output lives here
├── docker-compose.yml        # Local MySQL + Adminer for development
└── start.bat                 # One-click local dev stack (Windows)
```

There is no `scripts/` layout, no `models/v1`/`v2`/`production`, and no `train_vehicle_detector.py`/`quick_train.py`/`auto_download_coco.py`/`angled_camera_calibration.py` anywhere in this repo — those were an earlier layout that never matched what's tracked. See `docs/documentation.md §15` for the full history of what moved where.

## 🚀 Quick Start

```bash
# 1. Place downloaded Roboflow datasets under datasets/downloaded/

# 2. Set up the training environment
cd training
python3 -m venv venv_training
source venv_training/bin/activate  # Windows: venv_training\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Merge datasets
python run_merge_busay.py
# -> datasets/processed/busay_vehicle_detection/, datasets/processed/busay_accident_detection/

# 4. Train (YOLO26, not YOLOv8 — the old `--model n --epochs N` flat CLI is gone)
python train.py --dataset vehicle --model-size n --epochs 100
python train.py --dataset accident --model-size n --epochs 100   # crash model — not yet run, dataset is ready
```

Output: `models/runs/<vehicle|accident>/<dataset>_yolo26<size>_<timestamp>/weights/{best,last}.pt`

### Running the full stack locally

**One command (Windows):**

```bat
start.bat
```

Starts Docker Desktop if needed, brings up MySQL + Adminer via `docker-compose.yml`, creates any missing `.env` files, installs Node dependencies if absent, and launches all three services in their own windows. Prints every URL and login when it's done.

**Or start each service manually:**

```bash
# MySQL + Adminer (from repo root)
docker compose up -d              # MySQL on :3307, Adminer DB browser on :8080

# AI service (FastAPI)
cd server/ai-service
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt   # or requirements-cpu.txt on CPU-only machines
cp .env.example .env              # then set TRAFFIC_MODEL_PATH to your trained weight
python -m app.main                # http://localhost:8000

# Node service (Express)
cd server/node-service
npm install
cp .env.example .env              # set DB_*, then generate JWT_SECRET/ADMIN_PASSWORD/PI_AGENT_TOKEN
npm run dev                       # http://localhost:3001

# Client (Next.js)
cd client/web
npm install
npm run dev                       # http://localhost:3000
```

**Inspecting the database:** Adminer at `http://localhost:8080` — system `MySQL`, server `mysql`, username/password/database all per `docker-compose.yml` (local-dev defaults, not production credentials).

Log into the Admin Terminal (sidebar → Admin Terminal) with the `ADMIN_PASSWORD` you set in `server/node-service/.env`. The public, no-login status page is at `/status`.

### Testing against a running AI service

```bash
cd testing
python test_ai.py            # quick smoke test: health, stats, /api/detect*
python test_camera.py        # live camera + optional AI overlay, auto-detects camera index
python test_video.py path/to/video.mp4
python test_images.py path/to/image.jpg
```

To test a specific trained weights file directly (no server needed): `training/validate.py`.

### Camera calibration (perspective-corrected speed)

Open the web client → **Cameras** → select a camera → **Open Calibration Tool** → click the 4 corners of a known rectangle on the road (a lane marking works well) → enter its real width/length in meters → **Save Calibration**. This computes and stores a homography matrix per camera; the AI service uses it automatically on the next frame. See **View Calibration Guide** on the same page for the walkthrough, or `inference/speed_detection.py`'s header comment for the underlying math.

### Optional Pi-side features (opt-in flags on `camera_sender.py`)

```bash
--record --record-dir ./recordings          # local video segment capture, auto-upload
--ir-auto --onvif-port 80 --onvif-user U --onvif-pass P   # ONVIF IR-cut auto-switching
--no-adaptive-sampling                      # disable tiered AI-sampling backoff (on by default)
```

## 📋 Prerequisites

### Hardware
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Recommended for training and low-latency inference (this project's own trained run used an RTX 3060 Ti); CPU works but is much slower
- Two Raspberry Pis (4 and 5), each with a camera and its own HUB75 LED matrix

### Software
- **Python 3.9–3.12** for `training/` (ultralytics + PyTorch); the AI service runs on **3.10–3.12**
- **Node.js 18+** for `server/node-service` and `client/web`
- **MySQL 8.0**, self-hosted and local-only (bound to `localhost`, never exposed publicly — not even through Tailscale). Aiven was dropped entirely in Phase 0.5 after its hostname went NXDOMAIN. `migrate.ts` is the authoritative, idempotent schema source — point a fresh empty database at it and start the server.

## 🔧 Troubleshooting

### Import Error: "YOLO is not exported from module 'ultralytics'"
```bash
pip install ultralytics
```
Correct import: `from ultralytics import YOLO`.

### CUDA Out of Memory (training)
```bash
python train.py --dataset vehicle --model-size n --batch 4
```

### AI service falls back to `yolov8n.pt` instead of the trained model
Check `server/ai-service/.env`'s `TRAFFIC_MODEL_PATH` resolves to a real file (relative paths resolve against `server/ai-service/`, not your shell's CWD) and check the startup log line `Traffic detector ready — custom_model=...` — `False` means it fell back. `GET /api/stats` also reports live load state per model.

### Node service starts but logs "Database connection failed"
The server degrades gracefully (skips migrations/seeding, keeps serving) rather than crashing — but every DB-backed feature (cameras, detections, incidents, analytics, recordings, public status) will 500 until connectivity is restored. Check `DB_HOST`/`DB_PORT`/credentials in `server/node-service/.env` and that the host actually resolves (`nslookup $DB_HOST`) before assuming it's a code bug.

## ⚠️ Known follow-ups (not yet hardware-verified)

Everything below is code-complete and passed every check that doesn't require physical access to the Raspberry Pis, but is genuinely unverified in the real world — pending Tailscale connectivity to both Pis. Don't treat any of these as confirmed working until they've been checked live:

- **Both LED matrices' bug fixes** — Pi 4's legible-text-consistency fix and Pi 5's no-corruption-on-content-change fix (see `docs/Summarization.md` Phase 0) haven't been re-confirmed under a real, repeated run on the actual hardware.
- **Camera reachability on both Pis** — RTSP connectivity for Camera A (Pi 4) and Camera B (Pi 5), including Camera B's auto-discovery recovery path actually triggering and persisting a new IP in practice, not just in code.
- **Always-on 30 FPS live feed, sustained** — the capture/AI-dispatch/frame-push architecture was audited and found sound (Phase 2: no code-level bottleneck), but the actual delivered FPS under real camera load hasn't been measured. `client/web/components/video-feed.tsx` already has the client-side FPS instrumentation needed to check this the moment the feeds are reachable.

Everything else in this README (API endpoints, auth/CORS, the trained vehicle model, the design system, the web dashboard) has been live-verified against running service instances — see `docs/Summarization.md`'s Phase 4 section for the full evidence trail.

## 📖 Further Documentation

- **`docs/documentation.md`** — ground-truth audit of the whole codebase
- **`docs/Summarization.md`** — full phase-by-phase record of this revamp, including what's live-verified vs. hardware-blocked
- **`docs/PROJECT_STRUCTURE.md`** — detailed, folder-by-folder layout
- **`client/web/DESIGN.md`** — the "Night Watch" design system rationale
- **`training/README.md`**, **`raspi_scripts/README.md`**, **`server/README.md`**, **`server/ai-service/README.md`**, **`server/node-service/README.md`** — component-level docs
- **`docs/TRAINING_GUIDE.md`**, **`docs/CAMERA_TEST_GUIDE.md`**, **`docs/START_HERE.md`** — supplementary guides

## 📝 License

This project uses YOLO26/Ultralytics (AGPL-3.0) and PyTorch (BSD). For thesis and educational use.

---

**Good luck with your Busay blind curve system! 🎓🚗**
