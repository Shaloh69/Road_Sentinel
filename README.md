# 🚗 Road Sentinel — AI Blind-Curve Warning System

An AI-powered dual-camera vehicle detection, speed-estimation and incident-warning system, built for a real blind curve in Barangay Busay, Cebu, Philippines. YOLO26 detection, physical LED warning signs on both approaches, and a full monitoring dashboard.

**This is a thesis project.** The status below is deliberately honest — a safety system that overstates what it does is worse than one that admits its gaps.

---

## 📊 Where the system actually is

| Component | Status | Notes |
|---|---|---|
| Vehicle detection | ✅ **Working** | Trained YOLO26n, mAP50 0.90 / mAP50-95 0.70 |
| LED sign — Pi 4 (ESP32) | ✅ **Working** | Legible text, 250 fps, confirmed on the physical panel |
| LED sign — Pi 5 (STM32) | 🟡 **Untested** | Port compiles; never flashed to a board |
| Web dashboard | ✅ **Working** | Live view, incidents, analytics, public status page |
| Auth & CORS | ✅ **Working** | JWT login, authenticated `/admin` namespace, allowlisted CORS |
| MySQL logging | ✅ **Working** | `migrate.ts` is the authoritative schema |
| Speed estimation | ⚠️ **Approximate** | Production uses **uncorrected pixel distance**; homography exists but is not wired in |
| Crash / incident detection | ❌ **Not implemented** | Brightness-variance heuristic, labelled `isHeuristic` everywhere. No model trained |
| Night / low-light handling | ⚠️ **Partial** | ONVIF IR switching is opt-in and only in the legacy path |
| Failure alerting | ❌ **None** | Nothing notifies anyone when a camera, Pi or sign dies |

Legend: ✅ verified on real hardware · 🟡 code-complete, unverified · ⚠️ works but compromised · ❌ not built

---

## 🚦 Is it ready for production?

**For the vehicle-warning function on the Pi 4 approach: yes, as a supervised field pilot.**
**As the complete system described above: not yet.**

Three things stand between here and an unsupervised deployment:

1. **Crash detection is not real.** `incident_detector.py` falls back to a brightness-variance heuristic its own source calls a "simplified example". The dashboard, the database and the sign's `STOP` state all carry incident output. It is labelled `isHeuristic` honestly throughout — but a headline feature is a placeholder, and the accident dataset is prepared but the model has never been trained.

2. **Speed figures are systematically biased.** Production math uses raw pixel distance over time. Without perspective correction, a vehicle far up the curve reads slower than one near the camera. A working homography implementation exists in `inference/camera_calibration.py` — the server just never calls it.

3. **Half the physical output is unverified.** The Pi 4 sign is confirmed working. The Pi 5 sign has never been flashed. There has also been no recent clean end-to-end run with both cameras, both signs, the dashboard and the database observed together.

Beyond those: nothing alerts anyone when a component fails. For a roadside safety device that fails silently, that is the wrong failure mode — and it is cheap to fix.

**What is genuinely solid:** the detection model, the sign firmware and its serial protocol, the dashboard, authentication, and the database layer. The architecture also fails honestly by design — if the Pi goes quiet the sign shows `NO DATA` rather than a stale `SAFE`, because the board runs its own timeout rather than trusting the last thing it heard.

Full checklist: **[`docs/FUTURE_IMPROVEMENTS.md`](docs/FUTURE_IMPROVEMENTS.md)**.

---

## 🎯 Overview

- **Detects vehicles** approaching from both sides of the curve (car, motorcycle, bicycle, bus, truck)
- **Estimates speed** — perspective-corrected when calibrated, flat pixels-per-meter otherwise
- **Flags incidents** — heuristic placeholder, clearly labelled as such, until a trained model ships
- **Warns drivers** via a physical LED sign on each approach
- **Records** short video segments around detections, retrievable from the dashboard
- **Alerts** an external webhook on critical incidents; exposes a no-login public status page
- Logs everything to MySQL and shows it live on a Next.js dashboard ("Night Watch" design system)

### System architecture

```
[Camera A — Pi 4]                              [Camera B — Pi 5]
      │ RTSP (auto-discovered)                       │ RTSP (auto-discovered)
      ▼                                              ▼
 camera_sender.py ──POST /api/detect──▶ AI Service (FastAPI, :8000)
      │                                       │  YOLO26 vehicle model  ✅
      │                                       │  incident model        ❌ heuristic stub
      ▼                                       ▼
 led_sign_bridge.py                    Node Service (Express + Socket.IO, :3001)
      │ USB serial (115200)                   │ MySQL
      ▼                                       │ public ns: live feeds, incidents
 ┌──────────────────┐                         │ /admin ns: JWT-authenticated terminal
 │ ESP32  (Pi 4) ✅ │                         ▼
 │ STM32  (Pi 5) 🟡 │              Next.js client (:3000)
 │  HUB75 128×32    │              dashboard · monitor · analytics · incidents
 └──────────────────┘              history · reports · cameras · settings
   250 fps, own driver             admin terminal · /status (public, no login)
```

**The Pis do not drive the LED panels.** Each sends road state over USB serial to a
microcontroller running firmware from [`LEDMatrixDrivers/`](LEDMatrixDrivers/). The Pi-GPIO
path was tried exhaustively (~80 configurations) and never worked on these
1/8-scan FM6124 panels; it has been deleted. See
[`LEDMatrixDrivers/esp32/DEBUG_LOG.md`](LEDMatrixDrivers/esp32/DEBUG_LOG.md).

---

## ✨ Features

### Working

- **Real-time vehicle detection** — trained YOLO26n, five classes, dual-camera with independent per-camera trackers
- **Adaptive AI sampling** — full rate near a live detection, tiered backoff when quiet, without touching the live-view frame rate
- **RTSP auto-discovery** — the camera IP is not hardcoded; `camera_sender.py` finds and persists the real address
- **LED signs** — `SAFE` (green) / `VEHICLE INCOMING` + `SLOW DOWN` (flashing yellow) / `STOP` (flashing red), driven by a hand-written HUB75 driver
- **Public status page** (`/status`, no login) — phone-friendly, backed by the same state logic the signs use, so the page and the sign cannot disagree
- **Admin terminal** — shell access to the Node server or either Pi behind JWT login and an authenticated Socket.IO namespace; no SSH or open ports on the Pi
- **Recordings** — opt-in segment capture around detection activity, auto-uploaded and registered
- **Webhook alerts** on critical incidents (Slack/Discord/Zapier-compatible)
- **CSV export** — speed violations by hour, thesis-figure ready

### Compromised or missing

- **Crash detection** — heuristic only, no trained model
- **Speed accuracy** — no perspective correction in the production path
- **Calibration Tool** — the client's buttons are decorative; no handler wired
- **History / Reports pages** — built as fixture shells, now partially wired to real data; needs an audit that every panel reads live values
- **Recording playback** — the `recordings` table exists, the client has no player
- **Sign state in the dashboard** — the web UI cannot show what either physical sign is displaying

---

## 🔭 What is coming next

Ordered by what would most improve the system, not by effort. Full list in
[`docs/FUTURE_IMPROVEMENTS.md`](docs/FUTURE_IMPROVEMENTS.md).

1. **Train the crash/incident model.** The dataset is prepared. This closes the largest gap between claim and reality.
2. **Wire homography into production speed.** The code exists; it needs connecting and validating against ground truth.
3. **Flash and verify the Pi 5 sign**, then re-verify the Pi 4 sign after the driver restructure.
4. **Move detection onto the Pis.** Decided: Pi-only, no accelerator. The win is resilience — today, if the network or `irm-pc` is unreachable the sign goes blind. Local inference at ~5 fps keeps it warning drivers regardless. This is a restructure, not an addition: local detection removes the per-frame JPEG upload, so the pipeline gets *lighter* even with inference added.
5. **Failure alerting and a chain watchdog.** The sign's timeout covers a dead Pi; nothing covers a live Pi whose camera silently stopped.
6. **Publish the LED driver as its own repository.** There is no working public driver for 1/8-scan FM6124 panels, and the measurement method is the reusable part.

---

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

# 4. Train (YOLO26, not YOLOv8 — the old flat `--model n --epochs N` CLI is gone)
python train.py --dataset vehicle  --model-size n --epochs 100
python train.py --dataset accident --model-size n --epochs 100   # never run; dataset is ready
```

Output: `models/runs/<vehicle|accident>/<dataset>_yolo26<size>_<timestamp>/weights/{best,last}.pt`

### Running the full stack locally

**One command (Windows):** `start.bat` — starts Docker if needed, brings up MySQL + Adminer, creates missing `.env` files, installs Node dependencies, launches all three services and prints every URL and login.

**Or manually:**

```bash
docker compose up -d              # MySQL :3307, Adminer :8080

cd server/ai-service
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt   # or requirements-cpu.txt on CPU-only machines
cp .env.example .env              # set TRAFFIC_MODEL_PATH to your trained weight
python -m app.main                # http://localhost:8000

cd server/node-service
npm install && cp .env.example .env   # set DB_*, JWT_SECRET, ADMIN_PASSWORD, PI_AGENT_TOKEN
npm run dev                       # http://localhost:3001

cd client/web
npm install && npm run dev        # http://localhost:3000
```

Log into the Admin Terminal with the `ADMIN_PASSWORD` from `server/node-service/.env`. The public status page is at `/status`.

### Deploying to the Raspberry Pis

```bash
./raspi_scripts/setup_pi4.sh      # Camera A + ESP32 sign
./raspi_scripts/setup_pi5.sh      # Camera B + STM32 sign
```

Installs packages, venv, scripts, three systemd services and the udev rule that pins `/dev/roadsentinel-sign`. See [`raspi_scripts/README.md`](raspi_scripts/README.md).

### Flashing a sign

```bash
pio run -d LEDMatrixDrivers/esp32 -t upload      # Pi 4 sign
pio run -d LEDMatrixDrivers/stm32 -t upload      # Pi 5 sign
```

Read the port's `WIRING.md` first — the STM32 in particular has a temperature-marginal DFU bootloader with real gotchas.

### Testing

```bash
cd testing
python test_ai.py                 # health, stats, /api/detect*
python test_camera.py             # live camera + optional AI overlay
python test_video.py path/to/video.mp4
```

`training/validate.py` tests a weights file directly, no server needed.

### Optional Pi-side flags (`camera_sender.py`)

```bash
--record --record-dir ./recordings
--ir-auto --onvif-port 80 --onvif-user U --onvif-pass P
--no-adaptive-sampling
```

---

## 📋 Prerequisites

**Hardware** — 8GB RAM minimum (16GB+ recommended); a GPU for training (this project used an RTX 3060 Ti); two Raspberry Pis (4 and 5), each with a camera, a microcontroller and a HUB75 LED sign; a 5V 8A supply per sign.

**Software** — Python 3.9–3.12 for `training/` (3.10–3.12 for the AI service); Node.js 18+; MySQL 8.0 bound to localhost only, never exposed publicly — not even through Tailscale. `migrate.ts` is the authoritative, idempotent schema; point a fresh empty database at it and start the server. PlatformIO for the sign firmware.

---

## 🔧 Troubleshooting

**`YOLO is not exported from module 'ultralytics'`** — `pip install ultralytics`; the correct import is `from ultralytics import YOLO`.

**CUDA out of memory during training** — `python train.py --dataset vehicle --model-size n --batch 4`.

**AI service falls back to `yolov8n.pt`** — check `TRAFFIC_MODEL_PATH` in `server/ai-service/.env` resolves to a real file (relative paths resolve against `server/ai-service/`, not your shell's CWD). The startup line `Traffic detector ready — custom_model=...` reporting `False` means it fell back. `GET /api/stats` reports live load state per model.

**Node logs "Database connection failed"** — the server degrades gracefully rather than crashing, but every DB-backed feature will 500 until connectivity returns. Check `DB_HOST`/`DB_PORT`/credentials and that the host resolves before assuming a code bug.

**The sign is dark** — `ls -l /dev/roadsentinel-sign` (udev matched?), then `journalctl -u roadsentinel-display -n 50`. The bridge logs the board's `INFO` string on every connect, so the log already records which firmware and mapping were live. On the STM32, a missing symlink usually means the board is sitting in its DFU bootloader — tap NRST.

---

## ⚠️ Unverified — do not treat as working

- **Pi 5 STM32 sign** — compiles; never flashed to a board
- **ESP32 sign since the `LEDMatrixDrivers` restructure** — behaviour-identical and compiles, but not re-flashed
- **The bridge's udev rule, connect handshake and liveness check** — none have run on a Pi
- **Camera reachability on both Pis**, including Camera B's auto-discovery recovery path persisting a new IP in practice
- **Sustained live-feed frame rate under real camera load** — the architecture was audited as sound, but delivered FPS has never been measured. `components/video-feed.tsx` already has the instrumentation

---

## 📖 Further documentation

| Doc | Covers |
|---|---|
| [`docs/FUTURE_IMPROVEMENTS.md`](docs/FUTURE_IMPROVEMENTS.md) | **What is left, as a checklist** |
| [`docs/documentation.md`](docs/documentation.md) | Ground-truth audit of the whole codebase |
| [`docs/Summarization.md`](docs/Summarization.md) | Phase-by-phase revamp record |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | Folder-by-folder layout |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Hosting, Tailscale, tunnels |
| [`LEDMatrixDrivers/README.md`](LEDMatrixDrivers/README.md) | The sign driver: design, protocol, roadmap |
| [`LEDMatrixDrivers/esp32/DEBUG_LOG.md`](LEDMatrixDrivers/esp32/DEBUG_LOG.md) | How the panel was reverse-engineered, and two wrong turns |
| [`raspi_scripts/README.md`](raspi_scripts/README.md) | Pi-side services and diagnostics |
| [`client/web/DESIGN.md`](client/web/DESIGN.md) | "Night Watch" design system |
| `docs/LED_TROUBLESHOOTING.md` | Historical LED log — superseded, kept for the trail |

## 📝 License

Uses YOLO26/Ultralytics (AGPL-3.0) and PyTorch (BSD). For thesis and educational use.

---

**Good luck with your Busay blind curve system! 🎓🚗**
