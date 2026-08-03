# Road Sentinel — Codebase Documentation

Audit date: 2026-08-03. Branch `main` @ `ed6c0fd`. This document describes what is **actually implemented** in this repository today, verified by reading source files directly — not the aspirational pitch in `README.md`. Every existing top-level doc (`README.md`, `PROJECT_STRUCTURE.md`, `START_HERE.md`, `TRAINING_GUIDE.md`, `CAMERA_TEST_GUIDE.md`) describes an older `scripts/training/` + `scripts/download/` layout that **no longer matches the tracked repo**. See [§15 Doc Drift Log](#15-doc-drift-log) for the full list of stale claims.

---

## 1. System overview

Road Sentinel is a two-camera traffic-monitoring rig for a blind curve at Barangay Busay, Cebu. As actually implemented:

- Two Raspberry Pis each own one IP camera. **Pi 4** runs Camera A only (`CAM-A-001`, no LED). **Pi 5** runs Camera B (`CAM-B-002`) **and** drives a 128×32 HUB75 RGB LED matrix that shows status text (`raspi_scripts/setup_pi4.sh`, `raspi_scripts/setup_pi5.sh`).
- On each Pi, `raspi_scripts/camera/camera_sender.py` pulls the RTSP stream with OpenCV, JPEG-encodes frames, and POSTs them to the FastAPI **AI service** (`server/ai-service`) at `/api/detect`, passing the camera's `pixels_per_meter` and `speed_limit` so the AI service can estimate speed and auto-generate speeding incidents.
- The AI service runs two YOLOv8/YOLO26 models — `TrafficDetector` (vehicle detection + an in-process IoU tracker for speed) and `IncidentDetector` (crash/incident detection) — and returns detections + incidents as JSON (`server/ai-service/app/main.py`, `app/models/traffic_detector.py`, `app/models/incident_detector.py`).
- `camera_sender.py` forwards those results to the **Node service** (`server/node-service`) via `POST /api/detections` and `POST /api/incidents`, which persist to MySQL and broadcast over Socket.IO. It also pushes the raw JPEG to Node's in-memory frame buffer for live viewing — via a Socket.IO `pi_frame` event (primary, zero-HTTP-round-trip path added in commit `ed6c0fd`) with an HTTP `PUT /api/cameras/:id/frame` fallback.
- The **Next.js client** (`client/web`) subscribes to Node's Socket.IO server for live detections/incidents/camera-frame binary streams and renders a live monitor, incident feed, analytics dashboards, and camera configuration screens.
- Only **one** of the two planned models is actually trained: a YOLO26n vehicle detector (`models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/best.pt`). No crash/incident model has been trained anywhere in the repo — `IncidentDetector` therefore runs in a brightness-variance **heuristic fallback** in practice unless a real `incident.pt` is supplied (see §13).
- The LED display side (`raspi_scripts/display_manager.py`) is a separate, actively-worked subsystem (most of the last ~35 commits are LED timing/driver fixes) that renders REAL/TEST alert screens to the physical panel on Pi 5.

This is a functioning push-based pipeline (Pi → AI service → Node → MySQL/WebSocket → client), not the RTSP-pull-into-Node architecture the docs describe.

---

## 2. Architecture diagram

```
┌───────────────────┐        RTSP (H.264)        ┌───────────────────┐
│  IP Camera A       │ ─────────────────────────▶ │   Raspberry Pi 4   │
│  192.168.8.104:554 │                             │  camera_sender.py  │
└───────────────────┘                             │  pi_agent.py        │
                                                    └─────────┬──────────┘
┌───────────────────┐        RTSP (H.264)                    │
│  IP Camera B       │ ─────────────────────────▶ ┌──────────┴──────────┐
│  192.168.8.108:554 │  (seed.ts default:.102)     │   Raspberry Pi 5     │
└───────────────────┘                             │  camera_sender.py    │
                                                    │  pi_agent.py         │
                                                    │  display_manager.py  │
                                                    │  → HUB75 128×32 LED  │
                                                    └──────────┬───────────┘
                                                               │
                          JPEG POST /api/detect (multipart)    │  Socket.IO: pi_register,
                          pixels_per_meter, speed_limit         │  pi_frame, pi_command/pi_output
                                                               ▼
                                              ┌─────────────────────────────┐
                                              │  AI Service — FastAPI :8000  │
                                              │  TrafficDetector (YOLOv8/26) │
                                              │  IncidentDetector            │
                                              │  local media storage (/media)│
                                              └───────────────┬───────────────┘
                                        JSON {detections,incidents}
                                                               │
                    POST /api/detections, /api/incidents       │
                    (from camera_sender.py, not from Node)      ▼
                                              ┌─────────────────────────────┐
                                              │  Node Service — Express :3001│
                                              │  Socket.IO (same port)       │
                                              │  MySQL pool (Aiven)          │
                                              │  Admin terminal (spawn shell)│
                                              └───────────────┬───────────────┘
                                       SQL (mysql2)             │ WebSocket:
                                                               │ detection, incident,
                                                               ▼ camera_status, camera_frame:<id>
                                              ┌─────────────────────────────┐
                                              │  MySQL (Aiven)                │
                                              │  cameras, detections,          │
                                              │  incidents, hourly_analytics   │
                                              └───────────────────────────────┘

                                              ┌─────────────────────────────┐
                                              │  Next.js Client :3000         │
                                              │  monitor/incidents/analytics  │
                                              │  admin terminal (no auth)     │
                                              └───────────────────────────────┘
```

Notes on protocols/ports, all confirmed from code:
- AI service default port `8000` (`server/ai-service/app/main.py:340`, `.env`).
- Node service default port `3001`, Socket.IO shares the same HTTP server (`server/node-service/src/server.ts:24-35`).
- Next.js client default port `3000` (Next.js default; `NEXT_PUBLIC_API_URL` in client points at Node).
- MySQL via `mysql2` pool, optional TLS when `DB_SSL=true` (`server/node-service/src/config/database.ts:24-29`).

---

## 3. Repo structure (current, real)

| Path | Responsibility |
|---|---|
| `training/` | YOLO26 training scripts, dataset merge/convert/analyze tools (Python) |
| `testing/` | Standalone scripts that exercise the AI service over HTTP: camera, video, image, visual-GUI, API smoke tests |
| `inference/` | Two **standalone prototype** speed-detection classes (`speed_detection.py`, `camera_calibration.py`) + a batch frame extractor — not imported by the production server |
| `server/ai-service/` | FastAPI microservice; loads the two YOLO models, exposes `/api/detect*`, local file storage under `/media` |
| `server/node-service/` | Express + Socket.IO backend; MySQL access, REST API, WebSocket broadcast, admin shell terminal |
| `server/database/` | Static reference `mysql_schema.sql` (schema is **also** independently defined and actually applied by `node-service/src/database/migrate.ts` — the two differ, see §7 and §14) |
| `client/web/` | Next.js 15 / React 18 dashboard (HeroUI component library) |
| `raspi_scripts/` | Everything that runs on the two Raspberry Pis: camera capture/forwarding, LED matrix driver, Pi-side remote-terminal agent, setup scripts |
| `models/` | `README.md` (describes a `v1/v2/production` layout that does not exist) + the real output tree `models/runs/<dataset>/<run_name>/weights/{best,last,epochNN}.pt`, only a `vehicle` run present |
| `datasets/` | `downloaded/` (raw Roboflow export, untracked) and `processed/busay_vehicle_detection/`, `processed/busay_accident_detection/` (untracked, gitignored) |
| `config.yml` | Empty (`{}`) — not read by any code in the repo |
| `emulator_config.json` / `raspi_scripts/emulator_config.json` / `raspi_scripts/emulator.cfg` | Config files for the third-party `RGBMatrixEmulator` pip package (auto-discovered by filename convention, not referenced anywhere in this repo's own code) |
| `.claude/` | Local Claude Code permission settings only (`settings.local.json`) — no application content |
| `render.env.txt` | Untracked (gitignored) copy-paste template of Render.com deployment env vars — **contains live plaintext production DB and Supabase credentials**, see §14 |

`PROJECT_STRUCTURE.md` describes a completely different top-level layout (`scripts/training/`, `scripts/download/`, `scripts/extract_frames/`, `models/v1/v2/production/`). That layout is **not** what's tracked in git. It does still exist as an untracked, gitignored leftover directory (`scripts/`) containing only old venvs, `__pycache__`, and stale training-run artifacts — **zero `.py` source files remain in it**. See §15.

---

## 4. Training pipeline (`training/`)

| Script | Purpose | Status |
|---|---|---|
| `train.py` (441 lines) | Real, current trainer. `python train.py --dataset {vehicle,accident,both} --model-size {n,s,m,l,x} --epochs N`. Loads `yolo26<size>.pt`, auto-picks batch size from detected VRAM (tuned for an 8GB RTX 3060 Ti, `train.py:66-73`), writes to `models/runs/<dataset>/<dataset>_yolo26<size>_<timestamp>/`. | Functional — this produced the one existing trained run. **Bug**: `DATASETS_DIR` is computed as `Path(__file__).parent.parent.parent / "Road_Sentinel" / "datasets" / "processed"` (`train.py:47-48`), i.e. it assumes the repo's *parent* directory is literally named `Road_Sentinel`. On this checkout (folder named `RoadSentinel`, not `Road_Sentinel`) that path does not exist unless overridden. |
| `validate.py` (169 lines) | Despite the name, this is **not** a metrics-validation script — it's a prediction/testing CLI (`test_on_video`, `test_on_images`) using `model.predict()`, no `model.val()` call anywhere. Misleading filename. | Functional as a test-inference script |
| `merge_busay_datasets.py` (434 lines) | `merge_vehicle_detection_datasets()` merges "Traffic Surveillance" + "Day/Night" Roboflow exports into one 5-class vehicle set with explicit ID remapping tables; `prepare_accident_dataset()` copies the accident dataset through. | Functional |
| `run_merge_busay.py` (227 lines) | Interactive wrapper: searches known paths (including a hardcoded `/home/user/Road_Sentinel/datasets/downloaded`) for the three named Roboflow datasets, prints a merge plan, prompts `y/n`, calls the merge functions above. | Functional, but its own "next steps" printout still references `train_vehicle_detector.py` (`run_merge_busay.py:193,208`), a script that does not exist in this repo — only `train.py` does |
| `analyze_datasets.py` (193 lines) | Walks a directory tree for `data.yaml` files, prints per-dataset image counts and class lists for comparison. | Functional |
| `convert_aicity_track1_to_yolo.py` / `convert_aicity_track4_to_yolo.py` | Convert AI City Challenge Track 1 (multi-camera vehicle tracking) / Track 4 (traffic anomaly) datasets to YOLO format. | Present, standalone, not wired into `train.py`'s `DATASETS` dict |
| `download_roboflow_datasets.py` | Thin wrapper around the `roboflow` SDK to pull a dataset by workspace/project/version. | Functional utility |
| `download_test_video.py` | Downloads 3 fixed Pexels stock traffic videos into `test_videos/` for manual testing. | Functional utility, unrelated to Busay footage |

`training/README.md` itself is stale relative to the scripts that actually exist in the same folder — it documents `train_vehicle_detector.py` and `quick_train.py`, neither of which is present; the real files are `train.py` and `validate.py`. `README.md`'s claim that "`quick_train.py` removed — use `train.py` instead" (README.md:198) is the one place the top-level docs correctly acknowledge drift.

---

## 5. Testing suite (`testing/`)

All scripts in this folder talk to the AI service over plain HTTP at `http://localhost:8000` — none of them import the trained model directly.

| Script | Exercises |
|---|---|
| `test_ai.py` (175 lines) | Smoke test: `/health`, `/api/stats`, downloads a sample Unsplash image, posts it to `/api/detect`, `/api/detect/traffic`, `/api/detect/incidents` |
| `test_camera.py` (508 lines) | Scans local camera indices 0–4 (`cv2.VideoCapture(i, cv2.CAP_DSHOW)`), live preview with FPS overlay, optional per-frame AI calls, interactive Q/SPACE/C controls |
| `test_camera_native.py` (49 lines) | Minimal native-resolution camera-open test for diagnosing Windows camera issues (no AI) |
| `test_images.py` (195 lines) | Posts a single image or folder of images to `/api/detect`, optionally saves annotated output |
| `test_video.py` (321 lines) | Extracts frames from a video file at a configurable stride and posts each to `/api/detect` |
| `test_visual.py` (632 lines) | Threaded real-time GUI: captures from file or camera, runs detection on a background worker thread, overlays boxes/stats live |

`testing/README.md` is present in git but **empty** (0 bytes).

---

## 6. Inference pipeline (`inference/`)

This folder is **not** the code path the production server uses. It is two self-contained, older prototype classes plus a batch utility.

- **`speed_detection.py`** — `VehicleSpeedDetector` class. Loads the plain pretrained **`yolov8n.pt`** COCO model (not a Busay-trained weight, not YOLO26). Tracks with `model.track(persist=True, classes=[1,2,3,5,7])`, computes speed from raw pixel displacement between frame centers divided by a single scalar `ppm` (pixels-per-meter) supplied once at startup via an interactive two-click calibration (`calibrate_camera()`). No incident/crash detection at all — this script only estimates vehicle speed. `__main__` hardcodes `test_video.mp4` / `traffic_video.mp4` filenames that are not present in the repo.
- **`camera_calibration.py`** — `AngledCameraSpeedDetector` class. Same `yolov8n.pt` model and tracker, but replaces the linear ppm calculation with an actual **homography** (`cv2.getPerspectiveTransform` from 4 user-clicked road points to a bird's-eye rectangle, `calibrate_perspective()`), then transforms track points through the homography before computing speed (`transform_point()`, `calculate_speed()`). This is the only place in the whole repo that implements true perspective-corrected speed estimation; the production AI service (`traffic_detector.py`) uses simple raw-pixel-distance IoU tracking instead (see §7), not homography.
- **`extract_frames.py`** — Batch frame extractor: given a folder or single video, extracts frames at a target FPS with `cv2.imwrite(..., JPEG_QUALITY=95)`, has a CLI (`argparse`) distinct from `training/download_test_video.py`.

`inference/README.md` exists in git but is **empty** (0 bytes).

---

## 7. Server

### 7.1 `server/ai-service` (FastAPI, Python)

Entry point: `app/main.py`. Endpoints (all confirmed in `app/main.py`):

| Method & path | Purpose |
|---|---|
| `GET /` | Service banner |
| `GET /health` | `{status, timestamp}` |
| `POST /api/detect` | Runs both detectors on one image. Accepts `image`, `camera_id`, optional `confidence_threshold`, `pixels_per_meter`, `speed_limit`. When `pixels_per_meter > 0`, speed is computed via a per-camera IoU tracker inside `TrafficDetector`; when `speed_limit > 0`, any detection whose estimated speed exceeds it is turned into a synthesized `"speeding"` incident (`main.py:139-153`) with severity escalating at +15/+30 km/h over the limit. |
| `POST /api/detect/traffic` | Traffic-only detection |
| `POST /api/detect/incidents` | Incident-only detection |
| `POST /api/storage/upload` | Saves a file under `MEDIA_DIR` (default `./media`), returns a public URL built from `STORAGE_BASE_URL` (a Cloudflare Tunnel URL) or `http://localhost:PORT`; path-traversal (`..`) rejected |
| `DELETE /api/storage/delete` | Deletes a file under `MEDIA_DIR` |
| `GET /api/storage/list` | Lists stored files under an optional prefix |
| `GET /api/stats` | Reports whether each model is loaded, its configured path/device, and thresholds |

**Model loading** (`app/models/traffic_detector.py`, `app/models/incident_detector.py`):
- `TrafficDetector(model_path, device, confidence)` loads `TRAFFIC_MODEL_PATH` via `ultralytics.YOLO`. If it fails to load, it **silently falls back** to the stock `yolov8n.pt` and filters to COCO vehicle classes `{1,2,3,5,7}` (bicycle/car/motorcycle/bus/truck). It auto-detects "custom model" mode by checking whether class 0's name looks like a vehicle type.
- Speed estimation is a **per-camera in-memory IoU tracker** (`_trackers: Dict[camera_id, Dict[track_id, {bbox,time,class}]]`), matched by IoU > 0.25, tracks pruned after 2s idle (`TRACK_TTL`). Speed = center-to-center pixel distance ÷ `pixels_per_meter` ÷ Δt × 3.6 (`traffic_detector.py:63-122`). This is simple, un-corrected pixel-distance speed — **not** the homography approach implemented in `inference/camera_calibration.py`.
- `IncidentDetector(model_path, device, confidence)` loads `INCIDENT_MODEL_PATH`. If loading fails, `self.model = None` and `detect()` falls back to `_heuristic_detection()`: a placeholder that flags "congestion" purely from grayscale brightness variance (`incident_detector.py:118-161`) — explicitly commented in the source as "a simplified example."
- **In the checked-out `.env`** (untracked, not committed): `TRAFFIC_MODEL_PATH` points at the real trained weight `models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/best.pt` via a **hardcoded absolute Windows path** on a different drive/root (`C:\Projects\Thesis\2026\RoadSentinel\...`) than this checkout's own working directory. `INCIDENT_MODEL_PATH` is still the placeholder `./models/incident.pt`, which does not exist anywhere in the repo — so in practice **incident detection always runs in heuristic fallback mode**, never the real model, because no crash/incident model has ever been trained. `DEVICE=cpu` in this `.env` despite GPU hardware being referenced throughout the docs.

Inference cadence is not fixed server-side; it's driven by whatever rate the caller (`camera_sender.py`) posts frames at (JPEG_QUALITY=50, target 30 FPS capture but detection calls are throttled client-side, see §9).

### 7.2 `server/node-service` (Express + Socket.IO, TypeScript)

Entry point: `src/server.ts`. REST routes:

| Route file | Endpoints |
|---|---|
| `routes/cameras.ts` | `GET /`, `GET /:id`, `PUT /:id` (settings), `PUT /:id/status`, `PUT /:id/frame` (raw JPEG ingest, MJPEG buffer), `GET /:id/stream` (multipart MJPEG) |
| `routes/detections.ts` | `GET /` (filterable by camera/since), `POST /` (insert + broadcast `detection` event) |
| `routes/incidents.ts` | `GET /`, `GET /:id`, `POST /` (insert + broadcast `incident` event to camera room and global `incidents` room), `PUT /:id/status` |
| `routes/analytics.ts` | `GET /summary`, `GET /hourly`, `GET /speed` (histogram buckets) |

Socket.IO events (`src/server.ts:88-274`), all on the same port as the HTTP server:
- Camera/incident subscriptions: `subscribe_camera`, `unsubscribe_camera`, `subscribe_stream`/`unsubscribe_stream` (binary frame relay `camera_frame:<id>`), `subscribe_incidents`.
- Pi ingest: `pi_frame` (binary JPEG straight from a Pi, routed to `handlePiFrame()` in `routes/cameras.ts`), `pi_register` (Pi agent announces itself online), `pi_output` (Pi streams command stdout/stderr back to the requesting admin socket).
- **Admin remote shell** (`subscribe_admin`, `terminal_command`, `terminal_kill`): when `target === "server"`, the Node process itself calls `child_process.spawn(shell, [flag, command], {cwd: process.cwd(), env: process.env})` (`server.ts:222`) and streams stdout/stderr back over the socket. When `target` is `"pi4"`/`"pi5"`, the command is relayed over Socket.IO to that Pi's `pi_agent.py`, which runs it with `subprocess.Popen(["sh","-c",command], ...)` and streams output back. **There is no authentication anywhere in this path** — see §14.

`src/services/ai.service.ts` is a thin Axios client for the AI service's `/health`, `/api/detect*`, `/api/stats` — used for the startup health check in `server.ts`, not actually invoked by any route to proxy frames (camera_sender.py talks to the AI service directly).

`src/services/storage.service.ts` uploads/deletes files through the **AI service's** `/api/storage/*` endpoints (local-PC storage behind a Cloudflare Tunnel) — **not** Supabase. `src/config/supabase.ts` is an explicit no-op stub (`"Supabase has been replaced by PC local storage... This stub keeps server.ts unchanged"`, `supabase.ts:3-8`) that just logs a message; the `@supabase/supabase-js` dependency in `package.json` is unused dead weight.

`src/database/seed.ts` seeds two fixed cameras on every startup: `CAM-A-001` (rtsp `192.168.8.104`) and `CAM-B-002` (rtsp `192.168.8.102`, from `CAM_B_RTSP` default) — note this **conflicts** with `raspi_scripts/setup_pi5.sh`'s default of `192.168.8.108` for Camera B (see §14).

### 7.3 `server/database`

`server/database/mysql_schema.sql` is a **static reference file**, not what actually runs. The schema that is actually applied at every Node service startup lives in code: `server/node-service/src/database/migrate.ts` → `runMigrations()`, called from `startServer()` in `server.ts:321`. The two differ:

| Table | In `mysql_schema.sql` | In `migrate.ts` (what actually runs) |
|---|---|---|
| `cameras` | ✅ (`VARCHAR(36)` id, `pixels_per_meter` default 25.5) | ✅ (`VARCHAR(50)` id, `pixels_per_meter` default 8.0) |
| `detections` | ✅ | ✅ (matches closely) |
| `incidents` | ✅ | ✅ (matches closely) |
| Hourly analytics table | `analytics_hourly` | `hourly_analytics` — **different table name** |
| `recordings` | ✅ defined, with `video_url`/`thumbnail_url`/`vehicle_count` etc. | ❌ **never created** — no `CREATE TABLE recordings` in the migrations array at all |

Both define: `cameras` (id, name, location, rtsp_url, status enum, fps, resolution, pixels_per_meter, speed_limit, detection_confidence), `detections` (FK → cameras, vehicle_type enum, speed, confidence, bbox_x/y/width/height, direction, lane_number), `incidents` (FK → cameras, incident_type enum, severity enum, title, description, image_url, video_url, status enum, resolved_at/by, notes, metadata JSON), and an hourly rollup table (total_vehicles, avg/max/min_speed, per-type counts, incident_count, speeding_violations, peak_flow_minute).

Which service writes/reads: only `node-service` touches MySQL directly (via `mysql2` pool in `config/database.ts`); the AI service and the Pi scripts never connect to the database — they go through Node's REST API.

---

## 8. Client (`client/web`, Next.js 15 + React 18 + HeroUI)

Base template is HeroUI's "next-app-template" starter (`package.json:2` — `"name": "next-app-template"`, never renamed) with Tailwind CSS via `@heroui/theme`.

### Route inventory

| Route | Status | Notes |
|---|---|---|
| `/` (`app/page.tsx`, 389 lines) | **Live** | Dashboard: fetches `/api/analytics/summary`, `/api/cameras`, `/api/incidents`; live video via `VideoFeed`; subscribes to Socket.IO |
| `/monitor` (324 lines) | **Live** | Grid of live camera feeds (WebSocket binary frames + MJPEG fallback), real-time detection log via Socket.IO `detection` events |
| `/analytics` (360 lines) | **Live** | Calls `/api/analytics/summary`, `/hourly`, `/speed`; vehicle-type breakdown, speed histogram, hourly bar chart. "Export PDF"/"Export CSV" buttons have **no onClick handler** — not implemented |
| `/incidents` (256 lines) | **Live** | Calls `/api/incidents`, filter by status, live-updates via Socket.IO `incident` event, resolve/false-alarm/investigate actions call `PUT /api/incidents/:id/status` |
| `/cameras` (332 lines) | **Live** | Lists cameras, edit form calls `PUT /api/cameras/:id`, "Test Connection" calls `GET /api/cameras/:id`. "Open Calibration Tool" / "View Calibration Guide" buttons are **decorative, no handler** |
| `/admin` (473 lines) | **Live, unauthenticated** | Full remote-shell terminal against the server process and both Pis over Socket.IO (see §7.2, §14) |
| `/history` (201 lines) | **Fully hardcoded/placeholder** | `recordings` array is a literal in-file constant with fake timestamps/vehicle counts; no `fetch` call anywhere in the file; "Play" buttons have no handler. Matches the fact that the `recordings` table is never created (§7.3) |
| `/reports` (86 lines) | **Fully hardcoded/placeholder** | `recentReports` array is a literal constant; "Download" button has no handler |
| `/settings` (71 lines) | **Fully hardcoded/placeholder** | Switches have no `onChange`/state; "Save All Settings" / "Reset to Defaults" have no handlers |
| `/about`, `/blog`, `/pricing`, `/docs` | **Unused template boilerplate** | 9–13 lines each, e.g. `app/about/page.tsx` is just `<h1>About</h1>`. Leftover from the HeroUI starter template, not wired to anything Road-Sentinel-specific |

Note: `config/site.ts`'s `navItems`/`navMenuItems` (used for a top navbar, apparently unused in the actual rendered layout) do **not** list `/admin` at all, but the sidebar that's actually rendered (`components/sidebar.tsx:177-195`, wired into every page via `app/layout.tsx`) **does** link "Admin Terminal" → `/admin` directly in the primary nav, with no gating.

### Components

`components/video-feed.tsx` (308 lines) — the live-view player: subscribes to Socket.IO event `camera_frame:<cameraId>`, renders binary JPEG blobs to an `<img>`, falls back to the MJPEG `<img src={streamUrl}>` if no WS frame arrives within 5s (`useMjpegFallback`), measures live FPS/frame-interval client-side, pings every 55s to keep the socket alive through Cloudflare's 100s WS timeout. `components/alert-card.tsx`, `stat-card.tsx`, `camera-status.tsx` are presentational cards; `components/sidebar.tsx` / `navbar.tsx` are navigation (sidebar is what's actually rendered per `app/layout.tsx:46`); `components/animated-background.tsx` is a decorative gradient background.

### Data fetching / styling

Plain `fetch()` with `setInterval` polling (10–30s depending on page) plus Socket.IO for push updates — no React Query/SWR/RSC data layer. Styling is Tailwind utility classes with a hand-rolled purple/orange glassmorphism theme (`bg-white/10 backdrop-blur-md`, `#1B1931`/`#ED9E59` accents) applied ad hoc per-page; no shared design-token file beyond `components/primitives.ts` (HeroUI's default `title()`/`subtitle()` helpers, used only by the unused `/about` page).

---

## 9. Raspberry Pi side (`raspi_scripts/`)

Two coexisting camera-launch approaches are present in the repo simultaneously (see §15 for the discrepancy):

**Current/production path** (`setup_pi4.sh`, `setup_pi5.sh`, systemd services):
- `camera/camera_sender.py` (783 lines) — RTSP → OpenCV capture → JPEG (quality 50, target 30 FPS) → `POST {AI_URL}/api/detect` with `pixels_per_meter`/`speed_limit` fetched per-camera from Node (`fetch_camera_config()`) → forwards results to Node `POST /api/detections` / `/api/incidents` (deduping repeat incident types within a 30s window) → pushes frames to Node primarily via Socket.IO `pi_frame` emit, HTTP `PUT` as fallback. Includes ONVIF WS-Discovery + RTSP port-scanning **auto-discovery** logic if the configured RTSP URL fails repeatedly (`DISCOVERY_AFTER_FAILURES = 3`).
- `pi_agent.py` (177 lines) — connects outbound to the Node service over `python-socketio`, registers as `pi4`/`pi5` (`pi_register`), and on `pi_command` runs `subprocess.Popen(["sh","-c",command], preexec_fn=os.setsid)`, streaming stdout/stderr back via `pi_output`; `pi_kill` sends `SIGINT` to the process group. No inbound ports needed on the Pi. This is the backend for the Admin Terminal's "pi4"/"pi5" targets (§7.2, §8).
- `display_manager.py` (1318 lines, top-level) — the **unified**, currently-maintained LED driver. Auto-detects Pi 4 vs Pi 5 (`/dev/pio0` presence) and picks a backend: Pi 4 → `ledcat` subprocess over `/dev/mem` (hzeller C lib, needs `sudo`); Pi 5 → `led-image-viewer` subprocess (SwapOnVSync/coprocessor mode). Has a documented, currently-disabled RGBMatrixBackend Python-bindings path for Pi 5 with an explicit `# TODO: fix pixel mapping before re-enabling` (`display_manager.py:636`). Renders REAL/TEST severity-colored status screens (`SEVERITY_COLORS` for critical/high/medium/low).
- `setup_pi4.sh` / `setup_pi5.sh` — install scripts; Pi 4 installs only `roadsentinel-camera` + `roadsentinel-agent`; Pi 5 additionally installs `roadsentinel-display`. Default `NODE_URL=http://192.168.8.50:3001`, `AI_URL=http://192.168.8.50:8000`.
- `color_test.py`, `test_display.py` — hardware bring-up/diagnostic scripts for the LED panel (color cycling, auto-cycling status screens to keep the RP1 refresh thread "fresh" and avoid PWM timing drift — this exact issue dominates the last ~35 git commits, see §16).

**Legacy/parallel path** (`camera_reboot_autostart_setup.sh`, root of repo):
- A one-time installer that wires two **`ffplay`**-based desktop preview windows into the Pi's desktop-session autostart (`~/.config/autostart/roadsentinel-cameras.desktop`), independent of `camera_sender.py`/systemd. Also runs `set_ir_auto_all.py` (ONVIF, day/night IR switching) before launching the streams. Uses **different** hardcoded camera IPs (`192.168.8.104` / `192.168.8.108`) than `node-service`'s seeded default for Camera B (`192.168.8.102`, see §14).

**LED subfolders** `lcd/` (Pi 5, Adafruit PioMatter) and `lcd_pi4/` (Pi 4, hzeller rpi-rgb-led-matrix, build-from-source, needs `sudo`) contain earlier per-model `display_manager.py` implementations, each with trivial/placeholder git commit messages (`"123"`, `"789"`, etc.) predating the unified top-level `display_manager.py`. `raspi_scripts/README.md` still presents the `lcd/` vs `lcd_pi4/` split as the current setup path without mentioning the newer unified driver.

---

## 10. Models (`models/`)

- `models/README.md` describes a `v1/`, `v2/`, `production/` (symlinks) layout with a `train_vehicle_detector.py` script. **None of that exists.**
- What's actually present: `models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/{best.pt, last.pt, epoch0.pt, epoch10.pt, …, epoch90.pt}` — one completed YOLO26n training run for **vehicle detection only**, produced by `training/train.py`, plus its `args.yaml`, PR/F1/confusion-matrix curve PNGs, and `results.csv`.
- **No crash/accident/incident model weights exist anywhere in the repository** — not under `models/`, not under the legacy `runs/segment/` or `runs/detect/` trees (see below), not gitignored-but-present on disk. `training/train.py --dataset accident` has apparently never been run to completion (or its output was deleted).
- Two additional, older/legacy training-output trees exist at the repo root (both untracked/gitignored): `runs/detect/runs/vehicle_speed/{busay_v1, busay_vehicle_v1_custom}/` and `runs/segment/{train, train-2, train-3}/weights/{best,last}.pt` — the `segment/train-2` weights are the only other trained `.pt` files in the repo, from a YOLO **segmentation** task unrelated to any script currently in `training/` (no segmentation training code exists in the tracked `training/` folder).
- A stray `yolov8n.pt` (6.5 MB, pretrained COCO weights) sits at the repo root — used by `inference/speed_detection.py` and `inference/camera_calibration.py`, and as the AI service's fallback when a configured model path fails to load.

---

## 11. Datasets (`datasets/`)

- `datasets/downloaded/` (untracked, gitignored) contains one Roboflow export in place: `train/`, `test/`, `valid/` folders, `data.yaml`, `README.roboflow.txt` dated 2023-10-20. `train/images/` has 1,595 files. Only one of the three datasets the docs describe (Traffic Surveillance, Day/Night, Accident Detection) is actually present at the top level of `downloaded/` — no subfolders matching those three names exist; the single dataset here appears to already be pre-merged/pre-placed rather than sitting in the three-way "place your downloads here" structure `START_HERE.md` describes.
- `datasets/processed/busay_vehicle_detection/` and `datasets/processed/busay_accident_detection/` (both untracked, gitignored) **do** exist with `data.yaml` files and `train/valid/test` subfolders — i.e. the merge step described in `training/run_merge_busay.py` has been run and both processed datasets are ready, even though only the vehicle one has been trained on (§10).
- `datasets/README.md` and `datasets/processed/.gitkeep` are the only tracked files in this tree.

---

## 12. Config & environment

| Key / file | Read by | Effect |
|---|---|---|
| `config.yml` (root, `{}`)  | *(nothing)* | Empty placeholder, not referenced by any `.py`/`.ts`/`.js`/`.sh` file in the repo |
| `emulator_config.json` (root and `raspi_scripts/`), `raspi_scripts/emulator.cfg` | The third-party `RGBMatrixEmulator` pip package, by filename convention | Configures the browser-based HUB75 emulator (pixel size/style, target FPS, Pi 5 pinout/plane settings) for testing the LED code without physical hardware — not read by any code authored in this repo |
| `server/ai-service/.env` (untracked; `.env.example` tracked) | `app/main.py` via `os.getenv` | `HOST`, `PORT`, `WORKERS`, `TRAFFIC_MODEL_PATH`, `INCIDENT_MODEL_PATH`, `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `DEVICE`. Live `.env` on this checkout points `TRAFFIC_MODEL_PATH` at the real trained weight via a hardcoded absolute path on a different drive than this checkout; `INCIDENT_MODEL_PATH` is unresolved (§7.1) |
| `server/node-service/.env` (untracked; `.env.example` tracked) | `src/config/database.ts`, `src/server.ts`, `src/services/*` | `DB_HOST/PORT/USER/PASSWORD/NAME/SSL` (Aiven MySQL), `SUPABASE_*` (declared but unused — see §7.2), `AI_SERVICE_URL`, `AI_SERVICE_TIMEOUT`, `FRAME_PROCESSING_RATE`/`VIDEO_RECORDING_ENABLED`/`MAX_RECONNECT_ATTEMPTS` (declared in `.env.example` but **not referenced anywhere** in `src/`), `CORS_ORIGIN` (declared but Node actually hardcodes `cors({origin: "*"})` in `server.ts:38`, ignoring this var), `LOG_LEVEL`, `LOG_FILE` (declared but `logger.ts` hardcodes `logs/error.log`/`logs/combined.log`, ignoring `LOG_FILE`) |
| `client/web/.env.local` (untracked, present on disk; not read for this audit) | Next.js build | `NEXT_PUBLIC_API_URL` — points the browser at the Node service |
| `render.env.txt` (root, untracked/gitignored) | Manual copy-paste into Render.com's dashboard | Deployment template for Node + Next.js services on Render; **contains live, plaintext production secrets** (Aiven DB password, Supabase service-role key) committed to a local file — see §14 |
| `raspi_scripts/*setup*.sh` positional args | Bash scripts on the Pi | `NODE_URL`, `CAM_A_RTSP`/`CAM_B_RTSP`, `AI_URL` — defaults point at `192.168.8.50` (server) and `.104`/`.108` (cameras) |

---

## 13. Feature completeness matrix

| Feature | Status | Evidence |
|---|---|---|
| Vehicle detection | **Done** | `server/ai-service/app/models/traffic_detector.py`; real trained weight `models/runs/vehicle/.../best.pt` wired via `.env` |
| Speed estimation | **Done** (simple) | `traffic_detector.py:63-122` per-camera IoU tracker, raw pixel-distance/Δt; a more accurate homography-based version exists but is unused (`inference/camera_calibration.py`) |
| Camera calibration / homography | **Partial** | Real homography implementation exists (`inference/camera_calibration.py`) but is a disconnected standalone script never invoked by the server; production speed math uses uncorrected pixel distance. Client's "Calibration Tool" buttons (`app/cameras/page.tsx:311-327`) are decorative, no handler |
| Crash/anomaly detection | **Stubbed** | `server/ai-service/app/models/incident_detector.py:118-161` — no trained model exists anywhere in the repo (§10); falls back to a brightness-variance heuristic explicitly labeled "simplified example" in source |
| Dual-camera coordination | **Done** | `raspi_scripts/setup_pi4.sh` (Cam A only) / `setup_pi5.sh` (Cam B + LED); `node-service/src/database/seed.ts` seeds both camera rows; independent per-camera IoU trackers keyed by `camera_id` |
| LED warning output | **Done, actively fragile** | `raspi_scripts/display_manager.py`; ~35 of the most recent commits are timing/driver fixes for RP1 PWM drift; one backend explicitly disabled pending a pixel-mapping fix (`display_manager.py:636`) |
| Night vision handling | **Partial** | ONVIF auto-IR switching exists in the legacy `camera_reboot_autostart_setup.sh` path (`set_ir_auto_all.py`) but is not present in the current `camera_sender.py` production path; no IR/day-night logic in the AI models themselves |
| MySQL event logging | **Done** | `node-service/src/database/migrate.ts` (cameras/detections/incidents/hourly_analytics tables); routes insert on every POST from `camera_sender.py` |
| Web client live view | **Done** | `app/monitor/page.tsx`, `components/video-feed.tsx` — WebSocket binary frames with MJPEG fallback |
| Web client history/reporting | **Stubbed (UI shell only)** | `/history` and `/reports` pages are 100% hardcoded fixture data with no `fetch` calls; backing `recordings` table is never created by `migrate.ts` (§7.3) |
| Admin/ops access control | **Not implemented** | No auth on the Socket.IO admin-terminal path anywhere in `node-service` or `client/web` — see §14 |

---

## 14. Known issues / rough edges

1. **Unauthenticated remote command execution.** `server/node-service/src/server.ts:187-274` lets any Socket.IO client run arbitrary shell commands on the Node server process (`child_process.spawn`) and, via `pi_agent.py`, on both physical Raspberry Pis, with **no login, token, or origin check anywhere in the stack** (confirmed: no `middleware.ts`, no auth import in any `node-service` route, no login page in `client/web`). The sidebar navigation links directly to `/admin` for anyone who loads the client (`components/sidebar.tsx:177-195`). CORS is wide open (`cors({ origin: "*" })`, `server.ts:38`).
2. **Plaintext production secrets in a repo-root file.** `render.env.txt` (untracked/gitignored, so not in git history, but present in the working tree) contains a live Aiven MySQL password and a live Supabase service-role key in cleartext.
3. **Hardcoded, machine-specific absolute path.** The active AI-service `.env`'s `TRAFFIC_MODEL_PATH` is a Windows absolute path on a different drive letter/root than this checkout — will break on any other machine or clone.
4. **`training/train.py`'s `DATASETS_DIR` resolution bug** — assumes the repo's parent folder is literally named `Road_Sentinel` (§4); breaks silently to a `Dataset not found` message on any checkout not laid out that way (e.g. this one, `RoadSentinel`).
5. **Duplicate/overlapping logic across `training/`, `testing/`, `inference/`.** Three independent implementations of "run a detector against a video/image and draw boxes" exist (`training/validate.py`, `testing/test_video.py`/`test_images.py`, `inference/speed_detection.py`), with no shared code.
6. **Two incompatible schema sources of truth** for the analytics table name and the `recordings` table (§7.3) — a developer following `server/database/mysql_schema.sql` by hand would end up with a database Node's own migration code doesn't expect.
7. **Two independent, undocumented-as-separate camera-launch mechanisms** on the Pi (`camera_sender.py`/systemd vs. the `ffplay`/desktop-autostart path in `camera_reboot_autostart_setup.sh`) with **different hardcoded IPs for Camera B** (`.108` in the autostart script and `setup_pi5.sh`, vs. `.102` as `node-service`'s seeded DB default) — whichever is stale would silently point Camera B's config at the wrong device.
8. **Declared-but-ignored env vars** in `node-service`: `.env.example` documents `CORS_ORIGIN` and `LOG_FILE`, neither of which the code actually reads (`server.ts` hardcodes `origin: "*"`; `logger.ts` hardcodes its file paths).
9. **Unused dependency surface**: `@supabase/supabase-js`, `node-rtsp-stream`, `fluent-ffmpeg` are all declared in `server/node-service/package.json` but have zero imports anywhere in `src/` (grepped, no matches) — Supabase is an explicit no-op stub, and Node never touches RTSP or ffmpeg directly (that happens on the Pi).
10. **LED display subsystem is the most actively-patched code in the repo** and still has an open, explicitly-flagged bug: the Pi 5 `RGBMatrixBackend` (Python-bindings path) is disabled because `SetImage` mirrors output on chained panels, with a `# TODO: fix pixel mapping before re-enabling` (`display_manager.py:636`) — the currently-used fallback (`led-image-viewer` subprocess) requires periodically restarting the viewer process to avoid RP1 PWM timing drift (multiple `heartbeat`/`auto-cycle` commits).
11. **Confidence threshold defaults disagree across the stack**: `ai-service/.env.example` default `0.75`; the live `.env` on this checkout sets `0.5`; `node-service` seeds cameras with `detection_confidence = 0.5` (`seed.ts`) while `mysql_schema.sql`'s sample INSERT and column default use `0.75`.
12. **`models/runs/segment/`** contains trained YOLO segmentation weights (`runs/segment/train-2/weights/best.pt`) with no corresponding segmentation training code anywhere in tracked `training/` — orphaned artifact from work not represented in the current scripts.

---

## 15. Doc drift log

Concrete, file-by-file claims in the five top-level docs that no longer match the real repo:

**`PROJECT_STRUCTURE.md`**
- Entire documented tree (`scripts/training/`, `scripts/download/`, `scripts/extract_frames/`, `models/v1/`, `models/v2/`, `models/production/`) does not match tracked reality. The real top-level dirs are `training/`, `testing/`, `inference/`, `models/runs/`.
- Lists `train_vehicle_detector.py`, `quick_train.py`, `auto_download_coco.py`, `angled_camera_calibration.py` as the key scripts (lines 115-149) — **none of these files exist anywhere in the tracked repo.** The real equivalents are `training/train.py` and `inference/camera_calibration.py` (a differently-designed, differently-named, standalone class, not a drop-in replacement).
- `.gitignore` recommendations block (lines 271-289) does not match the actual `.gitignore`, which is far more extensive and Next.js/Node-specific.

**`README.md`**
- Project Structure block (lines 56-104) is accurate for `training/`, `testing/`, `inference/`, `server/`, `client/`, `datasets/`, `models/` folder *names*, but the file lists inside each (e.g. `test_api.py` under `testing/` — no such file exists; the real API-smoke-test file is `test_ai.py`) don't match.
- "Testing Your Trained Model" section (line 210) still says `cd scripts/download` and run `auto_download_coco.py` — neither the directory nor the file exists.
- "Camera Calibration" section (line 224) says `cd scripts/download && python angled_camera_calibration.py` — same, does not exist; closest real file is `inference/camera_calibration.py` with a different name and interface.
- "Utilities → Frame Extraction" (line 441) says `cd scripts/extract_frames` — real path is `inference/extract_frames.py` (no subfolder, no `scripts/` prefix).
- Documentation links (lines 453, 456, 530) point at `scripts/training/YOLO_NATIVE_DATASETS.md` and `scripts/training/NIGHT_VISION_DATASETS.md` — the first is actually at `training/YOLO_NATIVE_DATASETS.md` (no `scripts/` prefix) and the second (`NIGHT_VISION_DATASETS.md`) **does not exist anywhere in the repo** under any path.
- "Usage Examples → Training" (lines 390-398) invokes `python scripts/training/train_vehicle_detector.py` — file doesn't exist; real script is `training/train.py` with an entirely different CLI (`--dataset {vehicle,accident,both}` vs. the documented flat `--model/--epochs/--batch`).
- ENVIRONMENT_SETUP.md is referenced (line 285) but **does not exist** anywhere in the repo.

**`START_HERE.md`**
- Every `cd scripts/training` instruction (lines 75, 211, 252, 353) should be `cd training`.
- Step 6/7 training commands invoke `train_vehicle_detector.py` (lines 215, 256) — the real script is `train.py` with a different CLI.
- "Testing Your Models" (lines 356-365) and "Use in Speed Detection" (lines 371-376) reference `auto_download_coco.py`, which doesn't exist.

**`TRAINING_GUIDE.md`**
- "Option 1: Quick Start" (lines 104-117) documents `quick_train.py`, saving to `runs/vehicle_speed/quick_v1/weights/best.pt` — this script does not exist in the repo (nor does the `runs/vehicle_speed/quick_v1` path).
- "Option 2" and most of the rest of the document is built entirely around `train_vehicle_detector.py`, which does not exist; the real trainer (`train.py`) uses a `--dataset {vehicle,accident,both} --model-size {n,s,m,l,x}` interface with YOLO26 base weights, not the documented `--model {n,s,m,l,x} --epochs --batch` interface against YOLOv8.
- "Project Structure" block (lines 229-251) again shows the `scripts/training/`, `scripts/download/`, `scripts/extract_frames/` layout.
- Frames it as a **COCO-dataset, generic vehicle-class training** guide throughout; the repo's actual, functional training path (`training/train.py` + `merge_busay_datasets.py`) is Busay-specific (Roboflow merge → YOLO26), which this doc never describes.

**`CAMERA_TEST_GUIDE.md`**
- States "**Location**: `server/ai-service/test_camera.py`" (line 7) and gives `cd server/ai-service && python test_camera.py` throughout — the real, only copy of `test_camera.py` lives at `testing/test_camera.py`; there is no `test_camera.py` (or any test script) inside `server/ai-service/`.
- "Advanced Testing" (line 322) points at `server/ai-service/test_visual_pro.py` — **this file does not exist anywhere in the repository, tracked or untracked.**
- References `train_yolo26.py --dataset both --model-size s` (line 291) — the real filename is `train.py`, not `train_yolo26.py` (though the CLI shape it describes does match `train.py`'s actual interface, unlike the other docs above).

**`server/ai-service/README.md`** (not in the original doc-drift target list, but discovered mid-audit and material enough to flag)
- "Available Test Scripts" table (lines 228-239) and "Project Structure" block (lines 472-479) list `test_camera.py`, `test_visual.py`, `test_visual_pro.py`, `test_visual_optimal.py`, `test_video.py`, `test_images.py`, `test_ai.py` as living inside `server/ai-service/`. In reality all of the ones that exist (`test_camera.py`, `test_visual.py`, `test_video.py`, `test_images.py`, `test_ai.py`) live in the top-level `testing/` folder, and **`test_visual_pro.py` / `test_visual_optimal.py` do not exist anywhere in the repository at all.**

**`training/README.md`** (in-folder doc, also stale relative to its own folder's real contents)
- Documents `train_vehicle_detector.py` and `quick_train.py` (lines 138, 161-162, 225-268) — the real files in the same directory are `train.py` and `validate.py`.

**`raspi_scripts/README.md`**
- Reasonably accurate for what it covers, but silently omits `pi_agent.py`, `camera/camera_sender.py`, `color_test.py`, `test_display.py`, `setup_pi4.sh`, `setup_pi5.sh`, `hub75_piomatter_notes.md`, and `SETUP_GUIDE.html` — i.e. most of the folder's actual content is undocumented by its own README.
- Presents `lcd/` vs `lcd_pi4/` as *the* two current display implementations (lines 99-113); the actually-current, actively-maintained driver is the unified top-level `display_manager.py`, which auto-selects between the two backends and supersedes both subfolder scripts (confirmed via git history — `lcd/` and `lcd_pi4/`'s `display_manager.py` files have only early, placeholder-message commits like `"123"`, `"789"`).

---

## 16. Open threads

- **LED matrix RP1 timing drift** is the single most actively-worked problem in the repo: of the last ~35 commits on `main`, the large majority are iterative fixes to Pi 5 HUB75 panel corruption/mirroring/flicker (heartbeat restarts, pwm-bits tuning, backend swaps between `LedImageViewerBackend` and `RGBMatrixBackend`, a reverted attempt at `--led-rp1-rio=1`). The `RGBMatrixBackend` Python-bindings path remains explicitly disabled with an open `# TODO: fix pixel mapping before re-enabling` (`raspi_scripts/display_manager.py:636`).
- **Live camera streaming migration in progress**: the four most recent commits (`ea033a9`, `498f005`, `b74fc36`, `ed6c0fd`) show an active migration from MJPEG-only streaming to WebSocket binary frame push, with MJPEG kept only as a fallback — `components/video-feed.tsx` and `routes/cameras.ts`/`server.ts`'s `pi_frame` handling are the freshest code in the client/server split.
- **Crash/incident model training** has evidently never been completed: `training/train.py --dataset accident` is fully implemented and the merged `datasets/processed/busay_accident_detection/` dataset already exists on disk, but no output run folder exists under `models/runs/accident/` — this is a ready-to-run, not-yet-run step, not a missing capability.
- **Camera calibration UI is a dead end**: the client has "Open Calibration Tool" / "View Calibration Guide" buttons (`app/cameras/page.tsx:311-327`) with no handlers, while a real (if disconnected) homography calibration implementation already exists in `inference/camera_calibration.py` — the two were never connected.
- **History/Reports pages are unstarted**: both are pure static-fixture UI shells with matching backend gaps (no `recordings` table, no export logic) — consistent, not accidental, but definitively unimplemented rather than partially wired.
