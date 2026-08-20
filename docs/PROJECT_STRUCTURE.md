# Road Sentinel Project Structure

Real, current file structure for the Busay blind curve vehicle detection system.

## 📁 Root Directory Structure

```
RoadSentinel/
├── docs/                         # Ground-truth audit, revamp record, supplementary guides (this file included)
├── training/                     # YOLO26 training pipeline
├── testing/                      # HTTP integration tests against the deployed AI service
├── inference/                    # Standalone offline utilities (not the production path)
├── server/
│   ├── ai-service/               # FastAPI — detection, homography speed, media storage
│   ├── node-service/             # Express + Socket.IO — API, MySQL, auth
│   └── database/                 # mysql_schema.sql (generated; migrate.ts is authoritative)
├── client/web/                   # Next.js dashboard ("Night Watch" design system)
├── raspi_scripts/                # Runs on the Raspberry Pis
├── datasets/                     # downloaded/ (raw) + processed/ (merged, ready to train)
├── models/                       # models/runs/<dataset>/<run>/weights/ — real trained output
├── docker-compose.yml            # Local MySQL + Adminer for development
├── start.bat                     # One-click local dev stack (Windows)
└── README.md                     # Start here
```

---

## 📦 Detailed Structure

### `/docs` — Documentation

```
docs/
├── documentation.md              # Ground-truth codebase audit
├── Summarization.md              # Full phase-by-phase revamp record
├── ROAD_SENTINEL_REVAMP_MASTER.md  # The revamp's own planning doc
├── PROJECT_STRUCTURE.md          # This file
├── START_HERE.md
├── TRAINING_GUIDE.md
└── CAMERA_TEST_GUIDE.md
```

`client/web/DESIGN.md` is the one exception kept next to the code it describes, since it's specifically about that folder's design system.

---

### `/datasets` — Training Data

```
datasets/
├── README.md
├── downloaded/                  # Raw Roboflow export(s) — untracked, gitignored
└── processed/                   # Output of training/merge_busay_datasets.py
    ├── busay_vehicle_detection/{data.yaml,train,valid,test}
    └── busay_accident_detection/{data.yaml,train,valid,test}
```

Both processed datasets already exist on a working checkout; only `busay_vehicle_detection` has been trained on so far (see `/models` below). Merge with `python training/run_merge_busay.py` — there is no `scripts/` folder in this repo.

---

### `/models` — Trained Models

```
models/
├── README.md
└── runs/
    └── vehicle/
        └── vehicle_yolo26n_20260203_032528/
            ├── weights/{best,last,epoch0,epoch10,...}.pt
            ├── args.yaml, results.csv, results.png, confusion_matrix.png, ...
```

No `models/v1/`, `models/v2/`, or `models/production/` symlink layout exists. No `accident` run exists yet — the crash/incident model has never been trained (dataset is ready, GPU job hasn't been run). Output path is set by `training/train.py`'s `RUNS_DIR = PROJECT_ROOT / "models" / "runs"`, project/name derived from `--dataset`/`--model-size`/timestamp — not a manually chosen `--project`/`--name` pair.

---

### `/training` — Model Training

```
training/
├── README.md
├── requirements.txt
├── verify_setup.py                # Checks torch/ultralytics/opencv install + GPU detection
├── train.py                       # Real trainer: --dataset {vehicle,accident,both} --model-size {n,s,m,l,x}
├── validate.py                    # Tests a local .pt file directly (misleadingly named — it's a predict/test script, not model.val())
├── merge_busay_datasets.py        # Merge logic
├── run_merge_busay.py             # Interactive merge wrapper — run this first
├── analyze_datasets.py            # Dataset comparison tool
└── convert_aicity_track1_to_yolo.py / convert_aicity_track4_to_yolo.py / download_roboflow_datasets.py / download_test_video.py
```

There is no `train_vehicle_detector.py` or `quick_train.py` — the real, only trainer is `train.py`. (An earlier version of this repo had four additional strategy/guide documents built entirely around that nonexistent script; they were removed rather than fixed, since they described a training path that never existed here — see `documentation.md §15`.)

---

### `/inference` — Standalone Offline Utilities

```
inference/
├── README.md
├── speed_detection.py    # Standalone speed-tracking prototype (flat pixels-per-meter, no server)
└── extract_frames.py     # Batch frame extractor for dataset creation
```

`camera_calibration.py` (the homography prototype) was removed — its algorithm is now implemented directly in production (`server/ai-service/app/models/traffic_detector.py`), wired to the Cameras page's Calibration Tool. This folder is **not** the production inference path; the AI service (`server/ai-service`) is.

---

### `/testing` — Integration Tests

```
testing/
├── README.md
├── test_ai.py             # Quick smoke test: health, stats, /api/detect*
├── test_camera.py         # Live camera + optional AI overlay
├── test_camera_native.py  # Minimal native-resolution camera open (Windows diagnostic)
├── test_images.py         # POST images to the AI service
├── test_video.py          # POST video frames to the AI service
└── test_visual.py         # Real-time GUI with bounding-box overlay
```

All of these exercise the **deployed AI service over HTTP** (`http://localhost:8000` by default) — none of them import a model directly. For testing a specific trained `.pt` file with no server running, use `training/validate.py` instead.

---

### `/server` — Backend Services

```
server/
├── README.md
├── ai-service/            # FastAPI — see server/ai-service/README.md
├── node-service/          # Express + Socket.IO — see server/node-service/README.md
└── database/
    └── mysql_schema.sql   # Generated reference, kept in sync with migrate.ts by hand — migrate.ts is authoritative
```

MySQL is self-hosted and local-only (`docker-compose.yml` at the repo root for development; production runs the same way on `irm-pc`). Aiven was dropped entirely in Phase 0.5 after its hostname went NXDOMAIN.

### `/raspi_scripts` — Raspberry Pi Side

```
raspi_scripts/
├── README.md
├── camera/camera_sender.py   # RTSP → AI service → Node (production path, both Pis)
├── camera/README.md
├── display_manager.py        # Unified Pi 4/Pi 5 LED matrix driver
├── pi_agent.py                # Authenticated admin-terminal relay
├── setup_pi4.sh / setup_pi5.sh  # systemd install scripts (Camera A + LED / Camera B + LED)
├── fix_gpio_timing.sh          # Pi 4 GPIO-timing diagnostic/fixer
└── color_test.py, test_display.py, hub75_piomatter_notes.md  # Hardware bring-up/diagnostic scripts + notes
```

There used to also be `lcd/` and `lcd_pi4/` subfolders containing earlier, per-model LED driver implementations that predated the unified `display_manager.py` above — removed as dead code (their own git history was placeholder-message-only commits).

`camera_reboot_autostart_setup.sh` (repo root) is a **separate, legacy** ffplay-based desktop-autostart path, independent of the systemd services above — see `raspi_scripts/README.md`.

---

## 🔍 Quick Reference

| Task | Command | Output |
|------|---------|--------|
| Run the whole local dev stack | `start.bat` (Windows) | Dashboard, Node API, AI service, MySQL + Adminer all up |
| Merge datasets | `python training/run_merge_busay.py` | `datasets/processed/` |
| Train vehicle model | `python training/train.py --dataset vehicle --model-size n --epochs 100` | `models/runs/vehicle/.../weights/best.pt` |
| Train accident model | `python training/train.py --dataset accident --model-size n --epochs 100` | `models/runs/accident/.../weights/best.pt` (not yet run) |
| Test a local weight file | `python training/validate.py --model <path> --source <video/image>` | Annotated output |
| Test the running AI service | `python testing/test_ai.py` | Console output |
| Calibrate a camera | Web client → Cameras → Calibration Tool | Stored on `cameras.homography_points` |

## 📞 Help

- **Ground truth:** `docs/documentation.md`
- **Live revamp status:** `docs/Summarization.md`
- **Design system:** `client/web/DESIGN.md`
- **Training:** `training/README.md`
- **Raspberry Pi:** `raspi_scripts/README.md`
- **AI service:** `server/ai-service/README.md`
