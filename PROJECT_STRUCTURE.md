# Road Sentinel Project Structure

Real, current file structure for the Busay blind curve vehicle detection system. (This replaces an earlier version of this file that described a `scripts/`-based layout that no longer matches the tracked repo — see `documentation.md §15` for the full history.)

## 📁 Root Directory Structure

```
RoadSentinel/
├── training/                    # YOLO26 training pipeline
├── testing/                     # HTTP integration tests against the deployed AI service
├── inference/                   # Standalone offline utilities (not the production path)
├── server/
│   ├── ai-service/              # FastAPI — detection, homography speed, media storage
│   ├── node-service/            # Express + Socket.IO — API, MySQL, auth
│   └── database/                # mysql_schema.sql (generated; migrate.ts is authoritative)
├── client/web/                  # Next.js dashboard
├── raspi_scripts/               # Runs on the Raspberry Pis
├── datasets/                    # downloaded/ (raw) + processed/ (merged, ready to train)
├── models/                      # models/runs/<dataset>/<run>/weights/ — real trained output
├── documentation.md             # Ground-truth codebase audit
└── Summarization.md             # Live revamp status
```

---

## 📦 Detailed Structure

### `/datasets` — Training Data

```
datasets/
├── README.md
├── downloaded/                  # Raw Roboflow export(s) — untracked, gitignored
└── processed/                   # Output of training/merge_busay_datasets.py
    ├── busay_vehicle_detection/{data.yaml,train,valid,test}
    └── busay_accident_detection/{data.yaml,train,valid,test}
```

Both processed datasets already exist on a working checkout; only `busay_vehicle_detection` has been trained on so far (see `/models` below). Merge with `python training/run_merge_busay.py`, not `python scripts/training/run_merge_busay.py` — there is no `scripts/` folder in this repo.

---

### `/models` — Trained Models

```
models/
├── README.md                     # Describes an aspirational v1/v2/production layout — not what's here, ignore its paths
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
├── train.py                       # Real trainer: --dataset {vehicle,accident,both} --model-size {n,s,m,l,x}
├── validate.py                    # Tests a local .pt file directly (misleadingly named — it's a predict/test script, not model.val())
├── merge_busay_datasets.py        # Merge logic
├── run_merge_busay.py             # Interactive merge wrapper — run this first
├── analyze_datasets.py            # Dataset comparison tool
├── convert_aicity_track1_to_yolo.py / convert_aicity_track4_to_yolo.py
├── download_roboflow_datasets.py / download_test_video.py
└── DATASET_STRATEGY_GUIDE.md, DUAL_MODEL_TRAINING_GUIDE.md, OVERHEAD_CAMERA_GUIDE.md, YOLO_NATIVE_DATASETS.md
```

There is no `train_vehicle_detector.py` or `quick_train.py` — those were removed/renamed to `train.py` at some point before this repo's current history and the docs were never updated.

---

### `/inference` — Standalone Offline Utilities

```
inference/
├── speed_detection.py    # Standalone speed-tracking prototype (flat pixels-per-meter, no server)
└── extract_frames.py     # Batch frame extractor for dataset creation
```

`camera_calibration.py` (the homography prototype) was removed — its algorithm is now implemented directly in production (`server/ai-service/app/models/traffic_detector.py`), wired to the Cameras page's Calibration Tool. This folder is **not** the production inference path; the AI service (`server/ai-service`) is.

---

### `/testing` — Integration Tests

```
testing/
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
├── ai-service/            # FastAPI — see server/ai-service/README.md
├── node-service/           # Express + Socket.IO — see server/node-service/README.md
└── database/
    └── mysql_schema.sql    # Generated reference, kept in sync with migrate.ts by hand — migrate.ts is authoritative
```

### `/raspi_scripts` — Raspberry Pi Side

```
raspi_scripts/
├── camera/camera_sender.py   # RTSP → AI service → Node (production path, both Pis)
├── display_manager.py        # Unified Pi 4/Pi 5 LED matrix driver
├── pi_agent.py                # Authenticated admin-terminal relay
├── setup_pi4.sh / setup_pi5.sh  # systemd install scripts (Camera A only / Camera B + LED)
├── lcd_pi4/, lcd/             # Earlier per-model LED drivers — superseded by display_manager.py, kept for reference
└── color_test.py, test_display.py, fix_gpio_timing.sh  # Hardware bring-up/diagnostic scripts
```

`camera_reboot_autostart_setup.sh` (repo root) is a **separate, legacy** ffplay-based desktop-autostart path, independent of the systemd services above — see `raspi_scripts/README.md`.

---

## 🔍 Quick Reference

| Task | Command | Output |
|------|---------|--------|
| Merge datasets | `python training/run_merge_busay.py` | `datasets/processed/` |
| Train vehicle model | `python training/train.py --dataset vehicle --model-size n --epochs 100` | `models/runs/vehicle/.../weights/best.pt` |
| Train accident model | `python training/train.py --dataset accident --model-size n --epochs 100` | `models/runs/accident/.../weights/best.pt` (not yet run) |
| Test a local weight file | `python training/validate.py --model <path> --source <video/image>` | Annotated output |
| Test the running AI service | `python testing/test_ai.py` | Console output |
| Calibrate a camera | Web client → Cameras → Calibration Tool | Stored on `cameras.homography_points` |

## 📞 Help

- **Ground truth:** `documentation.md`
- **Live revamp status:** `Summarization.md`
- **Training:** `training/README.md`
- **Raspberry Pi:** `raspi_scripts/README.md`
- **AI service:** `server/ai-service/README.md`
