# Camera Test Guide

## Quick Start

This guide explains how to use `test_camera.py` to test camera connections with the Road Sentinel system.

**Location**: `testing/test_camera.py` (not `server/ai-service/` — an earlier version of this doc had the wrong path; see `documentation.md §15`)

## Features

- **Auto-Detection**: Scans for available cameras (indices 0-4)
- **Live Preview**: Real-time camera feed with FPS monitoring
- **AI Detection**: Optional detection via the AI service
- **Interactive Controls**: Toggle AI, switch cameras, quit on the fly
- **Visual Feedback**: Bounding boxes, labels, detection statistics

## Requirements

```bash
pip install opencv-python numpy requests
```

## Usage

```bash
cd testing

# List all available cameras
python test_camera.py --list

# Auto-detect and use first camera (with AI, if the AI service is running)
python test_camera.py

# Test a specific camera
python test_camera.py --camera 0

# Camera only, no AI
python test_camera.py --no-ai

# Custom confidence threshold
python test_camera.py --confidence 0.7
```

## Interactive Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **SPACE** | Toggle AI detection on/off |
| **C** | Switch to next available camera |

## Understanding the Display

### Info Overlay (Top Left)
```
Camera ID: 0           # Which camera is active
FPS: 28.5              # Current frames per second
AI Detection: ON       # AI status (ON/OFF)
Detections: 3          # Number of objects detected
```

### Bounding Boxes
- **Green**: Cars · **Orange**: Trucks · **Yellow**: Buses · **Magenta**: Motorcycles · **Cyan**: Bicycles · **Red**: Incidents

## AI Service Setup

```bash
cd server/ai-service
python -m app.main   # http://localhost:8000
```

`test_camera.py` auto-checks `http://localhost:8000/health` and runs camera-only if the service isn't reachable.

## Use Cases

**Quick camera check (no AI service needed):**
```bash
cd testing && python test_camera.py --no-ai
```

**Full detection test:**
```bash
# Terminal 1
cd server/ai-service && python -m app.main
# Terminal 2
cd testing && python test_camera.py
```

## Troubleshooting

### No Cameras Found
1. Check the camera is connected
2. Close other apps using the camera (Zoom, Skype, etc.)
3. Linux: `ls /dev/video*`, `v4l2-ctl --list-devices`; add user to `video` group

### AI Service Not Connecting
```bash
curl http://localhost:8000/health
```
Check `TRAFFIC_MODEL_PATH`/`INCIDENT_MODEL_PATH` in `server/ai-service/.env` resolve to real files.

### Low FPS
- Reduce camera resolution (edit `cap.set(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT, ...)` near the top of `test_camera.py`)
- Increase the AI processing interval, or press SPACE to disable AI temporarily

## Next Steps

1. **Calibrate for speed detection**: web client → Cameras → Calibration Tool (not a script — see `README.md`)
2. **More testing options**: `testing/test_visual.py` (real-time GUI with stats), `testing/test_video.py` / `test_images.py` (file-based), `testing/test_ai.py` (quick API smoke test)
3. **Production deployment**: `raspi_scripts/` — see `raspi_scripts/README.md`

---

**Need Help?** Check `README.md` or `TRAINING_GUIDE.md`
