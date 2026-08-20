# Inference — Standalone Offline Utilities

Scripts you run directly against a video file or camera, with no server involved. **Not the production path** — the deployed AI service (`server/ai-service`) is what actually runs in the live system; these are quick offline prototypes and utilities.

```bash
cd inference
pip install -r requirements.txt
```

## `speed_detection.py`

Standalone offline speed-tracking prototype. Loads the stock `yolov8n.pt` COCO model directly (no server, no trained Road Sentinel weight) and estimates speed from a flat pixels-per-meter value — no perspective correction. Useful for a quick local sanity check against a video file without the AI service running.

The production equivalent — homography-corrected speed, the trained vehicle model, calibration via the Cameras page's Calibration Tool — lives in `server/ai-service/app/models/traffic_detector.py`, not here.

## `extract_frames.py`

Batch video-to-frames extractor for building training datasets: point it at a folder of videos, get back extracted frames at a target FPS.

```bash
python extract_frames.py --input /path/to/videos --output /path/to/frames --fps 2
```

Run `python extract_frames.py --help` for the full option list.

## What used to be here

`camera_calibration.py` (a homography-matrix prototype) was removed — its algorithm is now the real, production implementation in `traffic_detector.py`, so the standalone prototype was redundant rather than a separate useful tool.
