# Camera Frame Sender — Pi 4 & Pi 5

Captures RTSP streams from the IP cameras and POSTs JPEG frames to the AI service for YOLO inference. No LED display — cameras only.

Works identically on **Raspberry Pi 4 Model B** and **Raspberry Pi 5**.

---

## How frames travel (fastest path)

```
[IP Camera A]                        [AI Server — RTX 3060 Ti]
rtsp://192.168.8.104                  http://192.168.8.50:8000
       │                                        │
       │  RTSP (H.264)                          │
       ▼                                        │
[Raspberry Pi]                                  │
  OpenCV decode → raw BGR frame                 │
  cv2.imencode JPEG (75%, ~35KB)                │
  aiohttp POST /api/detect ─────────────────────┘
                                                │
                                          YOLO inference
                                          ~30–80ms on GPU
                                                │
                                         JSON detections
                                                │
                                         Node Service :3001
                                                │
                                          Dashboard / DB
```

### Why HTTP POST and not ZeroMQ / raw TCP?

| Option | Latency | Server change needed | Chosen? |
|--------|---------|---------------------|---------|
| **HTTP POST (aiohttp keep-alive)** | ~5ms | None — uses existing `/api/detect` | **Yes** |
| ZeroMQ PUSH/PULL | ~0.3ms | Add ZMQ receiver to AI service | Future option |
| Raw TCP | ~0.5ms | Full custom protocol on server | Overkill |
| Server pulls RTSP direct | 0ms Pi | Server must reach camera IPs | Alternative |

HTTP POST at 30fps = 5ms × 30 = 150ms/s of network overhead per camera.
GPU inference at 30fps = 30–80ms per frame = the real bottleneck.
The HTTP overhead is negligible compared to inference time.

### Bandwidth per camera

```
30 fps × 35 KB/frame = ~1.05 MB/s per camera
Two cameras           = ~2.1 MB/s total
LAN capacity (100Mbps)= 12.5 MB/s
```
LAN is never the bottleneck.

---

## Quick Start

```bash
# On the Raspberry Pi:
cd ~/pauledison/Road_Sentinel/raspi_scripts/camera

bash setup.sh 192.168.8.50
# or with custom RTSP URLs:
bash setup.sh 192.168.8.50 rtsp://192.168.8.104:554/cam/realmonitor rtsp://192.168.8.108:554/cam/realmonitor
```

This installs everything and starts both cameras immediately.

---

## Arguments

```
bash setup.sh [AI_SERVER_IP] [CAM_A_RTSP] [CAM_B_RTSP] [TARGET_FPS]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `AI_SERVER_IP` | `192.168.8.50` | Server running the AI service |
| `CAM_A_RTSP` | `rtsp://192.168.8.104:554/cam/realmonitor` | Camera A RTSP URL |
| `CAM_B_RTSP` | `rtsp://192.168.8.108:554/cam/realmonitor` | Camera B RTSP URL |
| `TARGET_FPS` | `30` | Frames per second to send |

---

## After Setup

```bash
# Check both cameras are running:
~/camera_scripts/status_cameras.sh

# Live logs:
tail -f ~/camera_logs/camera_cam_a.log
tail -f ~/camera_logs/camera_cam_b.log

# Test AI server is reachable:
~/camera_scripts/test_ai_connection.sh

# Stop:
~/camera_scripts/stop_cameras.sh

# Start:
~/camera_scripts/start_cameras.sh
```

---

## Pi 4 vs Pi 5 — what's different

| | Pi 4 | Pi 5 |
|--|------|------|
| RTSP decode (IP cameras) | OpenCV + FFmpeg — identical | OpenCV + FFmpeg — identical |
| libcamera (CSI cameras) | Not needed (IP cameras) | Not needed (IP cameras) |
| CPU performance | ~4 fps headroom after RTSP decode | ~8 fps headroom after RTSP decode |
| Setup script changes | None | None |

Both Pi 4 and Pi 5 run this script identically for RTSP IP cameras.

---

## Manual run (without systemd)

```bash
source ~/venvs/cam_venv/bin/activate

# Camera A:
python3 ~/camera_scripts/camera_sender.py \
    --camera-id cam_a \
    --rtsp rtsp://192.168.8.104:554/cam/realmonitor \
    --ai http://192.168.8.50:8000

# Camera B (in another terminal):
python3 ~/camera_scripts/camera_sender.py \
    --camera-id cam_b \
    --rtsp rtsp://192.168.8.108:554/cam/realmonitor \
    --ai http://192.168.8.50:8000
```

---

## Tuning

| Flag | Default | When to change |
|------|---------|----------------|
| `--fps 15` | 30 | Lower if Pi CPU is maxed out |
| `--quality 60` | 75 | Lower if LAN is congested |
| `--quality 90` | 75 | Higher if detection accuracy drops |

Log output every 10s shows achieved FPS and error count:
```
10:22:01 [cam_a] INFO sent=300 errors=0 dropped=2 fps=29.8
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Cannot open RTSP stream` | Check camera is on and RTSP URL is correct: `ffplay rtsp://...` |
| `POST error: Cannot connect` | Run `~/camera_scripts/test_ai_connection.sh` — AI service may be down |
| `fps=8` in logs | Pi CPU overloaded — lower `--fps` or `--quality` |
| `errors=30+` in logs | AI server not responding — check server logs |
| Service not starting | `sudo journalctl -u roadsentinel-camera-cam_a -n 50` |
