# Road Sentinel AI Service

Python-based AI microservice for traffic and incident detection using YOLOv8.

## Features

- **Traffic Detection**: Vehicle detection and classification (car, truck, bus, motorcycle, bicycle)
- **Incident Detection**: Crash, speeding, wrong-way, stopped vehicle detection
- **Speed Estimation**: Calculate vehicle speeds based on frame-to-frame movement
- **FastAPI**: RESTful API for inference requests
- **GPU Acceleration**: CUDA support for fast inference

## Tech Stack

- **Framework**: FastAPI
- **AI Model**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Deep Learning**: PyTorch
- **Language**: Python 3.10+

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download or Train Models

Place your YOLOv8 model weights in the `models/` directory:

- `models/traffic.pt` - Traffic detection model
- `models/incident.pt` - Incident detection model

**Option 1: Use Pretrained COCO Model (for testing)**

The service will automatically download YOLOv8n if custom models aren't found.

**Option 2: Train Custom Models**

See the [YOLOv8 documentation](https://docs.ultralytics.com/modes/train/) for training your own models.

### 3. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
TRAFFIC_MODEL_PATH=./models/traffic.pt
INCIDENT_MODEL_PATH=./models/incident.pt
DEVICE=cuda  # or 'cpu' if no GPU available
```

### 4. Create Models Directory

```bash
mkdir -p models
```

## Running the Service

### Development

```bash
python -m app.main
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

### Combined Detection

```bash
POST /api/detect
Content-Type: multipart/form-data

Form Data:
- image: (file) JPEG image
- camera_id: (string) Camera identifier
- confidence_threshold: (float, optional) Override default confidence
```

Response:
```json
{
  "success": true,
  "camera_id": "CAM-A-001",
  "detections": [
    {
      "class": "car",
      "confidence": 0.92,
      "bbox": {
        "x": 100,
        "y": 200,
        "width": 150,
        "height": 100
      }
    }
  ],
  "incidents": [
    {
      "type": "speeding",
      "severity": "high",
      "confidence": 0.85,
      "description": "Speeding violation detected"
    }
  ],
  "processing_time_ms": 45.3,
  "timestamp": 1234567890.123
}
```

### Traffic Detection Only

```bash
POST /api/detect/traffic
```

### Incident Detection Only

```bash
POST /api/detect/incidents
```

### Service Statistics

```bash
GET /api/stats
```

## Model Information

### Traffic Detection

Detects and classifies:
- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles

### Incident Detection

Detects:
- Crashes / Collisions
- Speeding violations
- Wrong-way vehicles
- Stopped vehicles on roadway
- Traffic congestion
- Illegal parking

## GPU Requirements

- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum VRAM**: 4GB
- **CUDA Version**: 11.7+
- **cuDNN**: 8.0+

For CPU inference, set `DEVICE=cpu` in `.env` (slower performance).

## Performance

Typical inference times on NVIDIA RTX 3060:
- Traffic detection: 15-30ms per frame
- Incident detection: 20-40ms per frame
- Combined detection: 35-70ms per frame

## Project Structure

```
app/
├── models/
│   ├── traffic_detector.py   # YOLOv8 traffic detection
│   └── incident_detector.py  # YOLOv8 incident detection
└── main.py                    # FastAPI application

models/                        # Model weights directory
├── traffic.pt                 # Traffic detection weights
└── incident.pt                # Incident detection weights
```

## Training Custom Models

To train your own models:

1. Prepare dataset in YOLO format
2. Use Ultralytics YOLOv8:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    device=0
)
```

3. Export trained weights to `models/` directory

## License

MIT
