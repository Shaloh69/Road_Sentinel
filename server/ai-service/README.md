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

---

## 🚀 Complete Installation Guide

### Prerequisites

- **Python 3.10, 3.11, or 3.12** ([Download](https://www.python.org/downloads/))
- **NVIDIA GPU** (optional, but recommended for 10-20x faster inference)
- **CUDA Toolkit** (if using GPU) - usually comes with GPU drivers

---

## Windows Installation

### Step 1: Navigate to AI Service Directory

```powershell
cd C:\Projects\Thesis\2026\RoadSentinel\server\ai-service
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

#### Option A: With NVIDIA GPU (Recommended - 10-20x Faster)

```powershell
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio

# Install other dependencies
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings ultralytics opencv-python-headless numpy pillow python-dotenv
```

#### Option B: CPU Only (No GPU)

```powershell
# Install all dependencies (PyTorch will auto-select CPU version)
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings ultralytics opencv-python-headless numpy pillow python-dotenv
```

### Step 5: Verify Installation

```powershell
# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check if GPU is available (skip if CPU-only)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

You should see:
- **With GPU**: `CUDA available: True` + your GPU name
- **CPU only**: `CUDA available: False` + `CPU only`

### Step 6: Create Environment File

```powershell
# Copy example env file
copy .env.example .env
```

Edit `.env` file:
- **With GPU**: Set `DEVICE=cuda`
- **CPU only**: Set `DEVICE=cpu`

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model Configuration
TRAFFIC_MODEL_PATH=yolov8n.pt
INCIDENT_MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Device Configuration
DEVICE=cuda  # or 'cpu' if no GPU
```

### Step 7: Create Models Directory

```powershell
mkdir models
```

### Step 8: Start the Service

```powershell
python -m app.main
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**✅ Service is now running!**

---

## Linux/macOS Installation

### Step 1: Navigate to AI Service Directory

```bash
cd /home/user/Road_Sentinel/server/ai-service
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

#### With NVIDIA GPU

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio

# Install other dependencies
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings ultralytics opencv-python-headless numpy pillow python-dotenv
```

#### CPU Only

```bash
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings ultralytics opencv-python-headless numpy pillow python-dotenv
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 6: Configure Environment

```bash
cp .env.example .env
# Edit .env and set DEVICE=cuda or DEVICE=cpu
```

### Step 7: Create Models Directory

```bash
mkdir -p models
```

### Step 8: Start the Service

```bash
python -m app.main
```

---

## 🧪 Testing the AI Service

### Quick Test (Sample Image)

Open a **new terminal** (keep the service running), activate venv, and run:

```powershell
# Windows
cd C:\Projects\Thesis\2026\RoadSentinel\server\ai-service
.\venv\Scripts\Activate.ps1
python test_ai.py
```

```bash
# Linux/macOS
cd /home/user/Road_Sentinel/server/ai-service
source venv/bin/activate
python test_ai.py
```

### Test with Your Own Video

```powershell
# Windows
python test_video.py C:\path\to\your\traffic_video.mp4
```

```bash
# Linux/macOS
python test_video.py /path/to/your/traffic_video.mp4
```

**Options:**
- `--save` - Save annotated frames with bounding boxes
- `--show` - Display real-time detection window
- `--frame-rate 10` - Process every 10th frame (faster)
- `--confidence 0.3` - Lower confidence threshold (more detections)

**Example:**
```powershell
python test_video.py video.mp4 --save --frame-rate 5 --confidence 0.5
```

### Test with Images

```powershell
# Single image
python test_images.py image.jpg --save

# All images in folder
python test_images.py C:\path\to\images\ --folder
```

See [TESTING.md](./TESTING.md) for complete testing guide.

---

## API Endpoints

### Health Check

```bash
GET http://localhost:8000/health
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
POST http://localhost:8000/api/detect
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

### Other Endpoints

- `POST /api/detect/traffic` - Traffic detection only
- `POST /api/detect/incidents` - Incident detection only
- `GET /api/stats` - Service statistics

---

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

---

## GPU Requirements

- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum VRAM**: 4GB
- **CUDA Version**: 11.7+
- **cuDNN**: 8.0+

For CPU inference, set `DEVICE=cpu` in `.env` (10-20x slower).

---

## Performance

### With NVIDIA GPU (CUDA)
- Traffic detection: **15-30ms** per frame
- Incident detection: **20-40ms** per frame
- Combined detection: **35-70ms** per frame

### CPU Only
- Traffic detection: **200-500ms** per frame
- Incident detection: **300-600ms** per frame
- Combined detection: **500-1000ms** per frame

---

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

test_ai.py                     # Quick test script
test_video.py                  # Video testing script
test_images.py                 # Image testing script
TESTING.md                     # Complete testing guide
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'fastapi'"

You forgot to install dependencies or didn't activate the virtual environment.

**Solution:**
```powershell
# Windows
.\venv\Scripts\Activate.ps1
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings ultralytics opencv-python-headless numpy pillow python-dotenv
```

### "CUDA not available" (but you have NVIDIA GPU)

**Solution:**
```powershell
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Virtual environment activation fails (Windows)

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port 8000 already in use

**Solution:**
```powershell
# Change port in .env or start with different port
uvicorn app.main:app --port 8001
```

---

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
    device=0  # GPU 0, or 'cpu'
)
```

3. Export trained weights to `models/` directory

---

## Deactivating Virtual Environment

When you're done:

```powershell
# Windows
deactivate
```

```bash
# Linux/macOS
deactivate
```

---

## License

MIT
