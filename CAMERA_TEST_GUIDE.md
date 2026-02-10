# Camera Test Guide

## Quick Start

This guide explains how to use the `test_camera.py` script to test camera connections with the Road Sentinel system.

**Location**: `server/ai-service/test_camera.py`

## Features

- **Auto-Detection**: Automatically scans for available cameras (indices 0-4)
- **Live Preview**: Real-time camera feed with FPS monitoring
- **AI Detection**: Optional YOLO26 vehicle and incident detection
- **Interactive Controls**: Toggle AI, switch cameras, and quit on the fly
- **Visual Feedback**: Bounding boxes, labels, and detection statistics

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install opencv-python numpy requests
```

## Usage

### 1. List Available Cameras

Find all connected cameras:

```bash
cd server/ai-service
python test_camera.py --list
```

Output:
```
Scanning for available cameras (checking indices 0-4)...
✓ Camera 0 found: 1280x720 @ 30fps
✓ Camera 1 found: 640x480 @ 30fps

Found 2 camera(s): [0, 1]
```

### 2. Test Camera (Auto-Detect)

Run with the first available camera:

```bash
cd server/ai-service
python test_camera.py
```

This will:
- Auto-detect the first available camera
- Check if AI service is running
- Enable AI detection if service is available
- Show live camera feed with detections

### 3. Test Specific Camera

Test a specific camera by ID:

```bash
cd server/ai-service
python test_camera.py --camera 0
```

### 4. Camera Only Mode (No AI)

Test camera feed without AI detection:

```bash
cd server/ai-service
python test_camera.py --no-ai
```

This is useful for:
- Testing camera hardware
- Checking camera quality and FPS
- Running without the AI service

### 5. Custom Confidence Threshold

Adjust detection sensitivity (0.0 to 1.0):

```bash
cd server/ai-service
python test_camera.py --confidence 0.7
```

- Lower values (0.3-0.5): More detections, more false positives
- Higher values (0.7-0.9): Fewer detections, higher accuracy

## Interactive Controls

While the camera test is running, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| **Q** | Quit the application |
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

- **Green**: Cars
- **Orange**: Trucks
- **Yellow**: Buses
- **Magenta**: Motorcycles
- **Cyan**: Bicycles
- **Red**: Incidents (accidents, stopped vehicles)

Each box shows:
```
vehicle_type: 0.87
```

## AI Service Setup

### Starting the AI Service

For AI detection to work, you need to start the AI service first:

```bash
cd server/ai-service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Checking AI Service Status

The script automatically checks if the AI service is running at `http://localhost:8000`.

If the service is not available, you'll see:
```
✗ AI Service not available at http://localhost:8000
  Running in camera-only mode (no AI detection)
```

## Use Cases

### 1. Quick Camera Check

```bash
cd server/ai-service
python test_camera.py --no-ai
```

Verify camera is working without starting the AI service.

### 2. Full Detection Test

```bash
# Terminal 1: Start AI service
cd server/ai-service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Run camera test (same directory)
cd server/ai-service
python test_camera.py
```

Test complete pipeline with vehicle/incident detection.

### 3. Test Multiple Cameras

```bash
cd server/ai-service
python test_camera.py
# Press 'C' to cycle through cameras
```

Compare quality and positioning of different cameras.

### 4. Optimize Detection Settings

```bash
cd server/ai-service
python test_camera.py --confidence 0.6
# Adjust confidence until you get optimal results
```

Find the sweet spot between sensitivity and accuracy.

## Troubleshooting

### No Cameras Found

**Problem**: `✗ No cameras found!`

**Solutions**:
1. Check camera is connected (USB, built-in, network camera)
2. Try unplugging and reconnecting USB cameras
3. Check camera permissions:
   ```bash
   # Linux: Add user to video group
   sudo usermod -a -G video $USER
   ```
4. Close other applications using the camera (Zoom, Skype, etc.)
5. Test camera with system tools:
   ```bash
   # Linux
   ls /dev/video*
   v4l2-ctl --list-devices
   ```

### AI Service Not Connecting

**Problem**: `✗ AI Service not available`

**Solutions**:
1. Verify AI service is running:
   ```bash
   curl http://localhost:8000/health
   ```
2. Check if port 8000 is in use:
   ```bash
   lsof -i :8000
   ```
3. Review AI service logs for errors
4. Ensure models are downloaded and paths are correct in `.env`

### Low FPS

**Problem**: Frame rate is too low (< 10 FPS)

**Solutions**:
1. Reduce camera resolution:
   - Edit `test_camera.py` line 253-254
   - Change from 1280x720 to 640x480
2. Increase AI processing interval:
   - Edit line 276: `process_every_n_frames = 5`  # Process less frequently
3. Use faster model (nano instead of large)
4. Disable AI detection temporarily (press SPACE)

### Camera Opens But Shows Black Screen

**Problem**: Camera opens but no image

**Solutions**:
1. Wait a few seconds (some cameras need warm-up time)
2. Check camera lens cover is removed
3. Test camera exposure settings
4. Try a different camera index

## Advanced Configuration

### Custom AI Service URL

Edit `test_camera.py` line 13:

```python
AI_SERVICE_URL = "http://192.168.1.100:8000"  # Remote server
```

### Custom Camera Resolution

Edit lines 253-254 in `test_camera.py`:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Full HD
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

### Custom Processing Interval

Edit line 276:

```python
process_every_n_frames = 5  # Process every 5th frame (faster)
```

## Integration with Training

After training your YOLO26 models, use this script to test them:

1. Train your models:
   ```bash
   cd scripts/training
   python train_yolo26.py --dataset both --model-size s
   ```

2. Update AI service model paths in `server/ai-service/.env`:
   ```env
   TRAFFIC_MODEL_PATH=/path/to/best.pt
   INCIDENT_MODEL_PATH=/path/to/best.pt
   ```

3. Restart AI service and test:
   ```bash
   cd server/ai-service
   python -m uvicorn app.main:app --reload

   # In another terminal
   cd server/ai-service
   python test_camera.py --confidence 0.6
   ```

## Performance Tips

1. **For Smooth Playback**: Use `process_every_n_frames = 3-5`
2. **For Maximum Accuracy**: Use `process_every_n_frames = 1` (slower)
3. **For Testing Hardware**: Use `--no-ai` flag
4. **For Remote Cameras**: Consider network latency when setting timeouts

## Next Steps

Once camera testing is successful:

1. **Calibrate for Speed Detection**: Use `scripts/download/angled_camera_calibration.py`
2. **Advanced Testing**: Try `server/ai-service/test_visual_pro.py` for vehicle tracking
3. **Production Deployment**: Integrate with Node.js service for database logging
4. **Multi-Camera Setup**: Deploy multiple camera instances with unique IDs

---

**Need Help?** Check the main [README.md](README.md) or [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
