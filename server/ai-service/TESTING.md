# Testing Road Sentinel AI Service

Complete guide for testing the AI service with your own traffic videos, cameras, and images.

## Prerequisites

1. **AI Service Running**
   ```bash
   cd /home/user/Road_Sentinel/server/ai-service
   python -m app.main
   ```

   You should see:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

2. **Dependencies Installed**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎥 **VISUAL TESTING (RECOMMENDED)** - See AI in Action!

**Best for:** Real-time visual feedback with bounding boxes and live detection

### Test with Your Video File

```powershell
# Windows
python test_visual.py "C:\path\to\your\video.mp4"
```

```bash
# Linux/macOS
python test_visual.py /path/to/your/video.mp4
```

### Use Your Webcam/Laptop Camera

```powershell
# Use default webcam (camera 0)
python test_visual.py --camera

# Use specific camera (if you have multiple)
python test_visual.py --camera 1
```

### What You'll See

A window will open showing:
- ✅ **Live video playback** with AI detection
- ✅ **Bounding boxes** around detected vehicles (color-coded by type)
- ✅ **Vehicle labels** with confidence scores
- ✅ **Incident alerts** displayed at top
- ✅ **Real-time statistics** (FPS, detections, vehicle counts)
- ✅ **Live info panel** with session stats

**Vehicle Colors:**
- 🟢 **Green** - Cars
- 🟠 **Orange** - Trucks
- 🟡 **Yellow** - Buses
- 🟣 **Magenta** - Motorcycles
- 🔵 **Cyan** - Bicycles
- 🔴 **Red** - Incidents/Alerts

### Interactive Controls

While the window is open:
- **SPACE** - Pause/Resume playback
- **Q or ESC** - Quit
- **+ or =** - Increase confidence threshold
- **- or _** - Decrease confidence threshold

### Examples

```powershell
# Test with traffic video
python test_visual.py "C:\Users\Shaloh\Videos\traffic.mp4"

# Use webcam with lower confidence (detect more)
python test_visual.py --camera --confidence 0.3

# Test with external USB camera
python test_visual.py --camera 1
```

---

## 🎬 Testing with Your Own Videos

### Basic Video Test

```bash
python test_video.py /path/to/your/traffic_video.mp4
```

This will:
- Extract frames from your video
- Send each frame to the AI service
- Detect vehicles and incidents
- Show statistics and results

### Options

**Process faster (every 10 frames instead of 5):**
```bash
python test_video.py video.mp4 --frame-rate 10
```

**Lower confidence for more detections:**
```bash
python test_video.py video.mp4 --confidence 0.3
```

**Save annotated frames with bounding boxes:**
```bash
python test_video.py video.mp4 --save
```
Frames are saved to `output/video_name/`

**Show real-time detection (requires display):**
```bash
python test_video.py video.mp4 --show
```
Press `Q` to quit

**Combine options:**
```bash
python test_video.py video.mp4 --frame-rate 5 --confidence 0.5 --save --show
```

### Example Output

```
🎬 Testing Video: traffic_footage.mp4
================================================

📹 Video Info:
   Total frames: 3000
   FPS: 30.00
   Duration: 100.00 seconds
   Processing every 5 frames (~6.0 FPS)

✅ AI service is running

🔄 Processing video...

Frame     0: 5 vehicles, 0 incidents (45.3ms)
Frame     5: 6 vehicles, 0 incidents (42.1ms)
Frame    10: 4 vehicles, 1 incidents (48.7ms)
...

📊 DETECTION SUMMARY
================================================
Total frames processed: 600
Total vehicles detected: 2,847
Total incidents detected: 12
Average processing time: 44.23ms per frame

🚗 Vehicle Types:
   car: 2,145
   truck: 487
   motorcycle: 215

⚠️  Incident Types:
   speeding: 8
   congestion: 4
```

---

## 🖼️ Testing with Images

### Single Image Test

```bash
python test_images.py /path/to/traffic_image.jpg
```

**Save annotated image:**
```bash
python test_images.py image.jpg --save
```
Saves to `output/annotated_image.jpg`

### Test Folder of Images

```bash
python test_images.py /path/to/images/ --folder
```

This will test all images (jpg, png, bmp) in the folder.

### Example Output

```
🖼️  Testing image: highway_traffic.jpg

✅ Detection successful!
   Processing time: 42.15ms

🚗 Vehicles detected: 8
   1. car (confidence: 0.94)
      bbox: x=150, y=200, w=180, h=120
   2. truck (confidence: 0.87)
      bbox: x=400, y=180, w=220, h=160
   3. car (confidence: 0.82)
      bbox: x=600, y=210, w=170, h=110
   ...

⚠️  Incidents detected: 1
   1. speeding - high severity
      Speeding violation detected (confidence: 85%)
```

---

## 🧪 Quick Test (Sample Image)

If you don't have videos yet, test with a sample:

```bash
python test_ai.py
```

This downloads a sample traffic image and tests all endpoints.

---

## 📁 Where to Put Your Videos

You can put your videos anywhere, but for organization:

```bash
# Create a test videos folder
mkdir -p ~/test_videos

# Copy your videos there
cp /path/to/your/videos/*.mp4 ~/test_videos/

# Test them
python test_video.py ~/test_videos/video1.mp4
```

---

## 🎨 Viewing Results

### Saved Annotated Frames

When using `--save`, frames are saved to:
```
output/
├── video_name/
│   ├── frame_00000.jpg  (with bounding boxes)
│   ├── frame_00005.jpg
│   └── ...
```

You can view them with any image viewer or create a video:

```bash
# Create video from frames using ffmpeg
ffmpeg -framerate 6 -i output/video_name/frame_%05d.jpg -c:v libx264 output_annotated.mp4
```

---

## 🔧 Troubleshooting

### "AI service not running"

Start the service:
```bash
cd /home/user/Road_Sentinel/server/ai-service
python -m app.main
```

### "Could not open video file"

Make sure the video file path is correct and the format is supported (mp4, avi, mov, mkv).

### "Processing is slow"

- Increase `--frame-rate` to process fewer frames
- Use CPU-optimized requirements: `pip install -r requirements-cpu.txt`
- Consider using a smaller YOLOv8 model (yolov8n instead of yolov8s)

### "No detections found"

- Lower the confidence threshold: `--confidence 0.3`
- Check if your video has clear views of vehicles
- Make sure the AI service loaded the models correctly

### Video codec issues

If OpenCV can't read your video:
```bash
# Convert video to compatible format
ffmpeg -i input.mov -c:v libx264 -preset fast output.mp4
```

---

## 📊 Understanding Results

### Vehicle Types
- **car**: Passenger vehicles
- **truck**: Large vehicles, semi-trucks
- **bus**: Buses
- **motorcycle**: Motorcycles, scooters
- **bicycle**: Bicycles

### Confidence Scores
- **0.8 - 1.0**: Very confident detection
- **0.6 - 0.8**: Good detection
- **0.4 - 0.6**: Moderate detection (may have false positives)
- **< 0.4**: Low confidence (likely false positives)

### Incident Types
- **crash**: Vehicle collision detected
- **speeding**: Vehicle exceeding speed limit
- **stopped_vehicle**: Vehicle stopped on roadway
- **congestion**: Traffic congestion
- **wrong_way**: Wrong-way vehicle

---

## 💡 Tips

1. **Best Frame Rates**:
   - `--frame-rate 1`: Process every frame (slowest, most accurate)
   - `--frame-rate 5`: Good balance (~6 FPS detection)
   - `--frame-rate 10`: Faster processing (~3 FPS detection)

2. **Optimal Confidence**:
   - Start with `0.5` (default)
   - Lower to `0.3` if missing detections
   - Raise to `0.7` if too many false positives

3. **Save Storage**:
   - Don't use `--save` for long videos (generates many images)
   - Use `--save` only for short clips or debugging

4. **Real-time Display**:
   - Use `--show` only for short videos
   - Not recommended for videos > 5 minutes

---

## 🚀 Next Steps

After testing, you can:

1. **Integrate with Node.js Service**: The Node service will handle video streams automatically
2. **Setup Raspberry Pi**: Stream from Raspberry Pi to your server
3. **Connect to Database**: Store detection results in MySQL
4. **Build Dashboard**: View live detections in your Next.js frontend

---

## Examples

```bash
# Quick test with sample
python test_ai.py

# Test your traffic video
python test_video.py ~/Videos/traffic_cam_1.mp4

# Fast processing, save results
python test_video.py video.mp4 --frame-rate 10 --save

# Sensitive detection (lower confidence)
python test_video.py video.mp4 --confidence 0.3

# Real-time viewing
python test_video.py video.mp4 --show

# Test multiple images
python test_images.py ~/Pictures/traffic_images/ --folder

# Test single image and save
python test_images.py traffic.jpg --save
```
