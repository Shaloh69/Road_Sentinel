# Professional Visual Testing - 30 FPS with Unique Vehicle Tracking

This professional version includes:
- ✅ **Multi-threaded AI processing** (3-5 workers for 30+ FPS)
- ✅ **DeepSORT tracking** for unique vehicle identification
- ✅ **Bidirectional counting** (IN/OUT with entry/exit lines)
- ✅ **Vehicle state management** (handles vehicles that don't complete journey)
- ✅ **GPU optimizations** (FP16, optimized encoding)

## Installation

### 1. Install Additional Dependencies

```powershell
# Windows (in venv)
cd C:\Projects\Thesis\2026\RoadSentinel\server\ai-service
.\venv\Scripts\Activate.ps1

# Install tracking libraries
pip install --timeout 1000 --retries 10 deep-sort-realtime filterpy scipy
```

```bash
# Linux/Mac (in venv)
cd /path/to/RoadSentinel/server/ai-service
source venv/bin/activate

# Install tracking libraries
pip install deep-sort-realtime filterpy scipy
```

### 2. Verify Installation

```powershell
python -c "from deep_sort_realtime.deepsort_tracker import DeepSort; print('DeepSORT installed successfully!')"
```

## Usage

### Basic Usage (3 Workers)

```powershell
# Start AI service first (Terminal 1)
python -m app.main

# Run professional tester (Terminal 2)
python test_visual_pro.py "C:\path\to\your\video.mp4"
```

### Advanced Options

```powershell
# Use 5 workers for maximum performance (30+ FPS)
python test_visual_pro.py video.mp4 --workers 5

# Adjust confidence threshold
python test_visual_pro.py video.mp4 --confidence 0.3

# Custom counting lines (entry at 20%, exit at 80%)
python test_visual_pro.py video.mp4 --entry 0.2 --exit 0.8

# Combine options
python test_visual_pro.py video.mp4 --workers 5 --confidence 0.4 --entry 0.25 --exit 0.75
```

## Features Explained

### Multi-Threaded Processing

- **3 workers** (default): ~60 FPS processing capacity (20 FPS per worker)
- **5 workers**: ~100 FPS processing capacity (20 FPS per worker)
- Each worker runs AI detection in parallel
- Non-blocking queues prevent frame drops
- **Result**: Smooth 30 FPS video playback with real-time AI

### DeepSORT Tracking

- Assigns unique ID to each vehicle (e.g., ID:1, ID:2, ID:3)
- Tracks vehicles across frames using motion + appearance
- Handles occlusion and re-entry
- **No duplicate counting** - same car = same ID throughout video

### Bidirectional Counting

**Entry Line** (default 30% from top):
- Detects when unique vehicle crosses entry line
- Counts as "IN"
- Only counted once per unique vehicle

**Exit Line** (default 70% from top):
- Detects when unique vehicle crosses exit line
- Counts as "OUT"
- Only counted if vehicle previously crossed entry line

**Graceful Handling**:
- Vehicles that enter but never exit: Counted in "Still in frame"
- Vehicles that appear mid-frame: Tracked but not counted until they cross entry
- Lost tracks: Automatically removed after 30 frames without detection

### Statistics Display

**Left Panel:**
- Display FPS (video playback speed)
- AI Processing time per frame
- Number of workers
- Video progress
- **Unique Vehicles**: Total unique vehicles seen
- **Vehicles IN**: Crossed entry line
- **Vehicles OUT**: Crossed exit line
- **Active Now**: Currently visible vehicles
- **Still in frame**: IN - OUT (vehicles that entered but haven't exited)

**Right Panel:**
- Vehicle type breakdown (cars, trucks, buses, etc.)

### Counting Lines

- **Cyan lines** drawn on video
- **Entry line**: Labeled "ENTRY LINE"
- **Exit line**: Labeled "EXIT LINE"
- Configurable via `--entry` and `--exit` parameters

## Controls

While running:
- **SPACE** - Pause/Resume
- **Arrow Left** - Skip backward 1 second
- **Arrow Right** - Skip forward 1 second
- **Page Down** - Skip backward 10 seconds
- **Page Up** - Skip forward 10 seconds
- **+/-** - Adjust confidence threshold
- **Q or ESC** - Quit

## Performance Tuning

### For Maximum Speed (30+ FPS):

```powershell
# Use 5 workers + lower confidence
python test_visual_pro.py video.mp4 --workers 5 --confidence 0.4
```

### For Maximum Accuracy:

```powershell
# Use 3 workers + higher confidence
python test_visual_pro.py video.mp4 --workers 3 --confidence 0.6
```

### For High Traffic Videos:

```powershell
# More workers + moderate confidence + wider entry/exit lines
python test_visual_pro.py video.mp4 --workers 5 --confidence 0.5 --entry 0.25 --exit 0.75
```

## Understanding the Output

### During Playback:

```
✅ Vehicle #5 (car) ENTERED
🚪 Vehicle #5 (car) EXITED
✅ Vehicle #12 (truck) ENTERED
```

### Final Statistics:

```
📊 SESSION STATISTICS
======================================================================
Total frames displayed: 3000
Frames processed with AI: 3000

🚗 VEHICLE TRACKING:
   Unique vehicles seen: 45
   Vehicles ENTERED (IN): 42
   Vehicles EXITED (OUT): 38
   Still in frame: 4

🚙 Vehicle Types:
   car: 38
   truck: 5
   motorcycle: 2
======================================================================
```

**Interpretation:**
- 45 unique vehicles detected throughout video
- 42 crossed the entry line (counted as IN)
- 38 crossed the exit line (counted as OUT)
- 4 vehicles entered but never exited (still in frame or left before exit line)
- 3 vehicles appeared mid-frame (never crossed entry line)

## Troubleshooting

### "ModuleNotFoundError: No module named 'deep_sort_realtime'"

Install dependencies:
```powershell
pip install --timeout 1000 --retries 10 deep-sort-realtime filterpy scipy
```

### Video still laggy

Increase workers:
```powershell
python test_visual_pro.py video.mp4 --workers 5
```

### Too many false detections

Increase confidence:
```powershell
python test_visual_pro.py video.mp4 --confidence 0.6
```

### Vehicles counted multiple times

This shouldn't happen with DeepSORT! If it does:
1. Check that tracking is working (unique IDs visible on bounding boxes)
2. Try lower confidence to improve tracking: `--confidence 0.4`
3. Make sure AI service is running properly

### GPU Memory Issues

Reduce workers:
```powershell
python test_visual_pro.py video.mp4 --workers 2
```

## Comparison: Basic vs Professional

| Feature | test_visual.py (Basic) | test_visual_pro.py (Professional) |
|---------|------------------------|-----------------------------------|
| Processing Speed | ~10-20 FPS | 30+ FPS |
| Workers | 1 thread | 3-5 threads |
| Vehicle Tracking | ❌ No | ✅ DeepSORT |
| Unique IDs | ❌ No | ✅ Yes |
| Duplicate Counting | ❌ Yes (same car counted 90x) | ✅ No (counted once) |
| Bidirectional Count | ❌ No | ✅ IN/OUT |
| Line Crossing | ❌ No | ✅ Yes |
| GPU Optimization | Partial | ✅ FP16 + Optimized |
| Recommended For | Quick testing | Production, accurate analytics |

## Next Steps: Multi-Camera Support

This architecture is ready for multi-camera tracking! To add Camera 1 ↔ Camera 2 tracking:

1. Modify `detect_frame()` to include camera ID
2. Use DeepSORT's cross-camera re-identification
3. Track vehicles as they move from Camera 1 to Camera 2
4. Count IN when entering Camera 1, OUT when exiting Camera 2

The vehicle state management is already designed for this - just need to extend it!

## Tips

1. **Start with defaults**: `python test_visual_pro.py video.mp4`
2. **Adjust counting lines** based on your camera angle
3. **Use 5 workers** if you have a powerful GPU
4. **Lower confidence** (0.3-0.4) for better tracking in challenging conditions
5. **Higher confidence** (0.6-0.7) for cleaner detections in good lighting

## License

Part of Road Sentinel Traffic Monitoring System
