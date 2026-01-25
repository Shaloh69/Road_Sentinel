# Batch Video Frame Extractor

This folder contains utilities for extracting frames from videos to create datasets for AI/ML training.

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Navigate to this folder
cd scripts/extract_frames

# Create virtual environment
python3 -m venv venv_frames

# Activate virtual environment
# On Linux/Mac:
source venv_frames/bin/activate
# On Windows:
# venv_frames\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test OpenCV installation
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## Usage

### Extract from Single Video

```bash
# Activate environment
source venv_frames/bin/activate

# Extract frames at 30 FPS
python extract_frames.py video.mp4 -o output_frames -f 30

# Extract frames at 60 FPS
python extract_frames.py video.mp4 -o output_frames -f 60
```

### Batch Process Multiple Videos

```bash
# Process all videos in a folder
python extract_frames.py videos_folder/ -o dataset_frames -f 30

# Save all frames to one folder (no subfolders)
python extract_frames.py videos_folder/ -o all_frames --no-subfolders
```

### Examples

```bash
# Process Philippine road videos
python extract_frames.py ~/Videos/philippine_roads/ -o road_dataset -f 30

# Extract from dashcam footage at high FPS
python extract_frames.py dashcam.mp4 -o dashcam_frames -f 60

# Process multiple videos, separate folders for each
python extract_frames.py traffic_videos/ -o extracted_dataset -f 30
```

## Command Line Options

```
python extract_frames.py INPUT [OPTIONS]

Arguments:
  INPUT                 Input video file or folder containing videos

Options:
  -o, --output DIR     Output directory (default: extracted_frames)
  -f, --fps INT        Target FPS for extraction (default: 60)
  --no-subfolders      Save all frames to one folder without subfolders
  -h, --help           Show help message
```

## Output Structure

### With Subfolders (default)
```
output_folder/
├── video1/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── video2/
│   ├── frame_000000.jpg
│   └── ...
└── video3/
    └── ...
```

### Without Subfolders (--no-subfolders)
```
output_folder/
├── video1_frame_000000.jpg
├── video1_frame_000001.jpg
├── video2_frame_000000.jpg
└── ...
```

## Files

- `extract_frames.py` - Main frame extraction script
- `requirements.txt` - Python dependencies (opencv-python, numpy)
- `README.md` - This file

## Features

- ✅ Batch processing of multiple videos
- ✅ Automatic FPS calculation and frame interval
- ✅ Progress tracking and statistics
- ✅ Supports common video formats (MP4, AVI, MOV, MKV, etc.)
- ✅ Organized output with optional subfolders
- ✅ High-quality JPEG output (95% quality)

## Supported Video Formats

- `.mp4` - MPEG-4
- `.avi` - Audio Video Interleave
- `.mov` - QuickTime
- `.mkv` - Matroska
- `.flv` - Flash Video
- `.wmv` - Windows Media Video
- `.m4v` - MPEG-4 Video
- `.webm` - WebM

## Use Cases

### 1. Dataset Creation for Training
```bash
# Extract frames for YOLO training
python extract_frames.py training_videos/ -o yolo_dataset/images -f 10
```

### 2. Video Analysis
```bash
# Extract key frames for manual review
python extract_frames.py surveillance.mp4 -o key_frames -f 1
```

### 3. Data Augmentation
```bash
# Extract high-FPS frames for diverse dataset
python extract_frames.py source_video.mp4 -o augmented_data -f 60
```

## Performance Tips

- **Higher FPS** = More frames = Larger dataset
- **Lower FPS** = Fewer frames = Faster processing
- **JPEG Quality** = 95% (good balance of quality and size)
- **Storage:** Plan for ~100-500KB per frame

## Example Workflow

```bash
# 1. Activate environment
source venv_frames/bin/activate

# 2. Extract frames from Philippine road videos
python extract_frames.py ~/Videos/ph_roads/ -o road_frames -f 30

# 3. Check results
ls -lh road_frames/

# 4. Use frames for YOLO training
# Copy frames to your training dataset

# 5. Deactivate when done
deactivate
```

## Troubleshooting

### Video Won't Open
- Check file path and format
- Verify video isn't corrupted
- Ensure opencv-python is installed

### Out of Disk Space
- Reduce FPS: `-f 10` instead of `-f 60`
- Process fewer videos
- Clear output folder before running

### Slow Processing
- Normal for large videos
- Processing speed depends on video size and resolution
- Typical: 30-60 FPS processing speed

## Virtual Environment Management

```bash
# Activate
source venv_frames/bin/activate  # Linux/Mac
# venv_frames\Scripts\activate   # Windows

# Deactivate
deactivate

# Remove (if needed)
rm -rf venv_frames
```

## Statistics Example

```
======================================================================
PROCESSING SUMMARY
======================================================================
Total videos found: 5
Successfully processed: 5
Failed: 0
Total frames extracted: 12,450
Processing time: 45.23 seconds
----------------------------------------------------------------------

Processed Videos:
  • traffic1.mp4: 2,500 frames → road_frames/traffic1/
  • traffic2.mp4: 3,200 frames → road_frames/traffic2/
  • ...
======================================================================
```

## Notes

- Frames are saved as JPEG (95% quality)
- Frame naming: `frame_XXXXXX.jpg` (zero-padded)
- Original video FPS is auto-detected
- Frame interval calculated automatically
- Requires ~100-500KB per frame storage
