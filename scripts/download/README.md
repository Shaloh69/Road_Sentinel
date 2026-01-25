# Vehicle Speed Detection System

This folder contains the vehicle speed detection implementation using YOLOv8.

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Navigate to this folder
cd scripts/download

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test YOLO import
python -c "from ultralytics import YOLO; print('✅ Setup complete!')"
```

## Usage

### Run Speed Detection

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Run the speed detector
python auto_download_coco.py
```

### Customize Detection

Edit `auto_download_coco.py` and modify:
- **Camera calibration** (PPM - pixels per meter)
- **Video source** (input file path)
- **Output settings** (save location)
- **Detection parameters** (confidence threshold)

## Files

- `auto_download_coco.py` - Main speed detection script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Dependencies

Core requirements:
- **ultralytics** - YOLOv8 framework
- **opencv-python** - Video processing
- **numpy** - Numerical calculations
- **torch** - Deep learning backend

## Troubleshooting

### Import Error
```bash
# If you get "No module named 'ultralytics'"
pip install -r requirements.txt
```

### GPU Not Detected
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If no GPU, the system will use CPU (slower but works).

### Video Won't Open
- Check video file path
- Verify video format (MP4, AVI, MOV supported)
- Ensure opencv-python is installed correctly

## Notes

- This script uses **pre-trained** YOLO models (no training needed)
- First run will download YOLOv8 weights (~6MB)
- Works with CPU or GPU (GPU recommended for real-time)
- Supports video files and live camera feeds
