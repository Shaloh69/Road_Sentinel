# Road Sentinel - Vehicle Speed Detection Training Guide

## 🎯 Project Overview

This project implements a **Vehicle Speed Detection System** using YOLOv8 for object detection. The system can detect and track vehicles (bicycles, cars, motorcycles, buses, trucks) and estimate their speed from video footage.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Training Options](#training-options)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended | Best |
|-----------|---------|-------------|------|
| **RAM** | 8 GB | 16 GB | 32 GB |
| **Storage** | 50 GB free | 100 GB free | 200 GB free |
| **GPU** | None (CPU) | GTX 1660 Ti+ | RTX 3080+ |
| **GPU VRAM** | N/A | 6 GB+ | 12 GB+ |

### Software Requirements

- **Python**: 3.8 or higher (3.11 recommended)
- **CUDA**: 11.8 or higher (for GPU training)
- **Operating System**: Linux, macOS, or Windows

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Navigate to project directory
cd /home/user/Road_Sentinel

# Install all required packages
pip install -r requirements.txt
```

**What gets installed:**
- ✅ `ultralytics` - YOLOv8 framework
- ✅ `torch` & `torchvision` - PyTorch deep learning framework
- ✅ `opencv-python` - Computer vision library
- ✅ `numpy`, `pandas` - Data processing
- ✅ `matplotlib`, `seaborn` - Visualization

### 2. Verify Installation

```bash
# Test YOLO import
python3 -c "from ultralytics import YOLO; print('✅ YOLO installed successfully!')"

# Check GPU availability
python3 -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

---

## 📊 Dataset Information

### COCO 2017 Dataset

We use the **COCO 2017** dataset which includes:

- **330,000+ images** with vehicle annotations
- **5 vehicle classes**: bicycle, car, motorcycle, bus, truck
- **~77,000+ vehicle annotations**
- **Pre-split** into train/validation/test sets
- **Industry standard** - academically recognized
- **Auto-download** - YOLOv8 handles it automatically

### Dataset Statistics

| Category | Training Images | Validation Images | Total Annotations |
|----------|----------------|-------------------|-------------------|
| Bicycle | ~7,000 | ~500 | ~8,500 |
| Car | ~12,000 | ~1,000 | ~43,000 |
| Motorcycle | ~8,000 | ~600 | ~9,500 |
| Bus | ~6,000 | ~300 | ~6,000 |
| Truck | ~9,000 | ~500 | ~10,000 |

**Total:** ~42,000 images with ~77,000+ vehicle annotations

### Storage Requirements

- COCO Dataset: **~20 GB**
- Training outputs: **~2-5 GB**
- **Total:** ~25-30 GB free space needed

---

## 🚀 Training Options

### Option 1: Quick Start (Simplest)

**One-line command** - Auto-downloads everything and starts training:

```bash
cd scripts/training
python quick_train.py
```

**What it does:**
1. Downloads YOLOv8 pretrained weights
2. Downloads COCO dataset (~20GB)
3. Trains on vehicle classes
4. Saves model to `runs/vehicle_speed/quick_v1/weights/best.pt`

### Option 2: Full Training Pipeline (Recommended)

**Complete control** with command-line arguments:

```bash
cd scripts/training

# Default training (YOLOv8n, 100 epochs)
python train_vehicle_detector.py

# Train with larger model and more epochs
python train_vehicle_detector.py --model s --epochs 150 --batch 32

# Train on CPU with smaller batch
python train_vehicle_detector.py --batch 4

# Train with custom settings
python train_vehicle_detector.py \
  --model m \
  --epochs 200 \
  --batch 16 \
  --imgsz 640 \
  --project my_project \
  --name experiment_1
```

**Available arguments:**

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | n, s, m, l, x | n | Model size (n=nano, fastest) |
| `--epochs` | 1-1000 | 100 | Number of training epochs |
| `--batch` | 1-64 | 16 | Batch size (reduce if OOM) |
| `--imgsz` | 320-1280 | 640 | Input image size |
| `--project` | string | vehicle_speed | Project directory name |
| `--name` | string | coco_v1 | Experiment name |

### Option 3: Command Line YOLO (Quickest)

```bash
# One-line training
yolo train data=coco.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
```

---

## 💡 Usage Examples

### Training Examples

```bash
# 1. Quick training with defaults
python quick_train.py

# 2. High accuracy training (larger model, more epochs)
python train_vehicle_detector.py --model m --epochs 200 --batch 16

# 3. Fast experimental training (small dataset subset)
python train_vehicle_detector.py --model n --epochs 50 --batch 32

# 4. Low memory training (for limited GPU/CPU)
python train_vehicle_detector.py --model n --epochs 100 --batch 4
```

### Testing Trained Model

```bash
# Test on a video
python train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source test_video.mp4 \
  --conf 0.25

# Test on an image
python train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source test_image.jpg

# Test on a folder of images
python train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source images_folder/
```

### Using Trained Model in Speed Detection

```python
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('runs/vehicle_speed/coco_v1/weights/best.pt')

# Run detection on video
results = model.track(
    source='traffic_video.mp4',
    save=True,
    conf=0.25,
    iou=0.45,
    classes=[1, 2, 3, 5, 7]  # vehicles only
)
```

---

## 📁 Project Structure

```
Road_Sentinel/
├── scripts/
│   ├── download/
│   │   └── auto_download_coco.py      # Speed detector implementation
│   ├── extract_frames/
│   │   ├── extract_frames.py          # Frame extraction tool
│   │   └── requirements.txt
│   └── training/
│       ├── train_vehicle_detector.py  # Full training pipeline
│       └── quick_train.py             # Quick start training
├── runs/
│   └── vehicle_speed/                 # Training outputs
│       └── [experiment_name]/
│           ├── weights/
│           │   ├── best.pt           # ⭐ Best model (use this!)
│           │   └── last.pt           # Last checkpoint
│           ├── results.png           # Training curves
│           ├── confusion_matrix.png  # Performance viz
│           └── ...
├── requirements.txt                   # Python dependencies
└── TRAINING_GUIDE.md                 # This file
```

---

## ⏱️ Training Time Estimates

| Hardware | YOLOv8n (100 epochs) | YOLOv8s | YOLOv8m |
|----------|---------------------|---------|---------|
| **RTX 4090** | 4-6 hours | 8-12 hours | 16-24 hours |
| **RTX 3080** | 6-10 hours | 12-18 hours | 24-36 hours |
| **RTX 3060 Ti** | 10-15 hours | 18-28 hours | 36-48 hours |
| **GTX 1660 Ti** | 15-24 hours | 28-40 hours | 48-72 hours |
| **CPU only** | 5-7 days ⚠️ | Not recommended | Not recommended |

### Batch Size Guidelines

| GPU VRAM | Recommended Batch Size |
|----------|----------------------|
| **12 GB+** | 16-32 |
| **8 GB** | 16 |
| **6 GB** | 8 |
| **4 GB** | 4 |
| **CPU** | 2-4 |

---

## 📈 Monitoring Training

During training, you'll see:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances   Size
  1/100    3.5G      1.234      0.987      1.456      123      640
  2/100    3.5G      1.156      0.892      1.389      145      640
  3/100    3.5G      1.098      0.834      1.323      134      640
```

**Good signs:**
- ✅ Losses decreasing steadily
- ✅ mAP increasing
- ✅ No "CUDA out of memory" errors

**Bad signs:**
- ❌ Losses not decreasing (stuck)
- ❌ "CUDA out of memory" → reduce batch size
- ❌ NaN values → reduce learning rate

---

## 🐛 Troubleshooting

### Import Error: "YOLO is not exported from module 'ultralytics'"

**Problem:** Pylance reports that YOLO cannot be imported.

**Solution:**
```bash
# Install ultralytics
pip install ultralytics

# Verify installation
python -c "from ultralytics import YOLO; print('Success!')"
```

**Note:** The correct import is `from ultralytics import YOLO`, not `from ultralytics.yolo import YOLO`.

### CUDA Out of Memory

**Problem:** GPU runs out of memory during training.

**Solutions:**
1. Reduce batch size: `--batch 8` or `--batch 4`
2. Use smaller model: `--model n` instead of `s` or `m`
3. Reduce image size: `--imgsz 416` instead of 640

```bash
# Low memory configuration
python train_vehicle_detector.py --model n --batch 4 --imgsz 416
```

### Slow Training on CPU

**Problem:** Training is extremely slow without GPU.

**Solutions:**
1. Use cloud GPU services (Google Colab, Kaggle, AWS)
2. Reduce epochs: `--epochs 50`
3. Use smaller dataset subset
4. Consider using a pre-trained model without fine-tuning

### Dataset Not Downloading

**Problem:** COCO dataset fails to download.

**Solutions:**
1. Check internet connection
2. Manually download from: http://cocodataset.org
3. Use Roboflow (see alternative in main guide)

---

## 🎓 Best Practices for Your Thesis

### 1. Dataset Documentation

- ✅ Clearly cite COCO dataset in your thesis
- ✅ Document the vehicle classes used
- ✅ Explain why COCO was chosen (industry standard, large-scale)

**Citation:**
```
Lin, T.-Y., et al. "Microsoft COCO: Common Objects in Context."
ECCV 2014. https://cocodataset.org
```

### 2. Model Selection

For a thesis project:
- **YOLOv8n** - Fast inference, good for real-time demo
- **YOLOv8s/m** - Better accuracy, acceptable speed
- **YOLOv8l/x** - Highest accuracy, slower (research focus)

### 3. Performance Metrics

Document these metrics:
- **mAP50** - Mean Average Precision at 50% IoU
- **mAP50-95** - Mean across multiple IoU thresholds
- **Precision** - Accuracy of positive predictions
- **Recall** - Coverage of actual positives
- **Speed** - Inference time (FPS)

### 4. Experimental Setup

Document:
- Hardware used (GPU model, RAM)
- Training time
- Hyperparameters
- Data splits (train/val/test)
- Performance on Philippine road videos

---

## 🌟 Next Steps

After training is complete:

1. **Test the model** on your Philippine road videos
2. **Integrate** with the speed detection system
3. **Calibrate** camera parameters for accurate speed measurement
4. **Evaluate** performance on real-world scenarios
5. **Document** results for your thesis

---

## 📞 Support & Resources

- **YOLOv8 Documentation:** https://docs.ultralytics.com
- **COCO Dataset:** https://cocodataset.org
- **PyTorch Docs:** https://pytorch.org/docs
- **Ultralytics GitHub:** https://github.com/ultralytics/ultralytics

---

## ✅ Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Quick training
python scripts/training/quick_train.py

# Full training with options
python scripts/training/train_vehicle_detector.py --model n --epochs 100 --batch 16

# Test trained model
python scripts/training/train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source test_video.mp4

# Run speed detection
python scripts/download/auto_download_coco.py
```

---

**Good luck with your thesis! 🎓🚗💨**
