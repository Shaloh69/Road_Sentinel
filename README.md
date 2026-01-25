# 🚗 Road Sentinel - Vehicle Speed Detection System

An AI-powered vehicle speed detection system using YOLOv8 for real-time vehicle detection and tracking.

## 🎯 Overview

This project is a **thesis project** for vehicle speed detection in Philippine road conditions. It uses state-of-the-art YOLOv8 object detection to identify and track vehicles (bicycles, cars, motorcycles, buses, trucks) and estimate their speeds from video footage.

## ✨ Features

- **Real-time vehicle detection** using YOLOv8
- **Multi-class detection**: bicycle, car, motorcycle, bus, truck
- **Speed estimation** from video footage
- **Vehicle tracking** across video frames
- **Pre-trained on COCO dataset** (no manual annotation needed)
- **GPU accelerated** (also works on CPU)

## 📁 Project Structure

```
Road_Sentinel/
├── scripts/
│   ├── download/
│   │   └── auto_download_coco.py      # Complete speed detector implementation
│   ├── extract_frames/
│   │   ├── extract_frames.py          # Batch frame extraction utility
│   │   └── requirements.txt
│   └── training/
│       ├── train_vehicle_detector.py  # Full training pipeline
│       └── quick_train.py             # Quick start training
├── runs/                               # Training outputs (generated)
├── requirements.txt                    # Python dependencies
├── verify_setup.py                     # Setup verification script
├── TRAINING_GUIDE.md                   # Comprehensive training guide
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Verify Installation

```bash
# Check if all dependencies are installed
python verify_setup.py
```

### 2. Train the Model

**Option A: Quick Training (Simplest)**
```bash
cd scripts/training
python quick_train.py
```

**Option B: Full Control**
```bash
cd scripts/training
python train_vehicle_detector.py --model n --epochs 100 --batch 16
```

### 3. Test the Model

```bash
python train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source your_video.mp4
```

### 4. Run Speed Detection

```bash
python scripts/download/auto_download_coco.py
```

## 📋 Prerequisites

### Hardware
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB free space minimum
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support)

### Software
- **Python**: 3.8 or higher (3.11 recommended)
- **CUDA**: 11.8+ (for GPU training)
- **OS**: Linux, macOS, or Windows

## 📦 Installation

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python verify_setup.py
```

**What gets installed:**
- `ultralytics` - YOLOv8 framework
- `torch` & `torchvision` - PyTorch
- `opencv-python` - Computer vision
- `numpy`, `pandas` - Data processing
- Additional utilities

## 📊 Dataset

The system uses the **COCO 2017 dataset**:
- **330,000+ images** with vehicle annotations
- **5 vehicle classes**: bicycle, car, motorcycle, bus, truck
- **~77,000 vehicle annotations**
- **Auto-downloads** when you start training (no manual download needed)
- **Size**: ~20GB

## 🎓 For Thesis Students

### Why COCO Dataset?

✅ **Industry standard** - academically recognized
✅ **Large-scale** - sufficient data for robust training
✅ **Pre-annotated** - no manual labeling required
✅ **Generalizes well** - works globally including Philippines
✅ **Open source** - free to use for research

### Citation

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  booktitle={ECCV},
  year={2014}
}
```

## 📖 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Comprehensive training guide
  - Dataset information
  - Training options and parameters
  - Troubleshooting
  - Performance optimization
  - Best practices for thesis

## 🔧 Troubleshooting

### Import Error: "YOLO is not exported from module 'ultralytics'"

**Problem:** This Pylance error indicates `ultralytics` is not installed.

**Solution:**
```bash
pip install ultralytics
```

**Verification:**
```python
from ultralytics import YOLO  # This should work
```

**Note:** The correct import is `from ultralytics import YOLO`, NOT `from .yolo import YOLO`.

### CUDA Out of Memory

**Problem:** GPU runs out of memory during training.

**Solutions:**
1. Reduce batch size: `--batch 8` or `--batch 4`
2. Use smaller model: `--model n`
3. Reduce image size: `--imgsz 416`

```bash
python train_vehicle_detector.py --model n --batch 4 --imgsz 416
```

### Slow Training

**Problem:** Training is very slow.

**Solutions:**
- **No GPU:** Consider using cloud GPU (Google Colab, Kaggle, AWS)
- **Reduce epochs:** `--epochs 50` for quick experiments
- **Use smaller model:** `--model n` (nano)

## 📈 Model Performance

Training time estimates (100 epochs, YOLOv8n):

| Hardware | Training Time |
|----------|--------------|
| RTX 4090 | 4-6 hours |
| RTX 3080 | 6-10 hours |
| RTX 3060 Ti | 10-15 hours |
| GTX 1660 Ti | 15-24 hours |
| CPU only | 5-7 days ⚠️ |

## 🎯 Usage Examples

### Training

```bash
# Default training
python scripts/training/train_vehicle_detector.py

# Custom configuration
python scripts/training/train_vehicle_detector.py \
  --model m \
  --epochs 200 \
  --batch 16 \
  --name my_experiment
```

### Testing

```bash
# Test on video
python scripts/training/train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source traffic_video.mp4

# Test on image
python scripts/training/train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source test_image.jpg
```

### Speed Detection

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/vehicle_speed/coco_v1/weights/best.pt')

# Run detection and tracking
results = model.track(
    source='traffic_video.mp4',
    save=True,
    conf=0.25,
    classes=[1, 2, 3, 5, 7]  # vehicles only
)
```

## 🛠️ Utilities

### Frame Extraction

Extract frames from videos for dataset creation:

```bash
cd scripts/extract_frames

# Extract from single video
python extract_frames.py video.mp4 -o output_folder -f 30

# Batch process multiple videos
python extract_frames.py videos_folder/ -o frames_output -f 60
```

## 📞 Support

- **YOLOv8 Docs:** https://docs.ultralytics.com
- **COCO Dataset:** https://cocodataset.org
- **PyTorch:** https://pytorch.org/docs

## 🤝 Contributing

This is a thesis project. For questions or issues:
1. Check the [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. Run `python verify_setup.py` to diagnose issues
3. Review error messages and logs

## 📝 License

This project uses:
- **YOLOv8** (AGPL-3.0)
- **COCO Dataset** (CC BY 4.0)
- **PyTorch** (BSD)

For thesis and educational use.

---

## 🚦 Next Steps

1. ✅ **Verify setup:** `python verify_setup.py`
2. 📚 **Read guide:** `TRAINING_GUIDE.md`
3. 🚀 **Start training:** `python scripts/training/quick_train.py`
4. 🧪 **Test model:** Test on your Philippine road videos
5. 📊 **Document results:** Record metrics for thesis

---

**Good luck with your thesis! 🎓**
