# 🚗 Road Sentinel - Vehicle Speed Detection System

An AI-powered vehicle speed detection system using YOLOv8 for real-time vehicle detection and tracking.

## 🎯 Overview

This project is a **thesis project** for a blind curve warning system at Barangay Busay, Cebu, Philippines. It uses state-of-the-art YOLOv8 object detection to create a dual-camera safety system that:

- **Detects vehicles** approaching from both sides of a blind curve
- **Measures vehicle speed** to warn speeding drivers
- **Identifies crashes/anomalies** for emergency response
- **Displays LED warnings** to alert approaching drivers
- **Operates day and night** with infrared camera support

### System Architecture
```
[Camera A] ←─── Blind Curve ───→ [Camera B]
     ↓                                ↓
  Detection                       Detection
     ↓                                ↓
  Speed Est.                     Speed Est.
     ↓                                ↓
     └────────→ [Control System] ←────┘
                      ↓
              [LED Display Warnings]
              • ✅ Safe - No incoming
              • ⚠️ Slow down - Vehicle!
              • 🚨 Speed warning
              • ❌ Accident ahead!
```

## ✨ Features

### Vehicle Detection & Tracking
- **Real-time detection** using YOLOv8
- **Multi-class detection**: car, motorcycle, bicycle, bus, truck
- **Multi-camera tracking** across different viewpoints
- **Overhead/angled camera support** with perspective correction
- **Night vision capability** for 24/7 operation

### Speed & Safety Monitoring
- **Accurate speed estimation** with homography transformation
- **Crash/anomaly detection** for safety incidents
- **Dual-model architecture**:
  - Model 1: Vehicle detection & speed tracking
  - Model 2: Crash/anomaly detection
- **Database logging** (MySQL) for event tracking

### Flexible Training Options
- **⭐ Roboflow Universe** (RECOMMENDED) - 5 min setup, YOLO-native format
- **AI City Challenge** - Perfect for overhead cameras, includes speed data
- **COCO Dataset** - General purpose, pre-trained models available
- **Custom datasets** - Fine-tune on your own footage
- **GPU accelerated** (also works on CPU)

## 📁 Project Structure

```
Road_Sentinel/
├── 📂 training/                 # 🚀 YOLO26 Model Training
│   ├── train.py                 # Main training script (YOLO26 optimized)
│   ├── validate.py              # Model validation & testing
│   ├── merge_busay_datasets.py # Dataset merger
│   ├── analyze_datasets.py     # Dataset analysis
│   ├── requirements.txt         # Training dependencies
│   ├── .venv/                   # Isolated environment
│   └── *.md                     # Training guides
│
├── 📂 testing/                  # 🧪 Testing Suite
│   ├── test_camera.py           # Camera testing with auto-detection
│   ├── test_visual.py           # Visual GUI testing
│   ├── test_video.py            # Video file testing
│   ├── test_images.py           # Image testing
│   ├── test_api.py              # API endpoint testing
│   ├── requirements.txt         # Testing dependencies
│   └── .venv/                   # Isolated environment
│
├── 📂 inference/                # 🚗 Production Inference
│   ├── speed_detection.py       # Multi-vehicle speed detection
│   ├── camera_calibration.py   # Perspective calibration
│   ├── extract_frames.py        # Frame extraction utility
│   ├── requirements.txt         # Inference dependencies
│   └── .venv/                   # Isolated environment
│
├── 📂 server/                   # 🖥️ Backend Services
│   ├── ai-service/              # FastAPI AI microservice
│   ├── node-service/            # Node.js backend
│   └── database/                # MySQL schema
│
├── 📂 client/                   # 💻 Frontend
│   └── web/                     # Next.js React app
│
├── 📂 datasets/                 # 🗂️ Training Data
│   ├── downloaded/              # Raw datasets
│   └── processed/               # Ready-to-train
│
├── 📂 models/                   # 🎯 Trained Models
│   ├── v1/, v2/                 # Versions
│   └── production/              # Production models
│
├── 📖 CAMERA_TEST_GUIDE.md      # Camera testing guide
├── 📖 PROJECT_STRUCTURE.md      # Detailed structure
└── 📖 TRAINING_GUIDE.md         # Training documentation
```

✨ **Clean, organized structure with isolated environments for each component!**

## 🚀 Quick Start for Busay Project

### ⭐ **RECOMMENDED WORKFLOW** (Your Downloaded Datasets)

**Complete workflow for your 3 Roboflow datasets:**

```bash
# 1. Place downloaded datasets
mv ~/Downloads/Traffic-surveillance-system-1 datasets/downloaded/
mv ~/Downloads/Vehicle-Detection-Day-Night-1 datasets/downloaded/
mv ~/Downloads/Accident-detection-1 datasets/downloaded/

# 2. Setup environment
cd training
python3 -m venv venv_training
source venv_training/bin/activate  # Linux/Mac
# venv_training\Scripts\activate   # Windows

# 3. Install PyTorch with GPU (Python 3.8-3.12 required, NOT 3.13!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt

# 5. Merge datasets (automatic!)
python run_merge_busay.py
# This creates:
#   - datasets/processed/busay_vehicle_detection/
#   - datasets/processed/busay_accident_detection/

# 6. Train Model 1 (Vehicle Detection)
python train.py \
  --data ../../datasets/processed/busay_vehicle_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name vehicle_detection

# 7. Train Model 2 (Crash Detection)
python train.py \
  --data ../../datasets/processed/busay_accident_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name crash_detection
```

**Total time:**
- Setup + merge: ~10 min
- Training Model 1: ~6-8 hours
- Training Model 2: ~3-4 hours
- **Total: ~10-12 hours** (can run both models in parallel!)

**Output:**
- `models/v1/vehicle_detection/weights/best.pt`
- `models/v1/crash_detection/weights/best.pt`

📖 **Detailed guide:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

### Option B: AI City Challenge (Advanced - Best for Overhead Cameras)

```bash
# 1. Register and download AI City Challenge dataset
# https://www.aicitychallenge.org/

# 2. Convert to YOLO format
cd training
python convert_aicity_track1_to_yolo.py  # Vehicle tracking
python convert_aicity_track4_to_yolo.py  # Crash detection

# 3. Train dual models
# See: training/DUAL_MODEL_TRAINING_GUIDE.md
```

**Total time:** 30-60 min conversion + 10-13 hours training for both models

📖 **Full guide:** [training/OVERHEAD_CAMERA_GUIDE.md](training/OVERHEAD_CAMERA_GUIDE.md)

---

### Option C: COCO Dataset (Quick Test - Not Specialized)

```bash
# Quick training on COCO (includes all 80 classes)
cd training
source venv_training/bin/activate
# Note: quick_train.py removed - use train.py instead
python train.py --dataset vehicle --model-size n --epochs 50
```

⚠️ **Note:** COCO includes pizzas, dogs, etc. Use Roboflow or AI City for traffic-specific training.

---

### Testing Your Trained Model

```bash
# Test on a video file
cd scripts/download
python auto_download_coco.py  # Has speed detection built-in

# Or use your trained weights
python train.py \
  --test \
  --model-path ../runs/vehicle_speed/roboflow_busay_v1/weights/best.pt \
  --source your_busay_video.mp4
```

### Camera Calibration (For Overhead/Angled Cameras)

```bash
# Calibrate perspective for accurate speed measurement
cd scripts/download
python angled_camera_calibration.py
# Follow interactive calibration steps
```

## 📋 Prerequisites

### Hardware
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB free space minimum
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support)

### Software
- **Python**: 3.8 - 3.12 (⚠️ **NOT 3.13+**) - PyTorch with CUDA doesn't support 3.13 yet!
  - Recommended: Python 3.11 or 3.12
- **CUDA**: 11.8+ (for GPU training, CUDA 12.1+ recommended for RTX 30/40 series)
- **OS**: Linux, macOS, or Windows

## 📦 Installation

### ⚡ Quick Setup (Global Environment)

```bash
# Install all required packages globally
pip install -r requirements.txt

# Verify installation
python verify_setup.py
```

### 🎯 Recommended Setup (Isolated Environments)

**Each script folder has its own virtual environment to avoid conflicts:**

```bash
# Setup all environments automatically
./setup_all_environments.sh
```

**Or setup individually:**

```bash
# 1. Frame Extraction (lightweight, ~150MB)
cd scripts/extract_frames
python3 -m venv venv_frames
source venv_frames/bin/activate
pip install -r requirements.txt

# 2. Speed Detection (~2-3GB)
cd scripts/download
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Model Training (~3GB)
cd training
python3 -m venv venv_training
source venv_training/bin/activate
pip install -r requirements.txt
```

See **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** for detailed instructions.

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
python train.py --model n --batch 4 --imgsz 416
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

## 📚 Documentation

### Training Guides
- **[YOLO_NATIVE_DATASETS.md](scripts/training/YOLO_NATIVE_DATASETS.md)** ⭐ - Roboflow Universe guide (RECOMMENDED)
- **[OVERHEAD_CAMERA_GUIDE.md](training/OVERHEAD_CAMERA_GUIDE.md)** - Angled camera setup & AI City Challenge
- **[DUAL_MODEL_TRAINING_GUIDE.md](training/DUAL_MODEL_TRAINING_GUIDE.md)** - Complete dual-model system
- **[NIGHT_VISION_DATASETS.md](scripts/training/NIGHT_VISION_DATASETS.md)** - Night vision & infrared datasets
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Comprehensive training guide

### External Resources
- **Roboflow Universe:** https://universe.roboflow.com (50,000+ datasets)
- **AI City Challenge:** https://www.aicitychallenge.org
- **YOLOv8 Docs:** https://docs.ultralytics.com
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

## 🚦 Next Steps for Busay Project

### Recommended Path (Fastest to Production)

1. ✅ **Setup Environment**
   ```bash
   cd training
   python3 -m venv venv_training
   source venv_training/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

2. 📥 **Get Dataset** (Choose one)
   - **⭐ RECOMMENDED:** [Roboflow Universe](https://universe.roboflow.com) → Search "traffic surveillance overhead"
   - **Alternative:** [AI City Challenge](https://www.aicitychallenge.org) → Download Track 1 & 4

3. 🚀 **Train Models**
   ```bash
   # For Roboflow (already YOLO format)
   python train.py --data roboflow_data.yaml --epochs 100 --batch 4

   # For AI City (after conversion)
   python convert_aicity_track1_to_yolo.py
   python train.py --data aicity_2022_track1.yaml --epochs 100 --batch 4
   ```

4. 📐 **Calibrate Camera**
   ```bash
   cd ../download
   python angled_camera_calibration.py
   # Follow prompts to calibrate overhead camera perspective
   ```

5. 🧪 **Test on Busay Videos**
   ```bash
   python auto_download_coco.py  # Use your trained model
   ```

6. 📊 **Document for Thesis**
   - Record detection accuracy (mAP, precision, recall)
   - Measure speed estimation error (±KPH)
   - Test day/night performance
   - Log crash detection results

### Quick References

📖 **First time?** Read [YOLO_NATIVE_DATASETS.md](scripts/training/YOLO_NATIVE_DATASETS.md)
📖 **Overhead camera?** Read [OVERHEAD_CAMERA_GUIDE.md](training/OVERHEAD_CAMERA_GUIDE.md)
📖 **Need help?** Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

**Good luck with your Busay blind curve system! 🎓🚗**
