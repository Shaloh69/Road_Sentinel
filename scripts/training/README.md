# YOLOv8 Training Pipeline

This folder contains scripts for training custom YOLOv8 models for vehicle detection, speed measurement, and crash detection.

## 📊 Dataset Options (Choose One)

### ⭐ **RECOMMENDED: Roboflow Universe** (Easiest!)
- ✅ Already in YOLO format (no conversion needed!)
- ✅ 50,000+ datasets available
- ✅ Traffic surveillance, night vision, overhead camera datasets
- ✅ Setup time: 5 minutes
- ✅ Training time: 3-4 hours
- 📖 **See:** [YOLO_NATIVE_DATASETS.md](YOLO_NATIVE_DATASETS.md)

### Option 2: AI City Challenge (Best for Overhead Cameras)
- 🎯 Perfect for traffic surveillance and angled cameras
- 🎯 Includes speed estimation ground truth
- ⚠️ Requires conversion script (30-60 min setup)
- 📖 **See:** [OVERHEAD_CAMERA_GUIDE.md](OVERHEAD_CAMERA_GUIDE.md)

### Option 3: COCO Dataset (General Object Detection)
- 🎯 General purpose, includes vehicles
- ⚠️ Also includes pizzas, dogs, etc. (80 classes)
- ⚠️ Not specialized for traffic surveillance
- 📖 Covered in this README below

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Navigate to this folder
cd scripts/training

# Create virtual environment
python3 -m venv venv_training

# Activate virtual environment
# On Linux/Mac:
source venv_training/bin/activate
# On Windows:
# venv_training\Scripts\activate
```

### 2. Install PyTorch with GPU Support (If you have NVIDIA GPU)

⚠️ **IMPORTANT:** Python 3.8-3.12 required (NOT 3.13+)

```bash
# For CUDA 12.1+ (RTX 30/40 series - RECOMMENDED)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

### 3. Install Other Dependencies

```bash
# Install all training dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check all dependencies
python -c "from ultralytics import YOLO; import torch; print('✅ Setup complete!')"

# Check GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

## Usage

### 🚀 Quick Start with Roboflow (RECOMMENDED)

**Fastest way to train for your Busay project!**

1. **Get FREE API key:**
   - Go to: https://roboflow.com
   - Sign up (free)
   - Settings → API → Copy key

2. **Browse datasets:**
   - Go to: https://universe.roboflow.com
   - Search for: "traffic surveillance overhead"
   - Find dataset with 5,000+ images

3. **Download and train:**
   ```bash
   # Run the download guide
   python download_roboflow_datasets.py

   # Follow the instructions, then train:
   from roboflow import Roboflow
   from ultralytics import YOLO

   # Download dataset (already in YOLO format!)
   rf = Roboflow(api_key='YOUR_API_KEY')
   project = rf.workspace('workspace-name').project('project-name')
   dataset = project.version(1).download('yolov8')

   # Train immediately - no conversion needed!
   model = YOLO('yolov8n.pt')
   model.train(data=f'{dataset.location}/data.yaml', epochs=100, batch=4)
   ```

**Total time:** 5 min setup + 3-4 hours training = ✅ Production ready in one day!

📖 **Full guide:** [YOLO_NATIVE_DATASETS.md](YOLO_NATIVE_DATASETS.md)

---

### Quick Start with COCO (Alternative)

```bash
# Activate environment
source venv_training/bin/activate

# Run quick training (simplest method)
python quick_train.py
```

This will:
- Auto-download COCO dataset (~20GB)
- Auto-download YOLOv8 weights
- Start training immediately
- Save results to `runs/vehicle_speed/quick_v1/`

### Advanced Training

```bash
# Train with custom parameters
python train_vehicle_detector.py --model n --epochs 100 --batch 16

# Available options:
#   --model {n,s,m,l,x}  - Model size (n=nano, fastest)
#   --epochs INT         - Number of training epochs
#   --batch INT          - Batch size (reduce if out of memory)
#   --imgsz INT          - Input image size
#   --project STR        - Project name
#   --name STR           - Experiment name
```

### Test Trained Model

```bash
# Test on a video
python train_vehicle_detector.py \
  --test \
  --model-path runs/vehicle_speed/coco_v1/weights/best.pt \
  --source your_video.mp4
```

## Files

- `train_vehicle_detector.py` - Full training pipeline with options
- `quick_train.py` - Simplified one-command training
- `requirements.txt` - Python dependencies for training
- `README.md` - This file

## Training Options

### Model Sizes

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | Nano | Fastest | Good | Real-time, mobile |
| YOLOv8s | Small | Fast | Better | Balanced performance |
| YOLOv8m | Medium | Moderate | Best | High accuracy |
| YOLOv8l | Large | Slow | Excellent | Research, offline |
| YOLOv8x | XLarge | Slowest | Best | Maximum accuracy |

### Training Time Estimates (100 epochs, YOLOv8n)

| Hardware | Time |
|----------|------|
| RTX 4090 | 4-6 hours |
| RTX 3080 | 6-10 hours |
| RTX 3060 Ti | 10-15 hours |
| GTX 1660 Ti | 15-24 hours |
| CPU only | 5-7 days ⚠️ |

### Batch Size Guidelines

| GPU VRAM | Recommended Batch |
|----------|------------------|
| 12GB+ | 16-32 |
| 8GB | 16 |
| 6GB | 8 |
| 4GB | 4 |
| CPU | 2-4 |

## Dataset

### COCO 2017
- **Auto-downloads** on first training run
- **Size:** ~20GB
- **Images:** 330,000+ with annotations
- **Vehicle classes:** bicycle, car, motorcycle, bus, truck
- **Storage needed:** ~30GB total (dataset + outputs)

## Output Structure

After training:
```
runs/vehicle_speed/your_experiment/
├── weights/
│   ├── best.pt          ← Use this for deployment!
│   └── last.pt          ← Last checkpoint
├── results.png          ← Training curves
├── confusion_matrix.png ← Performance metrics
└── ...
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_vehicle_detector.py --batch 4

# Or use smaller model
python train_vehicle_detector.py --model n --batch 8
```

### Slow Training (CPU)
- Consider using free GPU: Google Colab, Kaggle
- Reduce epochs for testing: `--epochs 50`
- Use pre-trained model without fine-tuning

### Dataset Won't Download
- Check internet connection
- Ensure ~30GB free space
- Download will resume if interrupted

## Virtual Environment Management

```bash
# Activate environment
source venv_training/bin/activate  # Linux/Mac
# venv_training\Scripts\activate   # Windows

# Deactivate when done
deactivate

# Remove environment (if needed)
rm -rf venv_training
```

## Notes

- **First run:** Will download COCO dataset (~20GB)
- **GPU recommended:** CPU training is very slow
- **Storage:** Ensure 30GB+ free space
- **RAM:** 8GB minimum, 16GB+ recommended
- **Results:** Saved in `runs/` folder (auto-created)

## Next Steps

1. Train a model with this folder's environment
2. Copy `best.pt` to `../download/` folder
3. Update `auto_download_coco.py` to use your trained model
4. Run speed detection with custom-trained model
