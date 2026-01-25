# YOLOv8 Training Pipeline

This folder contains scripts for training custom YOLOv8 models on the COCO dataset for vehicle detection.

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

### 2. Install Dependencies

```bash
# Install all training dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check all dependencies
python -c "from ultralytics import YOLO; import torch; print('✅ Setup complete!')"

# Check GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

## Usage

### Quick Start Training

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
