# Models Folder

This folder contains all trained models for your Busay project, organized by version.

## 📁 Folder Structure

```
models/
├── v1/                          # Version 1 models
│   ├── vehicle_detection/
│   │   └── weights/
│   │       ├── best.pt          # Best model checkpoint
│   │       └── last.pt          # Latest checkpoint
│   └── crash_detection/
│       └── weights/
│           ├── best.pt
│           └── last.pt
│
├── v2/                          # Version 2 models (improved)
│   ├── vehicle_detection/
│   └── crash_detection/
│
└── production/                  # Production-ready models (symlinks)
    ├── vehicle_detector.pt -> ../v1/vehicle_detection/weights/best.pt
    └── crash_detector.pt -> ../v1/crash_detection/weights/best.pt
```

## 🚀 Training Models

### Model v1: First Training Run

```bash
cd scripts/training

# Train Model 1 - Vehicle Detection
python train_vehicle_detector.py \
  --data ../../datasets/processed/busay_vehicle_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name vehicle_detection

# Train Model 2 - Crash Detection
python train_vehicle_detector.py \
  --data ../../datasets/processed/busay_accident_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name crash_detection
```

**Output:**
- `models/v1/vehicle_detection/weights/best.pt`
- `models/v1/crash_detection/weights/best.pt`

### Model v2: Improved Training

After collecting real Busay footage or adjusting hyperparameters:

```bash
# Fine-tune on real Busay data
python train_vehicle_detector.py \
  --data ../../datasets/processed/busay_vehicle_v2/data.yaml \
  --weights ../../models/v1/vehicle_detection/weights/best.pt \
  --model n \
  --batch 4 \
  --epochs 50 \
  --project ../../models/v2 \
  --name vehicle_detection_finetuned
```

## 📊 Model Comparison

| Version | Dataset | Epochs | mAP@0.5 | Training Time | Status |
|---------|---------|--------|---------|---------------|--------|
| v1 | Roboflow merged | 100 | TBD | ~6-8 hrs | Training |
| v2 | + Busay footage | 150 | TBD | ~8-10 hrs | Planned |

## 🎯 Using Trained Models

### Test a Model

```bash
cd scripts/training

# Test vehicle detection model
python train_vehicle_detector.py \
  --test \
  --model-path ../../models/v1/vehicle_detection/weights/best.pt \
  --source /path/to/busay_video.mp4
```

### Deploy to Production

```bash
# Create production symlinks
ln -sf ../v1/vehicle_detection/weights/best.pt models/production/vehicle_detector.pt
ln -sf ../v1/crash_detection/weights/best.pt models/production/crash_detector.pt

# Use in speed detection script
cd scripts/download
# Edit auto_download_coco.py to use:
# model = YOLO('../../models/production/vehicle_detector.pt')
```

## 📈 Training Results

Each training run creates these files:

```
models/v1/vehicle_detection/
├── weights/
│   ├── best.pt              # Best model (highest mAP)
│   └── last.pt              # Last epoch checkpoint
├── results.png              # Training/validation curves
├── confusion_matrix.png     # Class confusion matrix
├── F1_curve.png             # F1 score curve
├── PR_curve.png             # Precision-Recall curve
├── P_curve.png              # Precision curve
└── R_curve.png              # Recall curve
```

## 💾 Storage Requirements

- **Each trained model**: ~6-12 MB (YOLOv8n)
- **Full training outputs**: ~50-100 MB per model
- **v1 + v2 models**: ~200-400 MB total

## 🔄 Version Management

### Version Naming Convention

- **v1**: Initial training on Roboflow datasets
- **v2**: Fine-tuned on real Busay footage
- **v3**: Optimized hyperparameters
- **vX_experiment**: Experimental variants

### Rollback to Previous Version

```bash
# Switch production to v1
ln -sf ../v1/vehicle_detection/weights/best.pt models/production/vehicle_detector.pt

# Or switch to v2
ln -sf ../v2/vehicle_detection_finetuned/weights/best.pt models/production/vehicle_detector.pt
```

## 📝 Model Performance Log

Keep track of model performance:

```
Version | Date       | mAP@0.5 | Precision | Recall | Notes
--------|------------|---------|-----------|--------|------------------
v1      | 2026-01-28 | 0.85    | 0.87      | 0.82   | Initial Roboflow
v2      | 2026-02-05 | 0.91    | 0.89      | 0.88   | + Busay footage
```

## 🧹 Cleanup

To save space, keep only best models:

```bash
# Remove all but best.pt
find models/ -name "last.pt" -delete

# Remove old experimental versions
rm -rf models/v1_experiment*
```

## 🎓 For Your Thesis

Document these for your thesis:
- Training parameters (epochs, batch size, learning rate)
- Hardware used (RTX 3050, training time)
- Performance metrics (mAP, precision, recall)
- Model size and inference speed
- Version improvements (v1 → v2 accuracy gains)
