# Datasets Folder

This folder contains all datasets for training your Busay vehicle detection system.

## 📁 Folder Structure

```
datasets/
├── downloaded/          # Raw datasets from Roboflow
│   ├── Traffic-surveillance-system-1/
│   ├── Vehicle-Detection-Day-Night-1/
│   └── Accident-detection-1/
│
└── processed/          # Merged/processed datasets ready for training
    ├── busay_vehicle_detection/     # Model 1: Vehicle detection
    └── busay_accident_detection/    # Model 2: Crash detection
```

## 📥 How to Use

### Step 1: Download Datasets from Roboflow

Place your downloaded Roboflow datasets in the `downloaded/` folder:

```bash
# After downloading from Roboflow, move them here:
mv ~/Downloads/Traffic-surveillance-system-1 datasets/downloaded/
mv ~/Downloads/Vehicle-Detection-Day-Night-1 datasets/downloaded/
mv ~/Downloads/Accident-detection-1 datasets/downloaded/
```

### Step 2: Process/Merge Datasets

```bash
cd scripts/training

# Run the automatic merger
python run_merge_busay.py
```

This will create processed datasets in `datasets/processed/`:
- `busay_vehicle_detection/` - For Model 1 (vehicle tracking & speed)
- `busay_accident_detection/` - For Model 2 (crash detection)

### Step 3: Train Models

```bash
# Train Model 1 (Vehicle Detection)
python train_vehicle_detector.py \
  --data ../../datasets/processed/busay_vehicle_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name vehicle_detection

# Train Model 2 (Crash Detection)
python train_vehicle_detector.py \
  --data ../../datasets/processed/busay_accident_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name crash_detection
```

## 🗂️ Dataset Versions

If you want to experiment with different dataset combinations:

```
processed/
├── busay_vehicle_v1/          # First version
├── busay_vehicle_v2/          # With additional data
└── busay_vehicle_night_only/  # Night-only variant
```

## 💾 Storage Requirements

- **Downloaded datasets**: ~2-5 GB per dataset
- **Processed datasets**: ~3-8 GB total
- **Total space needed**: ~15-20 GB

## 🧹 Cleaning Up

To free up space after merging:

```bash
# Keep only processed datasets, remove downloaded originals
rm -rf datasets/downloaded/*

# Or keep originals as backup (recommended)
```

## 📝 Notes

- Always keep backups of processed datasets
- Each dataset includes train/valid/test splits
- YOLO format: `data.yaml` + images + labels folders
