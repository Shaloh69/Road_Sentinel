# Datasets Folder

Training data for Road Sentinel's two models (vehicle detection, crash/incident detection).

## Structure

```
datasets/
├── downloaded/          # Raw Roboflow exports — untracked, gitignored
└── processed/           # Merged/converted datasets ready for training
    ├── busay_vehicle_detection/     # Model 1: vehicle detection & tracking
    └── busay_accident_detection/    # Model 2: crash/incident detection
```

## How to use

### 1. Download datasets from Roboflow

Place downloaded Roboflow exports (YOLO format) under `datasets/downloaded/`.

### 2. Merge into the two Busay-specific training sets

```bash
cd training
python run_merge_busay.py
```

This produces `datasets/processed/busay_vehicle_detection/` and `datasets/processed/busay_accident_detection/`, each with `data.yaml` + `train`/`valid`/`test` splits — both already exist on a working checkout.

### 3. Train

```bash
python train.py --dataset vehicle --model-size n --epochs 100
python train.py --dataset accident --model-size n --epochs 100
```

There is no `scripts/` folder and no `train_vehicle_detector.py` — `train.py` (in `training/`) is the real, only trainer, and it takes `--dataset {vehicle,accident,both}` rather than a raw path to a merged dataset.

## Storage

`datasets/downloaded/` and `datasets/processed/` are both gitignored — datasets are large (several GB) and machine-specific, not something to commit. Keep the raw `downloaded/` exports around if you might need to re-merge with different settings later; otherwise they're safe to delete once `processed/` exists.
