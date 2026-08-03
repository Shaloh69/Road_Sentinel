# YOLO26 Training Pipeline

Scripts for training vehicle-detection and crash/incident-detection models for Road Sentinel from your own merged Busay datasets. (This replaces an earlier version of this file that documented `train_vehicle_detector.py` and `quick_train.py` — neither exists in this repo. The real trainer is `train.py`. See `documentation.md §15` for the full drift history.)

## Setup

```bash
cd training
python3 -m venv venv_training
source venv_training/bin/activate   # Windows: venv_training\Scripts\activate
```

⚠️ **Python 3.9–3.12 required (NOT 3.13+)** — PyTorch/CUDA wheels lag behind.

```bash
# GPU (RTX 30/40 series — recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch torchvision torchaudio

pip install -r requirements.txt
python -c "from ultralytics import YOLO; import torch; print('Setup OK')"
```

## Dataset

Built from your own Roboflow exports, merged via `run_merge_busay.py` — not an auto-downloaded generic dataset. Place raw exports in `../datasets/downloaded/`, then:

```bash
python run_merge_busay.py
```

This produces `../datasets/processed/busay_vehicle_detection/` (merged Traffic Surveillance + Day/Night) and `../datasets/processed/busay_accident_detection/` (Accident Detection, prepared but not merged with anything else).

See `DATASET_STRATEGY_GUIDE.md` for merge-vs-single-dataset guidance and `YOLO_NATIVE_DATASETS.md` for the Roboflow Universe path.

## Training — `train.py`

```bash
python train.py --dataset {vehicle,accident,both} --model-size {n,s,m,l,x} [--epochs N] [--batch N] [--imgsz N] [--resume]
```

| Argument | Default | Notes |
|----------|---------|-------|
| `--dataset` | *(required)* | `vehicle`, `accident`, or `both` (trains sequentially) |
| `--model-size` | `s` | YOLO26 nano → xlarge |
| `--epochs` | 100 | |
| `--batch` | auto | Auto-picked from detected GPU VRAM if omitted |
| `--resume` | off | Resume from last checkpoint |
| `--export` | — | `onnx` / `torchscript` / `engine`, after training |

```bash
# Vehicle detection
python train.py --dataset vehicle --model-size n --epochs 100

# Accident/crash detection (dataset ready; not yet run on this checkout)
python train.py --dataset accident --model-size n --epochs 100

# Both, one after another
python train.py --dataset both --model-size s --epochs 100
```

**Output:** `../models/runs/<dataset>/<dataset>_yolo26<size>_<timestamp>/weights/{best,last,epochN}.pt`

## Testing a Trained Model

`validate.py` (name is a bit misleading — it runs `model.predict()` for a visual/console check, not `model.val()` for metrics):

```bash
python validate.py --model ../models/runs/vehicle/vehicle_yolo26n_<timestamp>/weights/best.pt --source your_video.mp4
```

To exercise the deployed AI service instead (HTTP, not a direct model load), use `../testing/test_video.py` / `test_images.py` / `test_ai.py`.

## Deploying a Trained Model

There's no `models/production/` symlink step. Point the AI service at your weight file directly:

```bash
cd ../server/ai-service
# .env: TRAFFIC_MODEL_PATH=../../models/runs/vehicle/vehicle_yolo26n_<timestamp>/weights/best.pt
python -m app.main
```

Check the startup log for `custom_model=True` — `False` means it fell back to stock `yolov8n.pt` (usually a bad/missing path).

## Files

| File | Purpose |
|------|---------|
| `train.py` | Trainer (YOLO26, dataset/model-size driven CLI above) |
| `validate.py` | Predict/test a local `.pt` against a video/image/folder |
| `run_merge_busay.py` | Interactive dataset-merge wrapper — run first |
| `merge_busay_datasets.py` | Merge logic (`run_merge_busay.py` calls into this) |
| `analyze_datasets.py` | Compare available datasets (image counts, classes) |
| `convert_aicity_track1_to_yolo.py` / `convert_aicity_track4_to_yolo.py` | AI City Challenge → YOLO format converters |
| `download_roboflow_datasets.py` | Roboflow SDK download helper |
| `download_test_video.py` | Pulls a few stock traffic videos for manual testing |

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| YOLO26n | Nano | Fastest | Good — real-time |
| YOLO26s | Small | Fast | Better |
| YOLO26m | Medium | Moderate | Best for 8GB VRAM |
| YOLO26l/x | Large/XL | Slow | Not recommended under 8GB VRAM |

## Training Time Estimates (100 epochs, nano)

| Hardware | Time |
|----------|------|
| RTX 4090 | 4-6 hours |
| RTX 3080 | 6-10 hours |
| RTX 3060 Ti | 10-15 hours |
| CPU only | Days — use a cloud GPU instead |

## Troubleshooting

**CUDA Out of Memory**
```bash
python train.py --dataset vehicle --model-size n --batch 4
```

**Dataset not found** — run `python run_merge_busay.py` first; confirm `../datasets/processed/<name>/data.yaml` exists.

**Import error on `ultralytics`** — `pip install ultralytics>=8.3.0` (YOLO26 needs a recent version).

## Next Steps

1. Train a model with this folder's environment
2. Point `server/ai-service/.env`'s `TRAFFIC_MODEL_PATH`/`INCIDENT_MODEL_PATH` at the resulting `best.pt`
3. Restart the AI service and confirm `custom_model=True` in its startup log
4. Calibrate cameras for perspective-corrected speed via the web client's Cameras → Calibration Tool
