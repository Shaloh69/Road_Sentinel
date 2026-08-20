# Road Sentinel - Vehicle Speed Detection Training Guide

## 🎯 Project Overview

This project trains YOLO26 models to detect and track vehicles (bicycles, cars, motorcycles, buses, trucks) for the Busay blind curve system, using merged Roboflow datasets specific to this project — **not** a generic COCO training walkthrough (an earlier version of this doc described a COCO-first workflow that doesn't match what `training/train.py` actually does; see `documentation.md §15`).

## 🔧 Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB+ |
| GPU | None (CPU) | 8GB+ VRAM (this project's own trained run used an RTX 3060 Ti) |
| Python | 3.9 | 3.11/3.12 (NOT 3.13 — PyTorch/CUDA wheels lag behind) |

## 📦 Installation

```bash
cd training
python3 -m venv venv_training
source venv_training/bin/activate   # Windows: venv_training\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "from ultralytics import YOLO; print('OK')"
```

## 📊 Dataset

Built from your own merged Roboflow datasets (`training/run_merge_busay.py`), not an auto-downloaded generic dataset:
- `datasets/processed/busay_vehicle_detection/` — car, motorcycle, bicycle, bus, truck
- `datasets/processed/busay_accident_detection/` — accident vs. no-accident

Both already exist on a working checkout once you've run the merge step; see `PROJECT_STRUCTURE.md`.

## 🚀 Training — `training/train.py`

There is no `quick_train.py` and no `train_vehicle_detector.py` in this repo — `train.py` is the only trainer, and its CLI is dataset-driven, not a flat `--model/--epochs` set:

```bash
python train.py --dataset {vehicle,accident,both} --model-size {n,s,m,l,x} [--epochs N] [--batch N] [--imgsz N] [--resume] [--pretrained PATH] [--export {onnx,torchscript,engine}]
```

| Argument | Default | Notes |
|----------|---------|-------|
| `--dataset` | *(required)* | `vehicle`, `accident`, or `both` (sequential) |
| `--model-size` | `s` | YOLO26 nano/small/medium/large/xlarge |
| `--epochs` | 100 | |
| `--batch` | auto | Auto-selected from detected GPU VRAM if omitted |
| `--imgsz` | 640 | |
| `--resume` | off | Resume from last checkpoint |

```bash
# Vehicle detection, nano, 100 epochs
python train.py --dataset vehicle --model-size n --epochs 100

# Accident/crash detection — dataset is ready, not yet run on this checkout
python train.py --dataset accident --model-size n --epochs 100

# Both, sequentially
python train.py --dataset both --model-size s --epochs 100
```

**Output:** `models/runs/<dataset>/<dataset>_yolo26<size>_<timestamp>/weights/{best,last,epochN}.pt` — not `models/v1/...` or `runs/vehicle_speed/...`.

## 💡 Using the Trained Model

The production path is the AI service, not a standalone script — point `server/ai-service/.env`'s `TRAFFIC_MODEL_PATH` at your `best.pt` (relative paths resolve against `server/ai-service/`) and run `python -m app.main`. See `README.md`'s "Running the full stack locally" section.

For a quick local sanity check without starting the server:

```bash
cd training
python validate.py --model ../models/runs/vehicle/vehicle_yolo26n_<timestamp>/weights/best.pt --source test_video.mp4
```

(`validate.py`'s name is a bit misleading — it runs `model.predict()`, not `model.val()`; it's a prediction/testing script, not a metrics-validation one.)

## ⏱️ Training Time Estimates (100 epochs, YOLO26n)

| Hardware | Time |
|----------|------|
| RTX 4090 | 4-6 hours |
| RTX 3080 | 6-10 hours |
| RTX 3060 Ti | 10-15 hours |
| CPU only | Days — use a cloud GPU instead |

## 🐛 Troubleshooting

**CUDA Out of Memory** — `python train.py --dataset vehicle --model-size n --batch 4` (or a smaller `--imgsz`).

**Import error on `ultralytics`** — `pip install ultralytics` (correct import is `from ultralytics import YOLO`).

**"Dataset not found"** — run `python run_merge_busay.py` first; check `datasets/processed/<name>/data.yaml` exists.

## 🎓 Best Practices for Your Thesis

Document: hardware used, hyperparameters, training time, final mAP50/mAP50-95/precision/recall (from the run's `results.png`/`results.csv`), model size, inference speed (FPS), and — if calibrated — the calibration point placement quality for the speed-estimation numbers, since that's the main accuracy driver for homography speed, not the algorithm choice.

## 📞 Support & Resources

- **YOLO/Ultralytics Docs:** https://docs.ultralytics.com
- **PyTorch:** https://pytorch.org/docs
- This repo: `documentation.md` (ground truth), `README.md`, `PROJECT_STRUCTURE.md`
