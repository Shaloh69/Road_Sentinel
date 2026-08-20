# 🚀 START HERE — Training Guide for Busay Project

**Step-by-step guide from zero to a trained model.** (Rewritten to match the real `training/` scripts — the previous version of this file referenced `scripts/training/train_vehicle_detector.py`, which doesn't exist in this repo. See `documentation.md §15` for the full drift history.)

---

## ✅ What You Need

Three Roboflow datasets, matched by name:
1. **Traffic Surveillance System** — vehicle detection (bus, car, motorbike, truck)
2. **Vehicle Detection (Day & Night)** — day/night vehicles
3. **Accident Detection** — crash detection

---

## 📋 STEP-BY-STEP

### STEP 1: Place Your Downloaded Datasets

```bash
cd RoadSentinel
mv ~/Downloads/Traffic-surveillance-system-1 datasets/downloaded/
mv ~/Downloads/Vehicle-Detection-Day-Night-1 datasets/downloaded/
mv ~/Downloads/Accident-detection-1 datasets/downloaded/
```

### STEP 2: Set Up the Training Environment

```bash
cd training                     # not scripts/training — that folder doesn't exist here
python3 --version                # must be 3.9–3.12, NOT 3.13 (PyTorch/CUDA wheels lag behind)
python3 -m venv venv_training
source venv_training/bin/activate   # Windows: venv_training\Scripts\activate
```

### STEP 3: Install PyTorch with GPU Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### STEP 4: Install Other Dependencies

```bash
pip install -r requirements.txt
python -c "from ultralytics import YOLO; print('OK')"
```

### STEP 5: Merge Your Datasets

```bash
python run_merge_busay.py
```

Merges Traffic Surveillance + Day/Night into `datasets/processed/busay_vehicle_detection/`, and prepares `datasets/processed/busay_accident_detection/` from the Accident Detection set. Missing folders (e.g. a dataset with only `train/`, no `valid/`/`test/`) are handled automatically — YOLO auto-splits for validation.

### STEP 6: Train the Vehicle Model

```bash
python train.py --dataset vehicle --model-size n --epochs 100
```

**Real CLI, not the old flat `--data/--project/--name` flags:** `--dataset {vehicle,accident,both}`, `--model-size {n,s,m,l,x}` (YOLO26, not YOLOv8), `--epochs`, `--imgsz`, `--batch` (auto-detected from VRAM if omitted), `--resume`.

**Output:** `models/runs/vehicle/vehicle_yolo26n_<timestamp>/weights/{best,last}.pt` — not `models/v1/vehicle_detection/weights/best.pt`. There is no `models/v1`/`v2`/`production` layout in this repo.

### STEP 7: Train the Accident (Crash) Model — separate GPU job

```bash
python train.py --dataset accident --model-size n --epochs 100
```

The merged dataset is ready at `datasets/processed/busay_accident_detection/`; this has not been run yet on this checkout (`models/runs/` currently has a `vehicle/` folder only, no `accident/`). Training both sequentially (`--dataset both`) or in two terminals both work.

### STEP 8: Point the AI Service at Your Trained Model

```bash
cd ../server/ai-service
cp .env.example .env
# Edit .env:
#   TRAFFIC_MODEL_PATH=../../models/runs/vehicle/vehicle_yolo26n_<timestamp>/weights/best.pt
#   (relative paths resolve against server/ai-service/, not your shell's cwd)
python -m app.main
```

Check the startup log for `Traffic detector ready — custom_model=True` — `False` means it silently fell back to the stock COCO `yolov8n.pt` instead of your trained weight.

### STEP 9: Test It

```bash
cd ../../testing
python test_ai.py                       # quick smoke test against the running AI service
python test_video.py your_video.mp4      # or test against your own footage
```

To test the raw `.pt` file directly without the server running, use `training/validate.py --model <path> --source <video>` instead.

### STEP 10: Calibrate for Accurate Speed (Optional but Recommended)

Speed defaults to a flat pixels-per-meter estimate, which is systematically wrong depending on where in frame a vehicle is tracked. For perspective-corrected speed: open the web client → **Cameras** → select a camera → **Open Calibration Tool** → click the 4 corners of a known rectangle on the road → enter its real width/length in meters → save. No script to run — it's a UI feature backed by `server/ai-service/app/models/traffic_detector.py`'s homography support.

---

## ⚠️ Troubleshooting

**"CUDA not available"** — check Python version (3.9–3.12), reinstall PyTorch with the CUDA index URL from Step 3.

**"No datasets found"** — confirm each folder under `datasets/downloaded/` has a `data.yaml`.

**"Out of memory"** — reduce `--batch` (or let it auto-detect by omitting the flag), or train one model at a time instead of `--dataset both`.

---

## 🎓 For Your Thesis

Document: training time per model, final mAP@0.5 (`results.png`, `confusion_matrix.png` in the run's output folder), hardware used, model size, inference speed. If you calibrate a camera, note the calibration point placement — that's the main accuracy driver for homography speed, not the algorithm.

---

## 🆘 Need Help?

- **Ground truth:** `documentation.md`
- **Structure:** `PROJECT_STRUCTURE.md`
- **Training details:** `training/README.md`
- **Main README:** `README.md`
