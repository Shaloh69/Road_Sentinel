# 🚀 START HERE - Complete Training Guide for Busay Project

**Step-by-step guide from zero to trained models**

---

## ✅ What You Have

You downloaded **3 Roboflow datasets**:
1. **Traffic Surveillance System** - Vehicle detection (bus, car, motorbike, truck)
2. **Vehicle Detection (Day & Night)** - Classes 0-7 (day/night vehicles)
3. **Accident Detection** - Crash detection (7 classes)

---

## 🎯 What the Merge Script Does

### ✅ Merges (for Model 1):
- Traffic Surveillance + Day/Night → **One combined vehicle detection dataset**
- Remaps class names to standard format (car, motorcycle, bicycle, bus, truck)
- Handles day/night images (merges them into same classes)

### ✅ Keeps Separate (for Model 2):
- Accident Detection → **Separate crash detection dataset**
- Simplifies to binary (no_accident vs accident)

### ✅ Handles Missing Folders:
- If Day/Night only has `train/` folder (no valid/test), that's OK!
- Script automatically skips missing folders
- YOLOv8 will auto-split training data for validation

---

## 📋 STEP-BY-STEP TRAINING GUIDE

### STEP 1: Place Your Downloaded Datasets

```bash
# Navigate to project
cd /home/user/Road_Sentinel

# Move your downloaded datasets to the organized location
mv ~/Downloads/Traffic-surveillance-system-1 datasets/downloaded/
mv ~/Downloads/Vehicle-Detection-Day-Night-1 datasets/downloaded/
mv ~/Downloads/Accident-detection-1 datasets/downloaded/

# Verify they're there
ls -la datasets/downloaded/
```

**You should see:**
```
datasets/downloaded/
├── Traffic-surveillance-system-1/
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── Vehicle-Detection-Day-Night-1/
│   ├── data.yaml
│   └── train/          # Only train folder - that's OK!
└── Accident-detection-1/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

---

### STEP 2: Setup Python Environment

```bash
# Navigate to training scripts
cd scripts/training

# Check your Python version (MUST be 3.8-3.12, NOT 3.13!)
python3 --version

# If you have Python 3.13, install Python 3.11 or 3.12 first!
# Then use python3.11 or python3.12 instead of python3

# Create virtual environment
python3 -m venv venv_training

# Activate it
source venv_training/bin/activate  # Linux/Mac
# venv_training\Scripts\activate   # Windows (if you're on Windows)
```

**You should see `(venv_training)` in your terminal prompt**

---

### STEP 3: Install PyTorch with GPU Support

```bash
# CRITICAL: Install PyTorch with CUDA FIRST (before requirements.txt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

**If you see `CUDA Available: False`**, you have a problem:
- Check Python version (must be 3.8-3.12, NOT 3.13)
- Reinstall PyTorch with correct CUDA index

---

### STEP 4: Install Other Dependencies

```bash
# Install ultralytics and other packages
pip install -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('✅ Ultralytics installed successfully')"
```

---

### STEP 5: Merge Your Datasets

```bash
# Run the automatic merger
python run_merge_busay.py
```

**What happens:**
1. Script searches for your 3 datasets in `datasets/downloaded/`
2. Shows you a merge plan
3. Asks for confirmation (type `y`)
4. Merges Traffic Surveillance + Day/Night → `datasets/processed/busay_vehicle_detection/`
5. Prepares Accident Detection → `datasets/processed/busay_accident_detection/`

**Expected output:**
```
🔍 Searching for datasets...

✅ Found Traffic Surveillance: Traffic-surveillance-system-1
✅ Found Day/Night Vehicle: Vehicle-Detection-Day-Night-1
✅ Found Accident Detection: Accident-detection-1

📋 MERGE PLAN:
✅ MODEL 1: Vehicle Detection (Merging 2 datasets)
   • Traffic Surveillance: Traffic-surveillance-system-1
   • Day/Night Vehicles: Vehicle-Detection-Day-Night-1
   → Output: datasets/processed/busay_vehicle_detection/

✅ MODEL 2: Crash Detection
   • Accident Detection: Accident-detection-1
   → Output: datasets/processed/busay_accident_detection/

Proceed with merge? (y/n): y

🔄 STARTING MERGE...

📦 Merging Vehicle Detection datasets...
   ✅ train: 5000 images
   ⚠️  No valid found, skipping...  # Day/Night only has train - OK!
   ⚠️  No test found, skipping...

✅ VEHICLE DETECTION MERGE COMPLETE!
📁 Output: datasets/processed/busay_vehicle_detection/
📊 Total images: 8000
   • Train: 8000
   • Valid: 0
   • Test: 0
🏷️  Classes: car, motorcycle, bicycle, bus, truck
   ⚠️  No validation set found - YOLO will auto-split from training data

🚨 Preparing Accident Detection dataset...
   ✅ train: 2000 images
   ✅ valid: 500 images
   ✅ test: 300 images

✅ ACCIDENT DETECTION DATASET READY!
```

**After merge, you'll have:**
```
datasets/processed/
├── busay_vehicle_detection/
│   ├── data.yaml
│   └── train/
│       ├── images/  (traffic_*.jpg + daynight_*.jpg)
│       └── labels/  (remapped to 0-4)
│
└── busay_accident_detection/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

---

### STEP 6: Train Model 1 (Vehicle Detection)

**Open Terminal 1:**

```bash
# Make sure you're in scripts/training with venv activated
cd /home/user/Road_Sentinel/scripts/training
source venv_training/bin/activate

# Train Model 1
python train_vehicle_detector.py \
  --data ../../datasets/processed/busay_vehicle_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name vehicle_detection
```

**What happens:**
- Downloads YOLOv8n pretrained weights (~6 MB)
- Starts training for 100 epochs
- Auto-splits train data for validation (since no valid folder)
- Saves checkpoints every epoch
- Shows live training progress

**Training time:** ~6-8 hours on RTX 3050

**Output location:**
```
models/v1/vehicle_detection/
├── weights/
│   ├── best.pt          # ⭐ BEST MODEL (use this!)
│   └── last.pt          # Last epoch
├── results.png          # Training curves
├── confusion_matrix.png
└── ...
```

---

### STEP 7: Train Model 2 (Crash Detection) - OPTIONAL: Run in Parallel!

**Open Terminal 2 (while Model 1 is training):**

```bash
# New terminal - navigate and activate venv
cd /home/user/Road_Sentinel/scripts/training
source venv_training/bin/activate

# Train Model 2
python train_vehicle_detector.py \
  --data ../../datasets/processed/busay_accident_detection/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --project ../../models/v1 \
  --name crash_detection
```

**Training time:** ~3-4 hours on RTX 3050

**💡 TIP:** You can train both models simultaneously! Your RTX 3050 can handle it.

**Output location:**
```
models/v1/crash_detection/
├── weights/
│   ├── best.pt          # ⭐ BEST MODEL (use this!)
│   └── last.pt
└── ...
```

---

### STEP 8: Monitor Training Progress

**While training, you'll see:**
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100      2.5G      1.234      0.567      1.123         45        640
  2/100      2.5G      1.156      0.523      1.089         42        640
  ...
 50/100      2.5G      0.234      0.123      0.345         38        640  # Getting better!
  ...
100/100      2.5G      0.156      0.089      0.234         35        640  # Final epoch

✅ Training complete!
Results saved to: models/v1/vehicle_detection/
Best model: models/v1/vehicle_detection/weights/best.pt
```

**What to look for:**
- ✅ Losses should decrease over time
- ✅ mAP (mean Average Precision) should increase
- ✅ GPU memory stable around 2-3 GB
- ⚠️ If GPU memory hits 4GB → reduce batch size to 2

---

### STEP 9: Verify Trained Models

```bash
# Check Model 1
ls -lh ../../models/v1/vehicle_detection/weights/best.pt

# Check Model 2
ls -lh ../../models/v1/crash_detection/weights/best.pt

# View training results
cd ../../models/v1/vehicle_detection
ls -la  # You should see results.png, confusion_matrix.png, etc.
```

---

### STEP 10: Create Production Symlinks

```bash
cd /home/user/Road_Sentinel/models/production

# Create symlinks to v1 models
ln -sf ../v1/vehicle_detection/weights/best.pt vehicle_detector.pt
ln -sf ../v1/crash_detection/weights/best.pt crash_detector.pt

# Verify
ls -la
# You should see:
# vehicle_detector.pt -> ../v1/vehicle_detection/weights/best.pt
# crash_detector.pt -> ../v1/crash_detection/weights/best.pt
```

---

## ✅ YOU'RE DONE! 🎉

You now have:
- ✅ **Model 1:** Vehicle detection (car, motorcycle, bicycle, bus, truck)
- ✅ **Model 2:** Crash detection (accident vs no_accident)
- ✅ **Production models:** Ready to use in `models/production/`

---

## 🧪 Testing Your Models

### Test on a Video:

```bash
cd /home/user/Road_Sentinel/scripts/training

# Test Model 1 (Vehicle Detection)
python train_vehicle_detector.py \
  --test \
  --model-path ../../models/production/vehicle_detector.pt \
  --source /path/to/your/test_video.mp4

# Test Model 2 (Crash Detection)
python train_vehicle_detector.py \
  --test \
  --model-path ../../models/production/crash_detector.pt \
  --source /path/to/your/test_video.mp4
```

### Use in Speed Detection:

```bash
cd ../download

# Edit auto_download_coco.py to use your trained model:
# model = YOLO('../../models/production/vehicle_detector.pt')

python auto_download_coco.py
```

---

## 📊 Summary of What Happens

| Step | Action | Time | Output |
|------|--------|------|--------|
| 1 | Move datasets | 1 min | `datasets/downloaded/` populated |
| 2 | Setup venv | 2 min | Virtual environment active |
| 3 | Install PyTorch | 5 min | GPU enabled |
| 4 | Install deps | 2 min | All packages ready |
| 5 | Merge datasets | 5-10 min | `datasets/processed/` created |
| 6 | Train Model 1 | 6-8 hrs | `models/v1/vehicle_detection/` |
| 7 | Train Model 2 | 3-4 hrs | `models/v1/crash_detection/` |
| 8 | Create symlinks | 1 min | `models/production/` ready |
| **TOTAL** | **~10-12 hours** | **2 production models!** |

---

## ⚠️ Troubleshooting

### Problem: "CUDA not available"
**Solution:**
- Check Python version: `python --version` (must be 3.8-3.12)
- Reinstall PyTorch with CUDA:
  ```bash
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### Problem: "No datasets found"
**Solution:**
- Check datasets are in `datasets/downloaded/`
- Each dataset must have `data.yaml` file
- Verify with: `ls -la datasets/downloaded/*/data.yaml`

### Problem: "Out of memory"
**Solution:**
- Reduce batch size: `--batch 2` (instead of 4)
- Close other GPU applications
- Train one model at a time (not in parallel)

### Problem: Day/Night dataset only has train folder
**Solution:**
- ✅ **This is OK!** The merge script handles it automatically
- YOLOv8 will auto-split training data for validation
- You'll see: "⚠️ No validation set found - YOLO will auto-split"

### Problem: Training is very slow
**Solution:**
- Verify GPU is being used: Check training output for "GPU_mem" column
- If no GPU detected, reinstall PyTorch with CUDA
- Normal speed: ~2-3 minutes per epoch on RTX 3050

---

## 🎓 For Your Thesis

**Document these metrics:**
- Training time per model
- Final mAP@0.5 (from results)
- Precision and Recall per class
- Model size (best.pt file size)
- Inference speed (FPS)
- Hardware used (RTX 3050 4GB)

**Find metrics in:**
- `models/v1/vehicle_detection/results.png` - Training curves
- `models/v1/vehicle_detection/confusion_matrix.png` - Per-class accuracy
- Terminal output at end of training - Final mAP scores

---

## 📖 Next Steps After Training

1. **Test on Busay videos** - Validate with real blind curve footage
2. **Fine-tune v2** - Add Busay-specific images, train v2 models
3. **Deploy to Raspberry Pi** - Optimize for edge device
4. **Camera calibration** - Run `angled_camera_calibration.py`
5. **Integrate LED display** - Connect to warning system

---

## 🆘 Need Help?

- **Structure guide:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Dataset strategy:** [scripts/training/DATASET_STRATEGY_GUIDE.md](scripts/training/DATASET_STRATEGY_GUIDE.md)
- **Dual models:** [scripts/training/DUAL_MODEL_TRAINING_GUIDE.md](scripts/training/DUAL_MODEL_TRAINING_GUIDE.md)
- **Main README:** [README.md](README.md)

---

**Good luck with your Busay blind curve system! 🎓🚗**
