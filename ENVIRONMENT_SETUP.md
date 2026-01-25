# Environment Setup Guide

This project uses **separate virtual environments** for each script folder to maintain isolation and avoid dependency conflicts.

## 📁 Project Structure

```
Road_Sentinel/
├── scripts/
│   ├── download/           # Speed detection system
│   │   ├── venv/          # Virtual environment for this folder
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── auto_download_coco.py
│   │
│   ├── extract_frames/     # Frame extraction utility
│   │   ├── venv_frames/   # Virtual environment for this folder
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── extract_frames.py
│   │
│   └── training/           # Model training pipeline
│       ├── venv_training/ # Virtual environment for this folder
│       ├── requirements.txt
│       ├── README.md
│       ├── train_vehicle_detector.py
│       └── quick_train.py
│
├── requirements.txt        # Global dependencies (optional)
└── ENVIRONMENT_SETUP.md   # This file
```

## 🎯 Why Separate Environments?

✅ **Dependency Isolation** - Each script has only what it needs
✅ **Avoid Conflicts** - Different versions won't interfere
✅ **Faster Setup** - Install only required packages per task
✅ **Cleaner Development** - Easy to troubleshoot issues
✅ **Production Ready** - Deploy only what you use

## 🚀 Setup Instructions

### Option 1: Setup All Environments (Recommended)

```bash
# Run this script from project root
cd /home/user/Road_Sentinel

# Setup all three environments
./setup_all_environments.sh
```

### Option 2: Setup Individual Environments

#### 1️⃣ Frame Extraction Environment

```bash
cd scripts/extract_frames

# Create virtual environment
python3 -m venv venv_frames

# Activate
source venv_frames/bin/activate  # Linux/Mac
# venv_frames\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import cv2; print('✅ Frames environment ready!')"

# Deactivate when done
deactivate
```

**Dependencies:** opencv-python, numpy (lightweight, ~150MB)

#### 2️⃣ Speed Detection Environment

```bash
cd scripts/download

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "from ultralytics import YOLO; print('✅ Detection environment ready!')"

# Deactivate when done
deactivate
```

**Dependencies:** ultralytics, torch, opencv-python, numpy (~2-3GB with CUDA)

#### 3️⃣ Training Environment

```bash
cd scripts/training

# Create virtual environment
python3 -m venv venv_training

# Activate
source venv_training/bin/activate  # Linux/Mac
# venv_training\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "from ultralytics import YOLO; import torch; print('✅ Training environment ready!')"

# Deactivate when done
deactivate
```

**Dependencies:** ultralytics, torch, opencv-python, pandas, matplotlib (~3GB with CUDA)

## 📝 Quick Reference

### Activation Commands

| Environment | Activate (Linux/Mac) | Activate (Windows) |
|-------------|---------------------|-------------------|
| **Frame Extraction** | `cd scripts/extract_frames && source venv_frames/bin/activate` | `cd scripts\extract_frames && venv_frames\Scripts\activate` |
| **Speed Detection** | `cd scripts/download && source venv/bin/activate` | `cd scripts\download && venv\Scripts\activate` |
| **Training** | `cd scripts/training && source venv_training/bin/activate` | `cd scripts\training && venv_training\Scripts\activate` |

### Deactivation

```bash
# From any environment
deactivate
```

## 🔄 Typical Workflow

### Workflow 1: Extract Frames → Train Model → Run Detection

```bash
# Step 1: Extract frames from videos
cd scripts/extract_frames
source venv_frames/bin/activate
python extract_frames.py ~/Videos/traffic/ -o frames_output -f 30
deactivate

# Step 2: Train model (if needed)
cd ../training
source venv_training/bin/activate
python quick_train.py
deactivate

# Step 3: Run speed detection
cd ../download
source venv/bin/activate
python auto_download_coco.py
deactivate
```

### Workflow 2: Just Run Detection (Pre-trained Model)

```bash
# Use pre-trained model directly
cd scripts/download
source venv/bin/activate
python auto_download_coco.py
deactivate
```

## 📦 Dependencies Overview

### Frame Extraction (`extract_frames/`)
```
opencv-python>=4.8.0
numpy>=1.24.0
```
**Total Size:** ~150MB

### Speed Detection (`download/`)
```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
```
**Total Size:** ~2-3GB (with CUDA)

### Training (`training/`)
```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
tqdm>=4.65.0
PyYAML>=6.0
```
**Total Size:** ~3GB (with CUDA)

## 🛠️ Management Commands

### Check Active Environment
```bash
which python
# Should show path to venv/bin/python
```

### List Installed Packages
```bash
# In activated environment
pip list
```

### Update Dependencies
```bash
# In activated environment
pip install -r requirements.txt --upgrade
```

### Remove Environment
```bash
# Deactivate first
deactivate

# Remove folder
rm -rf venv_frames  # or venv, venv_training
```

### Recreate Environment
```bash
# Remove old environment
rm -rf venv_frames

# Create new one
python3 -m venv venv_frames
source venv_frames/bin/activate
pip install -r requirements.txt
```

## ⚠️ Common Issues

### Wrong Environment Active

**Problem:** ImportError even though package is listed in requirements.txt

**Solution:**
```bash
# Check which environment is active
which python

# Deactivate and activate correct one
deactivate
cd scripts/training  # or download, extract_frames
source venv_training/bin/activate  # use correct venv name
```

### Package Not Found

**Problem:** "No module named 'xyz'"

**Solution:**
```bash
# Make sure environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Multiple Environments Conflict

**Problem:** Packages from one environment affecting another

**Solution:**
```bash
# Always deactivate before switching
deactivate

# Then activate the one you need
cd scripts/training
source venv_training/bin/activate
```

## 📊 Storage Requirements

| Environment | Disk Space |
|-------------|-----------|
| Frame Extraction | ~150MB |
| Speed Detection | ~2-3GB |
| Training | ~3GB |
| **Total** | **~5-6GB** |

Plus:
- COCO dataset: ~20GB (only for training)
- Output files: Variable

## 🎓 Best Practices

1. **Always activate before use**
   ```bash
   source venv/bin/activate
   ```

2. **Always deactivate when done**
   ```bash
   deactivate
   ```

3. **Don't mix environments**
   - Extract frames → use `venv_frames`
   - Train models → use `venv_training`
   - Run detection → use `venv`

4. **Keep requirements.txt updated**
   ```bash
   # In activated environment
   pip freeze > requirements.txt
   ```

5. **Add venv folders to .gitignore**
   ```bash
   # Already done in this project
   venv/
   venv_*/
   ```

## 🔍 Verification

### Verify All Environments

```bash
# Frame extraction
cd scripts/extract_frames
source venv_frames/bin/activate
python -c "import cv2; print('✅ Frames OK')"
deactivate

# Speed detection
cd ../download
source venv/bin/activate
python -c "from ultralytics import YOLO; print('✅ Detection OK')"
deactivate

# Training
cd ../training
source venv_training/bin/activate
python -c "from ultralytics import YOLO; import pandas; print('✅ Training OK')"
deactivate
```

## 📚 Additional Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [pip documentation](https://pip.pypa.io/)
- [Virtual environments guide](https://realpython.com/python-virtual-environments-a-primer/)

## ✅ Summary

- ✅ **3 separate environments** for different tasks
- ✅ **Isolated dependencies** prevent conflicts
- ✅ **Each folder** has its own README and requirements.txt
- ✅ **Simple activation** with source command
- ✅ **Easy management** with standard pip commands

---

**Remember:** Always activate the correct environment before running scripts!
