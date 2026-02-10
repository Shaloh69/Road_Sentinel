# Dataset Strategy Guide: Merge vs Train Individually

## 🎯 Quick Decision Guide

### ✅ Train on ONE Dataset (RECOMMENDED for Busay)

**Choose this if:**
- You have one large dataset (5,000+ images)
- Dataset has angled/overhead views matching your camera setup
- You want to start training quickly
- **This is the FASTEST path to production**

**Steps:**
1. Pick the dataset with MOST images and BEST camera angle match
2. Train directly on that dataset
3. Deploy and test in ~4-5 hours

---

### 🔄 Merge Multiple Datasets (Advanced)

**Choose this if:**
- You have multiple small datasets (1,000-3,000 images each)
- All datasets have similar vehicle classes
- You want maximum training data
- You have time for extra setup (30-60 min)

**Steps:**
1. Analyze datasets to check class compatibility
2. Merge with class remapping
3. Train on combined dataset

---

## 📊 For Your Situation

You mentioned you downloaded multiple datasets but skipped "Vehicle Detection - Overhead View" because "traffic surveillance has an angled view already."

### Recommended Approach:

**Use the "traffic surveillance angled view" dataset ONLY**

**Why?**
- ✅ Matches your Busay camera setup (angled/overhead)
- ✅ One dataset = simpler, faster workflow
- ✅ 5,000+ images is enough for excellent accuracy
- ✅ Start training immediately (no merging needed)
- ✅ Production-ready in 4-5 hours

**When to merge:**
- Only if your main dataset has <3,000 images
- Or if you want to add specific scenarios (night vision, different weather, etc.)

---

## 🔍 Step-by-Step: Analyze Your Datasets

### Step 1: Check what you downloaded

```bash
cd /home/user/Road_Sentinel/scripts/training
python analyze_datasets.py
```

This will show:
- How many images in each dataset
- What classes each dataset has
- Which dataset is best for single-model training
- Whether merging makes sense

### Step 2: Decision Time

**Scenario A: One dataset has 5,000+ images**
```bash
✅ BEST CHOICE: Use that dataset alone
```

**Scenario B: Multiple datasets with 2,000-3,000 images each**
```bash
🔄 CONSIDER: Merging to get 6,000-9,000 total images
```

**Scenario C: Datasets have different classes (car vs vehicle)**
```bash
🔄 CAN MERGE: Use merge_datasets.py with class remapping
```

---

## 📋 How to Use: Single Dataset (RECOMMENDED)

### Example: You downloaded "Traffic Surveillance Angled View"

```bash
# 1. Navigate to dataset location
cd /path/to/traffic-surveillance-angled-view

# 2. Check data.yaml exists
ls data.yaml

# 3. Train immediately!
cd /home/user/Road_Sentinel/scripts/training
source venv_training/bin/activate

python train_vehicle_detector.py \
  --data /path/to/traffic-surveillance-angled-view/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --name busay_traffic_v1
```

**Training time:** 3-4 hours on RTX 3050
**Result:** Production-ready model for Busay!

---

## 🔄 How to Use: Merge Multiple Datasets

### Example: You have 3 datasets to merge

```bash
cd /home/user/Road_Sentinel/scripts/training
source venv_training/bin/activate
python
```

```python
from merge_datasets import merge_datasets

# List all datasets you want to merge
datasets_to_merge = [
    '/path/to/traffic-surveillance-1',
    '/path/to/night-vision-vehicles',
    '/path/to/overhead-traffic',
]

# Merge them into one dataset
merge_datasets(
    dataset_paths=datasets_to_merge,
    output_path='./busay_merged_dataset',
    dataset_name='Busay Complete Traffic Dataset'
)
```

**Output:** Creates `busay_merged_dataset` folder with:
- All images combined
- Class names standardized (car, motorcycle, bicycle, bus, truck)
- Ready-to-use data.yaml

**Then train:**

```bash
python train_vehicle_detector.py \
  --data ./busay_merged_dataset/data.yaml \
  --model n \
  --batch 4 \
  --epochs 100 \
  --name busay_merged_v1
```

**Training time:** 6-8 hours (more images = longer training)

---

## 🚦 Class Name Compatibility

### Common Class Name Variations

Different datasets may use different names for the same vehicles:

| Standard Name | Alternative Names | Maps To |
|---------------|-------------------|---------|
| car | vehicle, automobile | car (class 0) |
| motorcycle | motorbike, scooter, moped | motorcycle (class 1) |
| bicycle | bike, cycle | bicycle (class 2) |
| bus | coach, transit | bus (class 3) |
| truck | van, lorry, pickup | truck (class 4) |

**The `merge_datasets.py` script automatically handles these mappings!**

### Example Class Remapping

**Dataset 1 classes:** car, bike, truck
**Dataset 2 classes:** vehicle, motorcycle, bus
**Dataset 3 classes:** automobile, motorbike, bicycle

**After merging:** All mapped to standard: car, motorcycle, bicycle, bus, truck

---

## 📊 Training Time Comparison

| Strategy | Setup Time | Training Time | Total Time |
|----------|------------|---------------|------------|
| Single dataset (5,000 img) | 0 min | 3-4 hours | **~4 hours** ✅ |
| Merged datasets (10,000 img) | 30 min | 6-8 hours | **~8 hours** |
| Merged datasets (15,000 img) | 30 min | 9-12 hours | **~12 hours** |

**Recommendation:** Start with single best dataset, deploy quickly, then fine-tune later if needed.

---

## 🎯 For Busay Blind Curve Project

### Recommended Workflow:

```bash
# Phase 1: Quick Deployment (Day 1)
# ----------------------------------
# Use single best angled/overhead dataset
# Train 100 epochs (~4 hours)
# Deploy and test at Busay
# ✅ SYSTEM OPERATIONAL!

# Phase 2: Optimization (Optional - Later)
# ----------------------------------------
# Collect real Busay footage
# Fine-tune model on actual blind curve videos
# Improve accuracy for specific conditions
# 🚀 OPTIMIZED FOR BUSAY!
```

---

## 💡 My Recommendation for You

Based on your message:

> "I downloaded all the datasets. I skipped Vehicle Detection - Overhead View because the traffic surveillance has an angled view already."

**✅ Perfect! Use the traffic surveillance angled view dataset ONLY**

**Steps:**

1. **Locate your dataset:**
   ```bash
   # It's probably in your Downloads folder or where Roboflow saved it
   # Look for a folder with 'traffic-surveillance' or similar name
   ```

2. **Check it has good images:**
   ```bash
   cd /path/to/traffic-surveillance-dataset
   ls train/images | wc -l  # Should show 3,000-10,000+
   ```

3. **Start training immediately:**
   ```bash
   cd /home/user/Road_Sentinel/scripts/training
   source venv_training/bin/activate

   python train_vehicle_detector.py \
     --data /path/to/your/traffic-surveillance-dataset/data.yaml \
     --model n \
     --batch 4 \
     --epochs 100 \
     --name busay_v1
   ```

4. **Wait 3-4 hours**, then you have a working model!

---

## ❓ FAQ

**Q: Will one dataset be enough?**
A: Yes! 5,000+ images is plenty for excellent vehicle detection.

**Q: Should I use all datasets I downloaded?**
A: No, pick the BEST one (most images, best camera angle). Quality > Quantity.

**Q: What if datasets have different class names?**
A: The merge script handles this automatically with class remapping.

**Q: Can I add more data later?**
A: Yes! Train on one dataset now, then fine-tune with additional data later.

**Q: Which dataset is best for Busay overhead cameras?**
A: "Traffic Surveillance" with angled/overhead views. Skip dashcam-view datasets.

---

## 🚀 Next Steps

1. Run `python analyze_datasets.py` to see what you have
2. Pick the BEST dataset (most images + overhead angle)
3. Start training on that ONE dataset
4. Get your Busay system running in 4-5 hours!
5. (Optional) Fine-tune later with merged datasets or real Busay footage

**Remember: Done is better than perfect! Start with one dataset and deploy quickly.** 🎓
