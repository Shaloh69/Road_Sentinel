#!/usr/bin/env python3
"""
Download and Setup BDD100K Dataset for Vehicle Detection
Includes day and night driving scenes - perfect for night vision cameras
"""

import os
import requests
from pathlib import Path

def download_bdd100k():
    """
    Guide to download BDD100K dataset
    """

    print("="*70)
    print("🌙 BDD100K DATASET SETUP FOR NIGHT VISION VEHICLE DETECTION")
    print("="*70)
    print()

    print("📋 STEP-BY-STEP GUIDE:")
    print()

    print("1️⃣  REGISTER (FREE)")
    print("   → Go to: https://bdd-data.berkeley.edu/")
    print("   → Click 'Register'")
    print("   → Fill form (use your school email for thesis)")
    print("   → Verify email")
    print()

    print("2️⃣  DOWNLOAD DETECTION SUBSET")
    print("   → Login to BDD100K portal")
    print("   → Go to 'Downloads' section")
    print("   → Download these files:")
    print()
    print("   ✅ bdd100k_images_100k.zip (Detection images - ~7GB)")
    print("   ✅ bdd100k_labels_release.zip (Annotations - ~500MB)")
    print()
    print("   💡 TIP: For thesis, you can use smaller subset:")
    print("      - bdd100k_images_10k.zip (10,000 images - ~700MB)")
    print()

    print("3️⃣  EXTRACT FILES")
    print("   Create folder structure:")
    print("   datasets/")
    print("   └── bdd100k/")
    print("       ├── images/")
    print("       │   ├── 10k/")
    print("       │   │   ├── train/")
    print("       │   │   └── val/")
    print("       └── labels/")
    print("           └── det_20/")
    print()

    print("4️⃣  CONVERT TO YOLO FORMAT")
    print("   Run: python convert_bdd_to_yolo.py")
    print()

    print("5️⃣  TRAIN YOUR MODEL")
    print("   python train_vehicle_detector.py --data bdd100k.yaml --epochs 100")
    print()

    print("="*70)
    print("📊 DATASET STATISTICS:")
    print("="*70)
    print()
    print("Full BDD100K (100k images):")
    print("  - Training: 70,000 images")
    print("  - Validation: 10,000 images")
    print("  - Test: 20,000 images")
    print("  - Nighttime: ~30,000 images (30%)")
    print("  - Size: ~7GB images + 500MB labels")
    print()
    print("Smaller Subset (10k images) - RECOMMENDED FOR THESIS:")
    print("  - Training: 7,000 images")
    print("  - Validation: 1,000 images")
    print("  - Test: 2,000 images")
    print("  - Nighttime: ~3,000 images")
    print("  - Size: ~700MB")
    print()

    print("="*70)
    print("🎯 VEHICLE CLASSES IN BDD100K:")
    print("="*70)
    print()
    print("  0: car")
    print("  1: truck")
    print("  2: bus")
    print("  3: motorcycle")
    print("  4: bicycle")
    print()
    print("Perfect for your speed detection system!")
    print()

    print("="*70)
    print("🌙 NIGHT VISION BENEFITS:")
    print("="*70)
    print()
    print("✅ Training on BDD100K nighttime images will help your model:")
    print("   - Detect vehicles in low light")
    print("   - Work with your infrared/night vision camera")
    print("   - Handle Barangay Busay nighttime conditions")
    print("   - Improve accuracy in all lighting conditions")
    print()

    print("="*70)
    print("⏱️  TRAINING TIME ESTIMATES (RTX 3050, batch=4):")
    print("="*70)
    print()
    print("  10k subset, 50 epochs:  ~2-3 hours")
    print("  10k subset, 100 epochs: ~4-5 hours")
    print("  100k full, 50 epochs:   ~12-15 hours")
    print("  100k full, 100 epochs:  ~24-30 hours")
    print()

    print("💡 RECOMMENDATION:")
    print("   Use 10k subset for thesis - good balance of:")
    print("   - Accuracy (better than COCO for roads)")
    print("   - Training time (4-5 hours)")
    print("   - Storage (700MB vs 7GB)")
    print()

    print("="*70)
    print("📞 NEED HELP?")
    print("="*70)
    print()
    print("BDD100K Documentation: https://doc.bdd100k.com/")
    print("GitHub: https://github.com/bdd100k/bdd100k")
    print("Paper: https://arxiv.org/abs/1805.04687")
    print()

    print("="*70)
    print("🚀 READY TO START?")
    print("="*70)
    print()
    print("1. Register at: https://bdd-data.berkeley.edu/")
    print("2. Download bdd100k_images_10k.zip")
    print("3. Download bdd100k_labels_release.zip")
    print("4. Extract to datasets/bdd100k/")
    print("5. Run: python convert_bdd_to_yolo.py")
    print("6. Train: python train_vehicle_detector.py --data bdd100k.yaml")
    print()

if __name__ == "__main__":
    download_bdd100k()
