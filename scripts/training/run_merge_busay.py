#!/usr/bin/env python3
"""
Easy-to-use script to merge Busay datasets
Just run this and follow the prompts!
"""

import os
from pathlib import Path
from merge_busay_datasets import merge_vehicle_detection_datasets, prepare_accident_dataset


def find_datasets():
    """
    Search for downloaded datasets
    """
    search_paths = [
        Path.home() / 'Downloads',
        Path.home(),
        Path('/home/user'),
        Path('/home/user/Road_Sentinel'),
    ]

    found_datasets = {
        'traffic': None,
        'daynight': None,
        'accident': None
    }

    print("🔍 Searching for datasets...")
    print()

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for item in search_path.iterdir():
            if not item.is_dir():
                continue

            name_lower = item.name.lower()

            # Look for Traffic Surveillance
            if 'traffic' in name_lower and 'surveillance' in name_lower:
                if (item / 'data.yaml').exists():
                    found_datasets['traffic'] = item
                    print(f"✅ Found Traffic Surveillance: {item}")

            # Look for Day/Night
            if 'vehicle' in name_lower and ('day' in name_lower or 'night' in name_lower):
                if (item / 'data.yaml').exists():
                    found_datasets['daynight'] = item
                    print(f"✅ Found Day/Night Vehicle: {item}")

            # Look for Accident
            if 'accident' in name_lower:
                if (item / 'data.yaml').exists():
                    found_datasets['accident'] = item
                    print(f"✅ Found Accident Detection: {item}")

    print()
    return found_datasets


def main():
    """
    Main function
    """
    print("="*80)
    print("🚀 BUSAY DATASET MERGER - AUTOMATIC")
    print("="*80)
    print()

    # Find datasets
    datasets = find_datasets()

    # Check what we found
    traffic_found = datasets['traffic'] is not None
    daynight_found = datasets['daynight'] is not None
    accident_found = datasets['accident'] is not None

    if not traffic_found and not daynight_found:
        print("❌ No vehicle detection datasets found!")
        print()
        print("Please ensure your datasets are in:")
        print("  • ~/Downloads")
        print("  • ~/")
        print("  • /home/user")
        print()
        print("Dataset folders should contain 'data.yaml' file")
        print()
        return

    # Show what we'll do
    print("="*80)
    print("📋 MERGE PLAN:")
    print("="*80)
    print()

    if traffic_found and daynight_found:
        print("✅ MODEL 1: Vehicle Detection (Merging 2 datasets)")
        print(f"   • Traffic Surveillance: {datasets['traffic'].name}")
        print(f"   • Day/Night Vehicles: {datasets['daynight'].name}")
        print(f"   → Output: busay_vehicle_detection/")
        print()
    elif traffic_found:
        print("⚠️  MODEL 1: Vehicle Detection (Only Traffic Surveillance found)")
        print(f"   • Traffic Surveillance: {datasets['traffic'].name}")
        print(f"   → You can train on this alone, or add Day/Night dataset")
        print()
    elif daynight_found:
        print("⚠️  MODEL 1: Vehicle Detection (Only Day/Night found)")
        print(f"   • Day/Night Vehicles: {datasets['daynight'].name}")
        print(f"   → You can train on this alone, or add Traffic Surveillance")
        print()

    if accident_found:
        print("✅ MODEL 2: Crash Detection")
        print(f"   • Accident Detection: {datasets['accident'].name}")
        print(f"   → Output: busay_accident_detection/")
        print()
    else:
        print("⚠️  MODEL 2: Crash Detection dataset not found")
        print("   → You can add this later")
        print()

    print("="*80)
    print()

    # Ask for confirmation
    response = input("Proceed with merge? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Merge cancelled")
        return

    print()
    print("="*80)
    print("🔄 STARTING MERGE...")
    print("="*80)
    print()

    # Merge vehicle detection datasets
    if traffic_found and daynight_found:
        print("📦 Merging Vehicle Detection datasets...")
        print()
        merge_vehicle_detection_datasets(
            traffic_surveillance_path=str(datasets['traffic']),
            day_night_path=str(datasets['daynight']),
            output_path='./busay_vehicle_detection'
        )
        print()
    elif traffic_found:
        print("📦 Using Traffic Surveillance only...")
        print(f"   → Already in YOLO format at: {datasets['traffic']}")
        print(f"   → Train with: --data {datasets['traffic']}/data.yaml")
        print()
    elif daynight_found:
        print("📦 Using Day/Night only...")
        print(f"   → Need to remap classes 0-7 to standard names")
        print(f"   → Use merge script with just this dataset")
        print()

    # Prepare accident detection
    if accident_found:
        print("🚨 Preparing Accident Detection dataset...")
        print()
        prepare_accident_dataset(
            accident_path=str(datasets['accident']),
            output_path='./busay_accident_detection'
        )
        print()

    # Final instructions
    print("="*80)
    print("✅ ALL DONE!")
    print("="*80)
    print()
    print("🎯 NEXT STEPS:")
    print()

    if traffic_found and daynight_found:
        print("1️⃣  Train Model 1 (Vehicle Detection):")
        print()
        print("   python train_vehicle_detector.py \\")
        print("     --data ./busay_vehicle_detection/data.yaml \\")
        print("     --model n \\")
        print("     --batch 4 \\")
        print("     --epochs 100 \\")
        print("     --name busay_vehicle_v1")
        print()
        print("   Training time: ~6-8 hours (merged dataset is larger)")
        print()

    if accident_found:
        print("2️⃣  Train Model 2 (Crash Detection):")
        print()
        print("   python train_vehicle_detector.py \\")
        print("     --data ./busay_accident_detection/data.yaml \\")
        print("     --model n \\")
        print("     --batch 4 \\")
        print("     --epochs 100 \\")
        print("     --name busay_accident_v1")
        print()
        print("   Training time: ~3-4 hours")
        print()

    print("💡 TIP: You can train both models simultaneously in different terminals!")
    print()
    print("📖 See: scripts/training/DUAL_MODEL_TRAINING_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
