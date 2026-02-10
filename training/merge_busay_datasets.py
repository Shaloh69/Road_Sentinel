#!/usr/bin/env python3
"""
Custom merger for Busay project datasets
Handles the specific datasets downloaded from Roboflow
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict


def merge_vehicle_detection_datasets(traffic_surveillance_path, day_night_path, output_path):
    """
    Merge Traffic Surveillance + Day/Night Vehicle Detection datasets

    Dataset schemas:
    - Traffic Surveillance: bus, car, motorbike, truck
    - Day/Night: 0-7 (0=motorcycle_day, 1=car_day, 2=bus_day, 3=truck_day,
                       4=motorcycle_night, 5=car_night, 6=bus_night, 7=truck_night)

    Output schema:
    - 0: car (includes day+night)
    - 1: motorcycle (includes day+night)
    - 2: bicycle (placeholder for future)
    - 3: bus (includes day+night)
    - 4: truck (includes day+night)
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output structure
    splits = ['train', 'valid', 'test']
    for split in splits:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("🔄 MERGING VEHICLE DETECTION DATASETS FOR BUSAY")
    print("="*80)
    print()

    # Define class mappings
    # Traffic Surveillance mapping
    traffic_mapping = {
        'car': 0,
        'motorbike': 1,
        'bus': 3,
        'truck': 4
    }

    # Day/Night dataset mapping
    # 0=motorcycle_day → 1, 1=car_day → 0, 2=bus_day → 3, 3=truck_day → 4
    # 4=motorcycle_night → 1, 5=car_night → 0, 6=bus_night → 3, 7=truck_night → 4
    daynight_mapping = {
        0: 1,  # motorcycle (day) → motorcycle
        1: 0,  # car (day) → car
        2: 3,  # bus (day) → bus
        3: 4,  # truck (day) → truck
        4: 1,  # motorcycle (night) → motorcycle
        5: 0,  # car (night) → car
        6: 3,  # bus (night) → bus
        7: 4,  # truck (night) → truck
    }

    total_images = 0
    split_counts = defaultdict(int)

    # Process Traffic Surveillance dataset
    print("📦 Dataset 1: Traffic Surveillance System")
    print("   Classes: bus, car, motorbike, truck")

    traffic_path = Path(traffic_surveillance_path)

    try:
        # Load config
        with open(traffic_path / 'data.yaml', 'r') as f:
            traffic_config = yaml.safe_load(f)

        traffic_classes = traffic_config.get('names', [])

        # Create class ID to name mapping
        traffic_id_to_name = {i: name for i, name in enumerate(traffic_classes)}

        for split in splits:
            img_dir = traffic_path / split / 'images'
            if not img_dir.exists():
                img_dir = traffic_path / traffic_config.get(split, split) / 'images'

            if not img_dir.exists():
                print(f"   ⚠️  No {split} found, skipping...")
                continue

            lbl_dir = img_dir.parent / 'labels'
            image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

            for img_file in image_files:
                new_name = f"traffic_{img_file.name}"

                # Copy image
                shutil.copy2(img_file, output_path / split / 'images' / new_name)

                # Remap labels
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    remapped_lines = []
                    with open(lbl_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_id = int(parts[0])
                                class_name = traffic_id_to_name.get(old_id, '').lower()

                                if class_name in traffic_mapping:
                                    new_id = traffic_mapping[class_name]
                                    parts[0] = str(new_id)
                                    remapped_lines.append(' '.join(parts))

                    if remapped_lines:
                        out_lbl = output_path / split / 'labels' / f"{Path(new_name).stem}.txt"
                        with open(out_lbl, 'w') as f:
                            f.write('\n'.join(remapped_lines))

                total_images += 1
                split_counts[split] += 1

            print(f"   ✅ {split}: {len(image_files)} images")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Process Day/Night dataset
    print("📦 Dataset 2: Vehicle Detection (Day & Night)")
    print("   Classes: 0-7 (motorcycle, car, bus, truck x day/night)")

    daynight_path = Path(day_night_path)

    try:
        # Load config
        with open(daynight_path / 'data.yaml', 'r') as f:
            daynight_config = yaml.safe_load(f)

        for split in splits:
            img_dir = daynight_path / split / 'images'
            if not img_dir.exists():
                img_dir = daynight_path / daynight_config.get(split, split) / 'images'

            if not img_dir.exists():
                print(f"   ⚠️  No {split} found, skipping...")
                continue

            lbl_dir = img_dir.parent / 'labels'
            image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

            for img_file in image_files:
                new_name = f"daynight_{img_file.name}"

                # Copy image
                shutil.copy2(img_file, output_path / split / 'images' / new_name)

                # Remap labels (0-7 → standard classes)
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    remapped_lines = []
                    with open(lbl_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_id = int(parts[0])

                                if old_id in daynight_mapping:
                                    new_id = daynight_mapping[old_id]
                                    parts[0] = str(new_id)
                                    remapped_lines.append(' '.join(parts))

                    if remapped_lines:
                        out_lbl = output_path / split / 'labels' / f"{Path(new_name).stem}.txt"
                        with open(out_lbl, 'w') as f:
                            f.write('\n'.join(remapped_lines))

                total_images += 1
                split_counts[split] += 1

            print(f"   ✅ {split}: {len(image_files)} images")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Create merged data.yaml
    # Check which splits actually have data
    actual_splits = {}
    if split_counts['train'] > 0:
        actual_splits['train'] = 'train/images'
    if split_counts['valid'] > 0:
        actual_splits['val'] = 'valid/images'
    if split_counts['test'] > 0:
        actual_splits['test'] = 'test/images'

    # If no valid/test, use train for validation (YOLOv8 will auto-split)
    if 'val' not in actual_splits and 'train' in actual_splits:
        print("   ⚠️  No validation set found - YOLO will auto-split from training data")
        actual_splits['val'] = 'train/images'

    merged_config = {
        'path': '.',
        **actual_splits,
        'nc': 5,
        'names': ['car', 'motorcycle', 'bicycle', 'bus', 'truck']
    }

    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False)

    print("="*80)
    print("✅ VEHICLE DETECTION MERGE COMPLETE!")
    print("="*80)
    print()
    print(f"📁 Output: {output_path}")
    print(f"📊 Total images: {total_images:,}")
    print(f"   • Train: {split_counts['train']:,}")
    print(f"   • Valid: {split_counts['valid']:,}")
    print(f"   • Test: {split_counts['test']:,}")
    print()
    print(f"🏷️  Classes: car, motorcycle, bicycle, bus, truck")
    print(f"   (Includes both day and night images merged into each class)")
    print()
    print(f"📄 Config: {output_path / 'data.yaml'}")
    print()
    print("🚀 Ready to train Model 1 (Vehicle Detection)!")
    print()


def prepare_accident_dataset(accident_path, output_path):
    """
    Prepare Accident Detection dataset for Model 2
    Simplifies to binary: accident vs no_accident

    Original classes: Accident, NoAccident, car, mild, moderate, motor_cycle, severe
    Simplified: 0=no_accident, 1=accident
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("🚨 PREPARING ACCIDENT DETECTION DATASET FOR BUSAY")
    print("="*80)
    print()

    # Map to binary classification
    accident_mapping = {
        'NoAccident': 0,
        'Accident': 1,
        'car': 0,  # Car present but no accident
        'motor_cycle': 0,  # Motorcycle present but no accident
        'mild': 1,  # Mild accident
        'moderate': 1,  # Moderate accident
        'severe': 1,  # Severe accident
    }

    splits = ['train', 'valid', 'test']
    for split in splits:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    accident_path = Path(accident_path)
    total_images = 0
    split_counts = defaultdict(int)

    try:
        with open(accident_path / 'data.yaml', 'r') as f:
            config = yaml.safe_load(f)

        classes = config.get('names', [])
        id_to_name = {i: name for i, name in enumerate(classes)}

        print(f"📦 Original classes: {', '.join(classes)}")
        print(f"📊 Mapping to: 0=no_accident, 1=accident")
        print()

        for split in splits:
            img_dir = accident_path / split / 'images'
            if not img_dir.exists():
                img_dir = accident_path / config.get(split, split) / 'images'

            if not img_dir.exists():
                print(f"   ⚠️  No {split} found, skipping...")
                continue

            lbl_dir = img_dir.parent / 'labels'
            image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

            for img_file in image_files:
                new_name = f"accident_{img_file.name}"

                # Copy image
                shutil.copy2(img_file, output_path / split / 'images' / new_name)

                # Remap to binary
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    remapped_lines = []
                    with open(lbl_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_id = int(parts[0])
                                class_name = id_to_name.get(old_id, '')

                                if class_name in accident_mapping:
                                    new_id = accident_mapping[class_name]
                                    parts[0] = str(new_id)
                                    remapped_lines.append(' '.join(parts))

                    if remapped_lines:
                        out_lbl = output_path / split / 'labels' / f"{Path(new_name).stem}.txt"
                        with open(out_lbl, 'w') as f:
                            f.write('\n'.join(remapped_lines))

                total_images += 1
                split_counts[split] += 1

            print(f"   ✅ {split}: {len(image_files)} images")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Create data.yaml
    # Check which splits actually have data
    actual_splits = {}
    if split_counts['train'] > 0:
        actual_splits['train'] = 'train/images'
    if split_counts['valid'] > 0:
        actual_splits['val'] = 'valid/images'
    if split_counts['test'] > 0:
        actual_splits['test'] = 'test/images'

    # If no valid/test, use train for validation (YOLOv8 will auto-split)
    if 'val' not in actual_splits and 'train' in actual_splits:
        print("   ⚠️  No validation set found - YOLO will auto-split from training data")
        actual_splits['val'] = 'train/images'

    accident_config = {
        'path': '.',
        **actual_splits,
        'nc': 2,
        'names': ['no_accident', 'accident']
    }

    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(accident_config, f, default_flow_style=False)

    print("="*80)
    print("✅ ACCIDENT DETECTION DATASET READY!")
    print("="*80)
    print()
    print(f"📁 Output: {output_path}")
    print(f"📊 Total images: {total_images:,}")
    print(f"   • Train: {split_counts['train']:,}")
    print(f"   • Valid: {split_counts['valid']:,}")
    print(f"   • Test: {split_counts['test']:,}")
    print()
    print(f"🏷️  Classes: no_accident, accident")
    print()
    print(f"📄 Config: {output_path / 'data.yaml'}")
    print()
    print("🚀 Ready to train Model 2 (Crash Detection)!")
    print()


def main():
    """
    Example usage for Busay project
    """
    print("="*80)
    print("🔄 BUSAY DATASET MERGER - CUSTOM FOR YOUR DATASETS")
    print("="*80)
    print()
    print("This script merges your specific Roboflow datasets:")
    print("  1. Traffic Surveillance System")
    print("  2. Vehicle Detection (Day & Night)")
    print("  3. Accident Detection (separate)")
    print()
    print("="*80)
    print("📋 USAGE:")
    print("="*80)
    print()
    print("from merge_busay_datasets import merge_vehicle_detection_datasets, prepare_accident_dataset")
    print()
    print("# Merge datasets for Model 1 (Vehicle Detection)")
    print("merge_vehicle_detection_datasets(")
    print("    traffic_surveillance_path='/path/to/Traffic-surveillance-system-1',")
    print("    day_night_path='/path/to/Vehicle-Detection-Day-Night-1',")
    print("    output_path='./busay_vehicle_detection'")
    print(")")
    print()
    print("# Prepare dataset for Model 2 (Crash Detection)")
    print("prepare_accident_dataset(")
    print("    accident_path='/path/to/Accident-detection-1',")
    print("    output_path='./busay_accident_detection'")
    print(")")
    print()
    print("="*80)
    print("🎯 DUAL MODEL TRAINING:")
    print("="*80)
    print()
    print("# Train Model 1 (Vehicle Detection & Speed)")
    print("python train_vehicle_detector.py \\")
    print("  --data ./busay_vehicle_detection/data.yaml \\")
    print("  --model n \\")
    print("  --batch 4 \\")
    print("  --epochs 100 \\")
    print("  --name busay_vehicle_model")
    print()
    print("# Train Model 2 (Crash Detection)")
    print("python train_vehicle_detector.py \\")
    print("  --data ./busay_accident_detection/data.yaml \\")
    print("  --model n \\")
    print("  --batch 4 \\")
    print("  --epochs 100 \\")
    print("  --name busay_accident_model")
    print()


if __name__ == "__main__":
    main()
