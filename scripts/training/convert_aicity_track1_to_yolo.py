#!/usr/bin/env python3
"""
Convert AI City Challenge 2022 Track 1 (Multi-Camera Vehicle Tracking) to YOLO Format
For vehicle detection, tracking, and speed measurement
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

def convert_aicity_track1_to_yolo(
    aicity_root='datasets/aicity_2022_track1',
    output_root='datasets/aicity_2022_track1_yolo'
):
    """
    Convert AI City 2022 Track 1 dataset to YOLO format

    Dataset structure:
    aicity_2022_track1/
    ├── train/
    │   ├── S01/  (Scenario 1)
    │   │   ├── c001/  (Camera 1)
    │   │   │   ├── img1/
    │   │   │   │   ├── 000001.jpg
    │   │   │   │   └── ...
    │   │   │   └── gt/
    │   │   │       └── gt.txt
    │   │   └── c002/  (Camera 2)
    │   └── S02/
    └── test/
    """

    print("="*70)
    print("🚗 AI CITY 2022 TRACK 1 → YOLO CONVERTER")
    print("="*70)
    print()

    # Check if dataset exists
    if not Path(aicity_root).exists():
        print("❌ ERROR: Dataset not found!")
        print(f"   Expected location: {aicity_root}")
        print()
        print("Please:")
        print("1. Register at https://www.aicitychallenge.org/")
        print("2. Download 2022 Track 1 dataset")
        print("3. Extract to:", aicity_root)
        return

    # Create output directories
    output_images_train = Path(output_root) / 'images' / 'train'
    output_images_val = Path(output_root) / 'images' / 'val'
    output_labels_train = Path(output_root) / 'labels' / 'train'
    output_labels_val = Path(output_root) / 'labels' / 'val'

    for dir_path in [output_images_train, output_images_val,
                     output_labels_train, output_labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process training data
    print("📊 Processing training set...")
    train_stats = process_split(
        Path(aicity_root) / 'train',
        output_images_train,
        output_labels_train,
        split='train'
    )

    print()
    print("="*70)
    print("✅ CONVERSION COMPLETE")
    print("="*70)
    print(f"\n📁 Output location: {output_root}")
    print(f"📊 Total images: {train_stats['total_images']}")
    print(f"🚗 Total vehicles: {train_stats['total_vehicles']}")
    print()

    # Create YAML config
    create_yaml_config(output_root)

def process_split(split_dir, output_images, output_labels, split='train'):
    """
    Process train or test split

    AI City Track 1 format:
    <frame_id>, <track_id>, <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <conf>, <class>, <visibility>
    """

    total_images = 0
    total_vehicles = 0

    # Iterate through scenarios (S01, S02, etc.)
    for scenario_dir in sorted(split_dir.glob('S*')):
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name
        print(f"  Processing {scenario_name}...")

        # Iterate through cameras (c001, c002, etc.)
        for camera_dir in sorted(scenario_dir.glob('c*')):
            if not camera_dir.is_dir():
                continue

            camera_name = camera_dir.name

            # Image directory
            img_dir = camera_dir / 'img1'
            if not img_dir.exists():
                continue

            # Ground truth file
            gt_file = camera_dir / 'gt' / 'gt.txt'
            if not gt_file.exists():
                print(f"    ⚠️  No ground truth for {camera_name}, skipping...")
                continue

            # Read ground truth
            gt_data = {}
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue

                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    bbox_left = float(parts[2])
                    bbox_top = float(parts[3])
                    bbox_width = float(parts[4])
                    bbox_height = float(parts[5])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    cls = int(parts[7]) if len(parts) > 7 else 1  # Default to car

                    if frame_id not in gt_data:
                        gt_data[frame_id] = []

                    gt_data[frame_id].append({
                        'track_id': track_id,
                        'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                        'class': cls,
                        'conf': conf
                    })

            # Process images
            for img_file in tqdm(sorted(img_dir.glob('*.jpg')),
                               desc=f"    {camera_name}"):
                frame_id = int(img_file.stem)

                # Skip if no annotations
                if frame_id not in gt_data:
                    continue

                # Read image to get dimensions
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                img_height, img_width = img.shape[:2]

                # Create unique filename
                output_name = f"{scenario_name}_{camera_name}_frame{frame_id:06d}"

                # Copy image
                output_img = output_images / f"{output_name}.jpg"
                shutil.copy(img_file, output_img)

                # Create YOLO label
                output_label = output_labels / f"{output_name}.txt"

                with open(output_label, 'w') as f:
                    for obj in gt_data[frame_id]:
                        bbox = obj['bbox']
                        cls = obj['class']

                        # Convert to YOLO format
                        # AI City: [left, top, width, height]
                        # YOLO: [x_center, y_center, width, height] (normalized)

                        x_center = (bbox[0] + bbox[2] / 2) / img_width
                        y_center = (bbox[1] + bbox[3] / 2) / img_height
                        width = bbox[2] / img_width
                        height = bbox[3] / img_height

                        # Clamp to [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        # YOLO uses class 0 for first class
                        # AI City classes: 1=car, 2=truck, 3=bus, etc.
                        yolo_class = 0  # Map all vehicles to class 0 for simplicity

                        f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        total_vehicles += 1

                total_images += 1

    return {
        'total_images': total_images,
        'total_vehicles': total_vehicles
    }

def create_yaml_config(output_root):
    """Create YOLO dataset configuration file"""

    yaml_content = f"""# AI City Challenge 2022 Track 1 - YOLO Format
# Multi-Camera Vehicle Tracking Dataset
# For vehicle detection, tracking, and speed measurement

# Dataset root
path: {os.path.abspath(output_root)}

# Train and validation sets
train: images/train
val: images/val

# Vehicle classes (simplified)
names:
  0: vehicle

# Number of classes
nc: 1

# Dataset info
download: |
  # AI City Challenge 2022 Track 1
  # 1. Register at https://www.aicitychallenge.org/
  # 2. Download Track 1 dataset
  # 3. Extract to datasets/aicity_2022_track1/
  # 4. Run: python convert_aicity_track1_to_yolo.py
"""

    yaml_path = Path('aicity_2022_track1.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ Created YOLO config: {yaml_path}")

def main():
    """Main conversion function"""

    print("="*70)
    print("🚗 AI CITY 2022 TRACK 1 CONVERTER")
    print("   Multi-Camera Vehicle Tracking → YOLO Format")
    print("="*70)
    print()

    # Convert dataset
    convert_aicity_track1_to_yolo()

    print("="*70)
    print("✅ READY TO TRAIN!")
    print("="*70)
    print()
    print("🚀 Next steps:")
    print("   python train_vehicle_detector.py \\")
    print("     --data aicity_2022_track1.yaml \\")
    print("     --model n \\")
    print("     --batch 4 \\")
    print("     --epochs 100 \\")
    print("     --name aicity_vehicle_tracking")
    print()

if __name__ == "__main__":
    main()
