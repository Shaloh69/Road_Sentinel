#!/usr/bin/env python3
"""
Convert BDD100K Dataset to YOLO Format
Filters to vehicle classes only and includes nighttime images
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# BDD100K to YOLO class mapping (vehicles only)
BDD_TO_YOLO = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'motorcycle': 3,
    'bike': 4,  # bicycle in BDD100K is called 'bike'
}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert BDD100K bbox format to YOLO format

    BDD100K: [x1, y1, x2, y2] (absolute coordinates)
    YOLO: [x_center, y_center, width, height] (normalized 0-1)
    """
    x1, y1, x2, y2 = bbox

    # Calculate center point and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm

def is_nighttime(image_name, attributes):
    """
    Check if image is taken at nighttime
    """
    if attributes and 'timeofday' in attributes:
        return attributes['timeofday'] == 'night'
    return False

def convert_bdd100k_to_yolo(
    bdd_root='datasets/bdd100k',
    output_root='datasets/bdd100k_yolo',
    split='train',
    include_night_only=False
):
    """
    Convert BDD100K labels to YOLO format

    Args:
        bdd_root: Path to BDD100K dataset
        output_root: Path to output YOLO format dataset
        split: 'train' or 'val'
        include_night_only: If True, only include nighttime images
    """

    print(f"\n{'='*70}")
    print(f"Converting BDD100K {split} set to YOLO format")
    print(f"{'='*70}\n")

    # Paths
    labels_file = Path(bdd_root) / 'labels' / 'det_20' / f'det_{split}.json'
    images_dir = Path(bdd_root) / 'images' / '10k' / split

    output_images_dir = Path(output_root) / 'images' / split
    output_labels_dir = Path(output_root) / 'labels' / split

    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Load BDD100K labels
    print(f"📂 Loading labels from: {labels_file}")
    with open(labels_file, 'r') as f:
        bdd_labels = json.load(f)

    print(f"📊 Total images in {split}: {len(bdd_labels)}")

    # Statistics
    total_images = 0
    nighttime_images = 0
    skipped_no_vehicles = 0
    total_vehicles = 0

    # Process each image
    for img_data in tqdm(bdd_labels, desc=f"Converting {split}"):
        img_name = img_data['name']

        # Check if nighttime
        attributes = img_data.get('attributes', {})
        is_night = is_nighttime(img_name, attributes)

        # Skip daytime images if only night is requested
        if include_night_only and not is_night:
            continue

        if is_night:
            nighttime_images += 1

        # Get labels (objects)
        labels = img_data.get('labels', [])

        # Filter to vehicle classes only
        vehicle_labels = []
        for label in labels:
            category = label.get('category', '')
            if category in BDD_TO_YOLO:
                vehicle_labels.append(label)

        # Skip images with no vehicles
        if not vehicle_labels:
            skipped_no_vehicles += 1
            continue

        # Copy image
        src_img = images_dir / img_name
        dst_img = output_images_dir / img_name

        if not src_img.exists():
            print(f"⚠️  Image not found: {src_img}")
            continue

        shutil.copy(src_img, dst_img)

        # Create YOLO label file
        label_name = img_name.replace('.jpg', '.txt')
        label_path = output_labels_dir / label_name

        # Assuming standard image size (check actual size if needed)
        img_width = 1280  # BDD100K standard width
        img_height = 720  # BDD100K standard height

        with open(label_path, 'w') as f:
            for label in vehicle_labels:
                category = label['category']
                class_id = BDD_TO_YOLO[category]

                # Get bounding box
                box2d = label.get('box2d', {})
                x1 = box2d.get('x1', 0)
                y1 = box2d.get('y1', 0)
                x2 = box2d.get('x2', 0)
                y2 = box2d.get('y2', 0)

                # Convert to YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    [x1, y1, x2, y2], img_width, img_height
                )

                # Write YOLO format: class x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_vehicles += 1

        total_images += 1

    # Print statistics
    print(f"\n{'='*70}")
    print(f"✅ CONVERSION COMPLETE - {split.upper()} SET")
    print(f"{'='*70}\n")
    print(f"📊 Statistics:")
    print(f"   Total images processed: {total_images}")
    print(f"   Nighttime images: {nighttime_images} ({nighttime_images/max(total_images,1)*100:.1f}%)")
    print(f"   Images skipped (no vehicles): {skipped_no_vehicles}")
    print(f"   Total vehicle annotations: {total_vehicles}")
    print(f"   Average vehicles per image: {total_vehicles/max(total_images,1):.1f}")
    print(f"\n📁 Output:")
    print(f"   Images: {output_images_dir}")
    print(f"   Labels: {output_labels_dir}")
    print()

def create_yolo_yaml(output_root='datasets/bdd100k_yolo'):
    """
    Create YOLO dataset configuration file
    """
    yaml_content = f"""# BDD100K Vehicle Detection Dataset - YOLO Format
# Includes day and night driving scenes
# Perfect for night vision/infrared cameras

# Dataset root
path: {os.path.abspath(output_root)}

# Train and validation sets
train: images/train
val: images/val

# Vehicle classes
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle

# Number of classes
nc: 5

# Dataset info
download: |
  # BDD100K Dataset Setup:
  # 1. Register at https://bdd-data.berkeley.edu/
  # 2. Download bdd100k_images_10k.zip
  # 3. Download bdd100k_labels_release.zip
  # 4. Extract to datasets/bdd100k/
  # 5. Run: python convert_bdd_to_yolo.py
"""

    yaml_path = Path('bdd100k.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ Created YOLO config: {yaml_path}")

def main():
    """
    Main conversion script
    """
    print("="*70)
    print("🌙 BDD100K TO YOLO CONVERTER - NIGHT VISION READY")
    print("="*70)
    print()

    # Check if BDD100K exists
    bdd_root = 'datasets/bdd100k'
    if not Path(bdd_root).exists():
        print("❌ ERROR: BDD100K dataset not found!")
        print()
        print("Please:")
        print("1. Register at https://bdd-data.berkeley.edu/")
        print("2. Download bdd100k_images_10k.zip and bdd100k_labels_release.zip")
        print("3. Extract to datasets/bdd100k/")
        print("4. Run this script again")
        print()
        print("Expected structure:")
        print("datasets/")
        print("└── bdd100k/")
        print("    ├── images/")
        print("    │   └── 10k/")
        print("    │       ├── train/")
        print("    │       └── val/")
        print("    └── labels/")
        print("        └── det_20/")
        print("            ├── det_train.json")
        print("            └── det_val.json")
        return

    # Convert training set
    convert_bdd100k_to_yolo(
        bdd_root=bdd_root,
        output_root='datasets/bdd100k_yolo',
        split='train',
        include_night_only=False  # Set to True for night-only dataset
    )

    # Convert validation set
    convert_bdd100k_to_yolo(
        bdd_root=bdd_root,
        output_root='datasets/bdd100k_yolo',
        split='val',
        include_night_only=False  # Set to True for night-only dataset
    )

    # Create YAML config
    create_yolo_yaml('datasets/bdd100k_yolo')

    print("="*70)
    print("✅ ALL DONE! Ready to train!")
    print("="*70)
    print()
    print("🚀 Next steps:")
    print("   python train_vehicle_detector.py --data bdd100k.yaml --epochs 100 --batch 4")
    print()

if __name__ == "__main__":
    main()
