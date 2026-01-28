#!/usr/bin/env python3
"""
Merge multiple Roboflow datasets into one unified dataset
Handles class name remapping and deduplication
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# Standard vehicle class mapping for Road Sentinel
STANDARD_CLASSES = {
    'car': 0,
    'motorcycle': 1,
    'bicycle': 2,
    'bus': 3,
    'truck': 4,
    'vehicle': 0,  # Map generic 'vehicle' to 'car'
    'motorbike': 1,  # Alternative spelling
    'bike': 2,  # Alternative name
    'van': 4,  # Map van to truck
    'scooter': 1,  # Map scooter to motorcycle
}


def load_dataset_config(dataset_path):
    """Load dataset configuration from data.yaml"""
    data_yaml = Path(dataset_path) / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"No data.yaml found in {dataset_path}")

    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_class_mapping(source_classes, target_classes=None):
    """
    Create mapping from source classes to target classes

    Args:
        source_classes: List of class names from source dataset
        target_classes: Dict of target class names (optional, uses STANDARD_CLASSES if None)

    Returns:
        Dict mapping source class ID to target class ID
    """
    if target_classes is None:
        target_classes = STANDARD_CLASSES

    mapping = {}

    for src_id, src_name in enumerate(source_classes):
        src_name_lower = src_name.lower().strip()

        if src_name_lower in target_classes:
            mapping[src_id] = target_classes[src_name_lower]
        else:
            print(f"⚠️  Warning: Class '{src_name}' not in standard mapping, skipping...")

    return mapping


def remap_annotation(label_file, class_mapping):
    """
    Remap class IDs in a YOLO annotation file

    Args:
        label_file: Path to label file
        class_mapping: Dict mapping old class ID to new class ID

    Returns:
        List of remapped annotation lines
    """
    if not label_file.exists():
        return []

    remapped_lines = []

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class_id = int(parts[0])

                if old_class_id in class_mapping:
                    new_class_id = class_mapping[old_class_id]
                    # Replace class ID but keep coordinates
                    parts[0] = str(new_class_id)
                    remapped_lines.append(' '.join(parts))

    return remapped_lines


def merge_datasets(dataset_paths, output_path, dataset_name="merged_dataset"):
    """
    Merge multiple datasets into one

    Args:
        dataset_paths: List of paths to datasets to merge
        output_path: Path where merged dataset will be created
        dataset_name: Name for the merged dataset
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output structure
    splits = ['train', 'valid', 'test']
    for split in splits:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"🔄 MERGING {len(dataset_paths)} DATASETS")
    print("="*80)
    print()

    # Statistics
    total_images = 0
    split_counts = defaultdict(int)

    # Process each dataset
    for ds_idx, ds_path in enumerate(dataset_paths, 1):
        ds_path = Path(ds_path)
        print(f"📦 Processing dataset {ds_idx}/{len(dataset_paths)}: {ds_path.name}")

        # Load config
        try:
            config = load_dataset_config(ds_path)
            source_classes = config.get('names', [])
            print(f"   Classes: {', '.join(source_classes)}")

            # Create class mapping
            class_mapping = create_class_mapping(source_classes)
            print(f"   Mapped {len(class_mapping)} classes")

            # Process each split
            for split in splits:
                # Find images directory
                img_dir = ds_path / split / 'images'
                if not img_dir.exists():
                    img_dir = ds_path / config.get(split, split) / 'images'

                if not img_dir.exists():
                    print(f"   ⚠️  No {split} images found, skipping...")
                    continue

                # Find labels directory
                lbl_dir = img_dir.parent / 'labels'

                # Copy and remap files
                image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

                for img_file in image_files:
                    # Generate unique filename
                    new_name = f"ds{ds_idx}_{img_file.name}"

                    # Copy image
                    shutil.copy2(
                        img_file,
                        output_path / split / 'images' / new_name
                    )

                    # Copy and remap label
                    lbl_file = lbl_dir / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        remapped_lines = remap_annotation(lbl_file, class_mapping)

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
    merged_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 5,
        'names': ['car', 'motorcycle', 'bicycle', 'bus', 'truck']
    }

    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False)

    print("="*80)
    print("✅ MERGE COMPLETE!")
    print("="*80)
    print()
    print(f"📁 Output location: {output_path}")
    print(f"📊 Total images: {total_images:,}")
    print(f"   • Train: {split_counts['train']:,}")
    print(f"   • Valid: {split_counts['valid']:,}")
    print(f"   • Test: {split_counts['test']:,}")
    print()
    print(f"🏷️  Classes: {', '.join(merged_config['names'])}")
    print()
    print(f"📄 Config file: {output_path / 'data.yaml'}")
    print()
    print("🚀 Ready to train!")
    print(f"   python train_vehicle_detector.py --data {output_path / 'data.yaml'} --epochs 100 --batch 4")
    print()


def main():
    """
    Main function - example usage
    """
    print("="*80)
    print("🔄 ROBOFLOW DATASET MERGER")
    print("="*80)
    print()

    print("This script merges multiple datasets with class remapping.")
    print()
    print("📋 USAGE EXAMPLE:")
    print()
    print("from merge_datasets import merge_datasets")
    print()
    print("datasets_to_merge = [")
    print("    '/path/to/dataset1',")
    print("    '/path/to/dataset2',")
    print("    '/path/to/dataset3',")
    print("]")
    print()
    print("merge_datasets(")
    print("    dataset_paths=datasets_to_merge,")
    print("    output_path='./merged_busay_dataset',")
    print("    dataset_name='Busay Traffic Merged'")
    print(")")
    print()
    print("="*80)
    print("💡 STANDARD CLASS MAPPING:")
    print("="*80)
    print()
    for name, idx in sorted(set(STANDARD_CLASSES.items()), key=lambda x: x[1]):
        print(f"   {idx}: {name}")
    print()
    print("Alternative names are automatically mapped:")
    print("   • 'vehicle' → 'car'")
    print("   • 'motorbike' → 'motorcycle'")
    print("   • 'bike' → 'bicycle'")
    print("   • 'van' → 'truck'")
    print("   • 'scooter' → 'motorcycle'")
    print()


if __name__ == "__main__":
    main()
