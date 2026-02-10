#!/usr/bin/env python3
"""
Analyze downloaded Roboflow datasets to help choose the best one
"""

import os
import yaml
from pathlib import Path

def analyze_dataset(dataset_path):
    """
    Analyze a single dataset and print statistics
    """
    data_yaml = Path(dataset_path) / "data.yaml"

    if not data_yaml.exists():
        print(f"❌ No data.yaml found in {dataset_path}")
        return None

    # Load dataset config
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # Count images
    train_path = Path(dataset_path) / config.get('train', 'train/images')
    val_path = Path(dataset_path) / config.get('val', 'valid/images')
    test_path = Path(dataset_path) / config.get('test', 'test/images')

    train_count = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png'))) if train_path.exists() else 0
    val_count = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png'))) if val_path.exists() else 0
    test_count = len(list(test_path.glob('*.jpg'))) + len(list(test_path.glob('*.png'))) if test_path.exists() else 0

    total_images = train_count + val_count + test_count

    # Get class names
    class_names = config.get('names', [])
    num_classes = config.get('nc', len(class_names))

    return {
        'path': dataset_path,
        'name': Path(dataset_path).name,
        'total_images': total_images,
        'train_images': train_count,
        'val_images': val_count,
        'test_images': test_count,
        'num_classes': num_classes,
        'classes': class_names,
        'data_yaml': str(data_yaml)
    }


def find_datasets(search_path='.'):
    """
    Find all datasets in the given path
    """
    datasets = []

    for root, dirs, files in os.walk(search_path):
        if 'data.yaml' in files:
            dataset_info = analyze_dataset(root)
            if dataset_info:
                datasets.append(dataset_info)

    return datasets


def print_dataset_comparison(datasets):
    """
    Print comparison table of datasets
    """
    if not datasets:
        print("❌ No datasets found!")
        return

    print("="*80)
    print("📊 DATASET COMPARISON")
    print("="*80)
    print()

    for i, ds in enumerate(datasets, 1):
        print(f"Dataset #{i}: {ds['name']}")
        print(f"  📁 Path: {ds['path']}")
        print(f"  📊 Total Images: {ds['total_images']:,}")
        print(f"      • Train: {ds['train_images']:,}")
        print(f"      • Valid: {ds['val_images']:,}")
        print(f"      • Test: {ds['test_images']:,}")
        print(f"  🏷️  Classes ({ds['num_classes']}): {', '.join(ds['classes'])}")
        print(f"  📄 YAML: {ds['data_yaml']}")
        print()

    print("="*80)
    print("💡 RECOMMENDATIONS:")
    print("="*80)
    print()

    # Find best dataset
    best_dataset = max(datasets, key=lambda x: x['total_images'])

    print(f"✅ BEST CHOICE (Most images): {best_dataset['name']}")
    print(f"   {best_dataset['total_images']:,} images")
    print(f"   Classes: {', '.join(best_dataset['classes'])}")
    print()

    # Check for class compatibility
    all_classes = set()
    for ds in datasets:
        all_classes.update(ds['classes'])

    print(f"🔍 UNIQUE CLASSES ACROSS ALL DATASETS:")
    print(f"   {', '.join(sorted(all_classes))}")
    print()

    # Check if merging makes sense
    if len(datasets) > 1:
        print("🤔 SHOULD YOU MERGE?")
        print()

        # Check class overlap
        class_sets = [set(ds['classes']) for ds in datasets]
        common_classes = set.intersection(*class_sets) if class_sets else set()

        if len(common_classes) >= 3:
            print(f"   ✅ Common classes: {', '.join(common_classes)}")
            print(f"   ✅ Merging could work well!")
            print(f"   → Total merged images: {sum(ds['total_images'] for ds in datasets):,}")
        else:
            print(f"   ⚠️  Different class schemas detected")
            print(f"   ⚠️  Merging requires class remapping")
            print(f"   💡 Recommend: Use best single dataset first")


def main():
    """
    Main function to analyze datasets
    """
    print("="*80)
    print("🔍 ROBOFLOW DATASET ANALYZER")
    print("="*80)
    print()

    # Search current directory and common locations
    search_paths = [
        '.',
        os.path.expanduser('~/'),
        os.path.expanduser('~/Downloads'),
        '/home/user/Road_Sentinel',
    ]

    all_datasets = []

    print("Searching for datasets...")
    for path in search_paths:
        if os.path.exists(path):
            datasets = find_datasets(path)
            all_datasets.extend(datasets)

    # Remove duplicates
    seen = set()
    unique_datasets = []
    for ds in all_datasets:
        if ds['path'] not in seen:
            seen.add(ds['path'])
            unique_datasets.append(ds)

    print(f"Found {len(unique_datasets)} dataset(s)")
    print()

    if unique_datasets:
        print_dataset_comparison(unique_datasets)
    else:
        print("="*80)
        print("❌ NO DATASETS FOUND")
        print("="*80)
        print()
        print("Make sure you've downloaded datasets from Roboflow.")
        print()
        print("Datasets should contain a 'data.yaml' file with this structure:")
        print("  dataset_folder/")
        print("    ├── data.yaml")
        print("    ├── train/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    └── valid/")
        print("        ├── images/")
        print("        └── labels/")
        print()
        print("📍 Where to place datasets:")
        print("   • /home/user/Road_Sentinel/datasets/")
        print("   • Or any location - just note the path")


if __name__ == "__main__":
    main()
