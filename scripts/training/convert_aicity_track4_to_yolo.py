#!/usr/bin/env python3
"""
Convert AI City Challenge 2021 Track 4 (Traffic Anomaly Detection) to YOLO Format
For crash detection and anomaly detection
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

def convert_aicity_track4_to_yolo(
    aicity_root='datasets/aicity_2021_track4',
    output_root='datasets/aicity_2021_track4_yolo'
):
    """
    Convert AI City 2021 Track 4 dataset to YOLO format

    Dataset structure:
    aicity_2021_track4/
    ├── train/
    │   ├── videos/
    │   │   ├── video_1.mp4
    │   │   └── ...
    │   └── annotations/
    │       ├── video_1.json
    │       └── ...
    └── test/
    """

    print("="*70)
    print("🚨 AI CITY 2021 TRACK 4 → YOLO CONVERTER")
    print("   Traffic Anomaly Detection")
    print("="*70)
    print()

    # Check if dataset exists
    if not Path(aicity_root).exists():
        print("❌ ERROR: Dataset not found!")
        print(f"   Expected location: {aicity_root}")
        print()
        print("Please:")
        print("1. Register at https://www.aicitychallenge.org/")
        print("2. Download 2021 Track 4 dataset")
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
    train_stats = process_videos(
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
    print(f"🚨 Anomaly frames: {train_stats['anomaly_frames']}")
    print(f"✅ Normal frames: {train_stats['normal_frames']}")
    print()

    # Create YAML config
    create_yaml_config(output_root)

def process_videos(split_dir, output_images, output_labels, split='train'):
    """
    Process videos and extract frames with annotations

    Track 4 annotation format (JSON):
    {
      "video_id": "video_1",
      "frames": [
        {
          "frame_id": 1,
          "anomaly": true/false,
          "anomaly_type": "crash"/"stalled"/"accident",
          "objects": [
            {
              "bbox": [x, y, w, h],
              "class": "car"/"truck"/"bus",
              "involved_in_anomaly": true/false
            }
          ]
        }
      ]
    }
    """

    total_images = 0
    anomaly_frames = 0
    normal_frames = 0

    # Find videos and annotations
    video_dir = split_dir / 'videos'
    anno_dir = split_dir / 'annotations'

    if not video_dir.exists() or not anno_dir.exists():
        print(f"❌ Missing videos or annotations directory")
        return {
            'total_images': 0,
            'anomaly_frames': 0,
            'normal_frames': 0
        }

    # Process each video
    for video_file in tqdm(sorted(video_dir.glob('*.mp4')),
                          desc="  Processing videos"):
        video_name = video_file.stem
        anno_file = anno_dir / f"{video_name}.json"

        if not anno_file.exists():
            print(f"    ⚠️  No annotations for {video_name}, skipping...")
            continue

        # Load annotations
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        # Open video
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"    ⚠️  Cannot open {video_name}, skipping...")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Process frames
        for frame_anno in annotations.get('frames', []):
            frame_id = frame_anno['frame_id']
            is_anomaly = frame_anno.get('anomaly', False)
            anomaly_type = frame_anno.get('anomaly_type', 'normal')

            # Set video to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ret, frame = cap.read()

            if not ret:
                continue

            # Create unique filename
            output_name = f"{video_name}_frame{frame_id:06d}"

            # Save image
            output_img = output_images / f"{output_name}.jpg"
            cv2.imwrite(str(output_img), frame)

            # Create YOLO label
            output_label = output_labels / f"{output_name}.txt"

            with open(output_label, 'w') as f:
                # Write objects
                for obj in frame_anno.get('objects', []):
                    bbox = obj['bbox']  # [x, y, w, h]
                    involved = obj.get('involved_in_anomaly', False)

                    # Determine class
                    # 0 = normal vehicle
                    # 1 = vehicle involved in anomaly
                    yolo_class = 1 if involved else 0

                    # Convert to YOLO format (normalized)
                    x_center = (bbox[0] + bbox[2] / 2) / frame_width
                    y_center = (bbox[1] + bbox[3] / 2) / frame_height
                    width = bbox[2] / frame_width
                    height = bbox[3] / frame_height

                    # Clamp to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            total_images += 1
            if is_anomaly:
                anomaly_frames += 1
            else:
                normal_frames += 1

        cap.release()

    return {
        'total_images': total_images,
        'anomaly_frames': anomaly_frames,
        'normal_frames': normal_frames
    }

def create_yaml_config(output_root):
    """Create YOLO dataset configuration file"""

    yaml_content = f"""# AI City Challenge 2021 Track 4 - YOLO Format
# Traffic Anomaly Detection Dataset
# For crash detection and safety monitoring

# Dataset root
path: {os.path.abspath(output_root)}

# Train and validation sets
train: images/train
val: images/val

# Classes
names:
  0: vehicle_normal
  1: vehicle_in_anomaly

# Number of classes
nc: 2

# Dataset info
download: |
  # AI City Challenge 2021 Track 4
  # 1. Register at https://www.aicitychallenge.org/
  # 2. Download Track 4 dataset
  # 3. Extract to datasets/aicity_2021_track4/
  # 4. Run: python convert_aicity_track4_to_yolo.py
"""

    yaml_path = Path('aicity_2021_track4.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ Created YOLO config: {yaml_path}")

def main():
    """Main conversion function"""

    print("="*70)
    print("🚨 AI CITY 2021 TRACK 4 CONVERTER")
    print("   Traffic Anomaly Detection → YOLO Format")
    print("="*70)
    print()

    # Convert dataset
    convert_aicity_track4_to_yolo()

    print("="*70)
    print("✅ READY TO TRAIN!")
    print("="*70)
    print()
    print("🚀 Next steps:")
    print("   python train_vehicle_detector.py \\")
    print("     --data aicity_2021_track4.yaml \\")
    print("     --model n \\")
    print("     --batch 4 \\")
    print("     --epochs 100 \\")
    print("     --name aicity_anomaly_detection")
    print()

if __name__ == "__main__":
    main()
