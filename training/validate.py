#!/usr/bin/env python3
"""
Test trained vehicle detection model on videos/images
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse


def test_on_video(model_path, video_source, conf_threshold=0.7, save_output=True):
    """
    Test model on video file or webcam

    Args:
        model_path: Path to trained model weights
        video_source: Video file path, image folder, or webcam (0)
        conf_threshold: Confidence threshold (0.0-1.0)
        save_output: Whether to save annotated results
    """

    print("\n" + "="*70)
    print("🧪 TESTING VEHICLE DETECTION MODEL")
    print("="*70 + "\n")

    # Load model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"🎯 Testing on: {video_source}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   Save results: {save_output}\n")

    # Run prediction
    try:
        results = model.predict(
            source=video_source,
            conf=conf_threshold,      # Confidence threshold
            iou=0.45,                  # NMS IoU threshold
            show=False,                # Don't show live (use cv2.imshow instead)
            save=save_output,          # Save annotated results
            save_txt=True,             # Save detection coordinates
            save_conf=True,            # Save confidence scores
            stream=True,               # Stream mode for videos
            verbose=True               # Show detection info
        )

        # Process results frame by frame
        frame_count = 0
        total_detections = 0

        for result in results:
            frame_count += 1
            detections = len(result.boxes)
            total_detections += detections

            # Get detection details
            if detections > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if frame_count % 30 == 0:  # Print every 30 frames
                        print(f"Frame {frame_count}: Detected {class_name} (conf: {confidence:.2f})")

        print(f"\n✅ Testing complete!")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Total vehicles detected: {total_detections}")
        print(f"   Average detections per frame: {total_detections/frame_count:.2f}")
        print(f"\n📁 Results saved to: runs/detect/predict/\n")

        return results

    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_on_images(model_path, image_folder, conf_threshold=0.7):
    """
    Test model on folder of images
    """

    print("\n" + "="*70)
    print("🧪 TESTING ON IMAGE FOLDER")
    print("="*70 + "\n")

    model = YOLO(model_path)

    # Get all images
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

    print(f"📁 Found {len(image_files)} images")
    print(f"🎯 Processing with confidence threshold: {conf_threshold}\n")

    # Run prediction
    results = model.predict(
        source=str(image_folder),
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True
    )

    # Analyze results
    total_detections = 0
    detection_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

    for result in results:
        total_detections += len(result.boxes)
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            if class_name in detection_counts:
                detection_counts[class_name] += 1

    print(f"\n✅ Testing complete!")
    print(f"   Images processed: {len(image_files)}")
    print(f"   Total vehicles detected: {total_detections}")
    print(f"\n📊 Detections by class:")
    for class_name, count in detection_counts.items():
        print(f"   - {class_name}: {count}")
    print(f"\n📁 Results saved to: runs/detect/predict/\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test trained vehicle detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on video file
  python test_model.py --model runs/detect/runs/vehicle_speed/busay_v1/weights/best.pt --source test_video.mp4

  # Test on webcam
  python test_model.py --model runs/detect/runs/vehicle_speed/busay_v1/weights/best.pt --source 0

  # Test on image folder
  python test_model.py --model runs/detect/runs/vehicle_speed/busay_v1/weights/best.pt --source ../../datasets/processed/busay_vehicle_detection/valid/images

  # Test with higher confidence threshold
  python test_model.py --model runs/detect/runs/vehicle_speed/busay_v1/weights/best.pt --source test_video.mp4 --conf 0.8
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Video file, image folder, or webcam (0)')
    parser.add_argument('--conf', type=float, default=0.7,
                       help='Confidence threshold (default: 0.7)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')

    args = parser.parse_args()

    # Check if source is image folder
    if Path(args.source).is_dir():
        test_on_images(args.model, args.source, args.conf)
    else:
        test_on_video(args.model, args.source, args.conf, not args.no_save)


if __name__ == "__main__":
    main()
