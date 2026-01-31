#!/usr/bin/env python3
"""
Image Testing Script for Road Sentinel AI Service
Tests the AI service with your own traffic images
"""

import cv2
import requests
import sys
from pathlib import Path
import argparse


AI_SERVICE_URL = "http://localhost:8000"


def test_image(image_path: str, confidence: float = 0.5, save_output: bool = False):
    """
    Test AI service with a single image

    Args:
        image_path: Path to image file
        confidence: Detection confidence threshold
        save_output: Save annotated image
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"🖼️  Testing image: {image_path.name}")

    # Check AI service
    try:
        requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
    except:
        print("❌ AI service not running. Start it with:")
        print("   cd server/ai-service && python -m app.main")
        return

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', image)

    # Send to AI service
    files = {"image": ("test.jpg", buffer.tobytes(), "image/jpeg")}
    data = {
        "camera_id": "TEST-IMAGE",
        "confidence_threshold": str(confidence)
    }

    try:
        response = requests.post(
            f"{AI_SERVICE_URL}/api/detect",
            files=files,
            data=data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            print("\n✅ Detection successful!")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")

            detections = result.get('detections', [])
            incidents = result.get('incidents', [])

            # Show detections
            if detections:
                print(f"\n🚗 Vehicles detected: {len(detections)}")
                for i, det in enumerate(detections, 1):
                    bbox = det['bbox']
                    print(f"   {i}. {det['class']} (confidence: {det['confidence']:.2f})")
                    print(f"      bbox: x={bbox['x']}, y={bbox['y']}, "
                          f"w={bbox['width']}, h={bbox['height']}")
            else:
                print("\n   No vehicles detected")

            # Show incidents
            if incidents:
                print(f"\n⚠️  Incidents detected: {len(incidents)}")
                for i, inc in enumerate(incidents, 1):
                    print(f"   {i}. {inc['type']} - {inc['severity']} severity")
                    print(f"      {inc['description']}")
            else:
                print("   No incidents detected")

            # Save annotated image
            if save_output:
                annotated = image.copy()

                # Draw detections
                for det in detections:
                    bbox = det['bbox']
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    label = f"{det['class']} {det['confidence']:.2f}"
                    cv2.putText(annotated, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                output_path = Path("output") / f"annotated_{image_path.name}"
                output_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(output_path), annotated)
                print(f"\n💾 Saved annotated image: {output_path}")

        else:
            print(f"❌ Detection failed: HTTP {response.status_code}")
            print(f"   {response.text}")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_folder(folder_path: str, confidence: float = 0.5):
    """
    Test AI service with all images in a folder

    Args:
        folder_path: Path to folder containing images
        confidence: Detection confidence threshold
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        print(f"❌ Not a directory: {folder_path}")
        return

    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"❌ No images found in: {folder_path}")
        return

    print(f"📁 Found {len(image_files)} images in {folder_path}")
    print()

    # Test each image
    for i, img_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")
        test_image(str(img_path), confidence=confidence)


def main():
    parser = argparse.ArgumentParser(
        description='Test Road Sentinel AI with images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python test_images.py /path/to/image.jpg

  # Test with lower confidence
  python test_images.py image.jpg --confidence 0.3

  # Save annotated image
  python test_images.py image.jpg --save

  # Test all images in folder
  python test_images.py /path/to/images/ --folder
        """
    )

    parser.add_argument('path', help='Path to image file or folder')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--save', action='store_true',
                       help='Save annotated image')
    parser.add_argument('--folder', action='store_true',
                       help='Process all images in folder')

    args = parser.parse_args()

    if args.folder:
        test_folder(args.path, confidence=args.confidence)
    else:
        test_image(args.path, confidence=args.confidence, save_output=args.save)


if __name__ == "__main__":
    main()
