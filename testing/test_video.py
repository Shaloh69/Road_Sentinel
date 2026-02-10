#!/usr/bin/env python3
"""
Video Testing Script for Road Sentinel AI Service
Tests the AI service with your own traffic videos
"""

import cv2
import requests
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import argparse

# AI Service URL
AI_SERVICE_URL = "http://localhost:8000"


def extract_frames(video_path: str, frame_rate: int = 5):
    """
    Extract frames from video at specified rate

    Args:
        video_path: Path to video file
        frame_rate: Extract 1 frame every N frames (default: 5 = ~6 FPS for 30 FPS video)

    Yields:
        Tuple of (frame_number, frame_image)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 Video Info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Processing every {frame_rate} frames (~{fps/frame_rate:.1f} FPS)")
    print()

    frame_count = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            yield frame_count, frame
            processed += 1

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {processed} frames from {total_frames} total frames")


def detect_frame(frame, camera_id: str = "TEST-VIDEO", confidence: float = 0.5) -> Dict[str, Any]:
    """
    Send frame to AI service for detection

    Args:
        frame: OpenCV frame (numpy array)
        camera_id: Camera identifier
        confidence: Detection confidence threshold

    Returns:
        Detection results dictionary
    """
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)

    # Send to AI service
    files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    data = {
        "camera_id": camera_id,
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
            return response.json()
        else:
            print(f"⚠️  Detection failed: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return None


def draw_detections(frame, detections: List[Dict], incidents: List[Dict]):
    """
    Draw bounding boxes and labels on frame

    Args:
        frame: OpenCV frame
        detections: List of vehicle detections
        incidents: List of incidents

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Draw vehicle detections in green
    for det in detections:
        bbox = det['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label
        label = f"{det['class']} {det['confidence']:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x, y - label_size[1] - 10),
                     (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw incidents in red
    if incidents:
        y_offset = 30
        for inc in incidents:
            text = f"⚠️  {inc['type'].upper()} - {inc['severity']}"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

    return annotated


def test_video(video_path: str, frame_rate: int = 5, confidence: float = 0.5,
               save_output: bool = False, show_frames: bool = False):
    """
    Test AI service with video file

    Args:
        video_path: Path to video file
        frame_rate: Process every Nth frame
        confidence: Detection confidence threshold
        save_output: Save annotated frames to output folder
        show_frames: Display frames in real-time (requires GUI)
    """
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return

    print("=" * 70)
    print(f"🎬 Testing Video: {video_path.name}")
    print("=" * 70)
    print()

    # Check if AI service is running
    try:
        response = requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ AI service is not responding. Please start it first:")
            print("   cd server/ai-service")
            print("   python -m app.main")
            return
        print("✅ AI service is running\n")
    except:
        print("❌ Cannot connect to AI service. Please start it first:")
        print("   cd server/ai-service")
        print("   python -m app.main")
        return

    # Create output directory if saving
    if save_output:
        output_dir = Path("output") / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving output to: {output_dir}\n")

    # Statistics
    stats = {
        'total_frames': 0,
        'vehicles_detected': 0,
        'incidents_detected': 0,
        'vehicle_types': {},
        'incident_types': {},
        'processing_times': []
    }

    # Process video
    print("🔄 Processing video...\n")

    for frame_num, frame in extract_frames(str(video_path), frame_rate):
        stats['total_frames'] += 1

        # Detect
        start_time = time.time()
        result = detect_frame(frame, confidence=confidence)
        processing_time = (time.time() - start_time) * 1000

        if result:
            stats['processing_times'].append(processing_time)

            detections = result.get('detections', [])
            incidents = result.get('incidents', [])

            # Update statistics
            stats['vehicles_detected'] += len(detections)
            stats['incidents_detected'] += len(incidents)

            for det in detections:
                vtype = det['class']
                stats['vehicle_types'][vtype] = stats['vehicle_types'].get(vtype, 0) + 1

            for inc in incidents:
                itype = inc['type']
                stats['incident_types'][itype] = stats['incident_types'].get(itype, 0) + 1

            # Print frame results
            print(f"Frame {frame_num:5d}: {len(detections)} vehicles, "
                  f"{len(incidents)} incidents ({processing_time:.1f}ms)")

            # Draw annotations
            if save_output or show_frames:
                annotated = draw_detections(frame, detections, incidents)

                if save_output:
                    output_path = output_dir / f"frame_{frame_num:05d}.jpg"
                    cv2.imwrite(str(output_path), annotated)

                if show_frames:
                    cv2.imshow('Detection Results', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Stopped by user")
                        break

    if show_frames:
        cv2.destroyAllWindows()

    # Print summary
    print("\n" + "=" * 70)
    print("📊 DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Total vehicles detected: {stats['vehicles_detected']}")
    print(f"Total incidents detected: {stats['incidents_detected']}")

    if stats['processing_times']:
        avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
        print(f"Average processing time: {avg_time:.2f}ms per frame")

    print("\n🚗 Vehicle Types:")
    for vtype, count in sorted(stats['vehicle_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {vtype}: {count}")

    if stats['incident_types']:
        print("\n⚠️  Incident Types:")
        for itype, count in sorted(stats['incident_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {itype}: {count}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Test Road Sentinel AI with your own traffic videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python test_video.py /path/to/traffic_video.mp4

  # Process every 10 frames (faster)
  python test_video.py video.mp4 --frame-rate 10

  # Save annotated frames
  python test_video.py video.mp4 --save

  # Show real-time detection (requires display)
  python test_video.py video.mp4 --show

  # Lower confidence threshold (more detections)
  python test_video.py video.mp4 --confidence 0.3
        """
    )

    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--frame-rate', type=int, default=5,
                       help='Process every Nth frame (default: 5)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--save', action='store_true',
                       help='Save annotated frames to output folder')
    parser.add_argument('--show', action='store_true',
                       help='Display frames in real-time (press Q to quit)')

    args = parser.parse_args()

    test_video(
        args.video,
        frame_rate=args.frame_rate,
        confidence=args.confidence,
        save_output=args.save,
        show_frames=args.show
    )


if __name__ == "__main__":
    main()
