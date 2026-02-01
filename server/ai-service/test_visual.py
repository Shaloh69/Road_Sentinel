#!/usr/bin/env python3
"""
Visual Real-Time Testing GUI for Road Sentinel AI Service
Shows live video with bounding boxes and detection results
"""

import cv2
import requests
import numpy as np
import sys
import time
from pathlib import Path
import argparse
from typing import Optional, Dict, List, Any

# AI Service URL
AI_SERVICE_URL = "http://localhost:8000"

# Colors for different vehicle types (BGR format for OpenCV)
VEHICLE_COLORS = {
    'car': (0, 255, 0),        # Green
    'truck': (255, 165, 0),    # Orange
    'bus': (0, 255, 255),      # Yellow
    'motorcycle': (255, 0, 255),  # Magenta
    'bicycle': (255, 255, 0),  # Cyan
    'unknown': (128, 128, 128) # Gray
}

INCIDENT_COLOR = (0, 0, 255)  # Red for incidents


class VisualTester:
    """Visual testing application with GUI"""

    def __init__(self, confidence: float = 0.5, show_fps: bool = True, process_every_n_frames: int = 3):
        self.confidence = confidence
        self.show_fps = show_fps
        self.paused = False
        self.process_every_n_frames = process_every_n_frames  # Only run AI every N frames for smooth playback
        self.current_frame = None
        self.current_annotated = None
        self.last_result = None  # Store last AI result to reuse

        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'vehicle_types': {},
            'incidents': 0,
            'avg_processing_time': 0,
            'processing_times': []
        }

    def check_ai_service(self) -> bool:
        """Check if AI service is running"""
        try:
            response = requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def detect_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Send frame to AI service for detection"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)

            # Send to AI service
            files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            data = {
                "camera_id": "VISUAL-TEST",
                "confidence_threshold": str(self.confidence)
            }

            start_time = time.time()
            response = requests.post(
                f"{AI_SERVICE_URL}/api/detect",
                files=files,
                data=data,
                timeout=10
            )
            processing_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                result['processing_time_ms'] = processing_time
                return result

            return None

        except Exception as e:
            print(f"Detection error: {e}")
            return None

    def draw_detections(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        if not result:
            return annotated

        detections = result.get('detections', [])
        incidents = result.get('incidents', [])

        # Draw vehicle detections
        for det in detections:
            vehicle_type = det['class']
            confidence = det['confidence']
            bbox = det['bbox']

            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

            # Get color for vehicle type
            color = VEHICLE_COLORS.get(vehicle_type, VEHICLE_COLORS['unknown'])

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            label = f"{vehicle_type} {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Ensure label stays within frame
            label_y = max(y - 10, label_size[1] + 10)

            cv2.rectangle(annotated,
                         (x, label_y - label_size[1] - 10),
                         (x + label_size[0], label_y),
                         color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Update statistics
            self.stats['vehicle_types'][vehicle_type] = \
                self.stats['vehicle_types'].get(vehicle_type, 0) + 1

        # Draw incidents at top
        if incidents:
            y_offset = 30
            for inc in incidents:
                text = f"⚠ {inc['type'].upper()} - {inc['severity']}"
                cv2.rectangle(annotated, (5, y_offset - 25), (width - 5, y_offset + 5),
                             INCIDENT_COLOR, -1)
                cv2.putText(annotated, text, (10, y_offset - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 40
                self.stats['incidents'] += 1

        # Update statistics
        self.stats['total_detections'] += len(detections)

        return annotated

    def draw_info_panel(self, frame: np.ndarray, fps: float, processing_time: float,
                       current_pos: int = 0, total_frames: int = 0) -> np.ndarray:
        """Draw information panel on frame"""
        height, width = frame.shape[:2]

        # Semi-transparent background for info panel
        overlay = frame.copy()
        panel_height = 240
        cv2.rectangle(overlay, (0, 0), (380, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw info text
        y_pos = 25
        line_height = 22

        # Calculate video progress
        progress_text = ""
        if total_frames > 0:
            progress_pct = (current_pos / total_frames) * 100
            progress_text = f"Progress: {current_pos}/{total_frames} ({progress_pct:.1f}%)"

        info_lines = [
            f"FPS: {fps:.1f}",
            f"Processing: {processing_time:.1f}ms",
            progress_text if progress_text else f"Frames: {self.stats['total_frames']}",
            f"Detections: {self.stats['total_detections']}",
            f"Incidents: {self.stats['incidents']}",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "Arrow Left/Right - Skip 1s",
            "PageUp/Down - Skip 10s",
            "Q - Quit"
        ]

        for line in info_lines:
            if line:  # Skip empty lines
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += line_height

        # Draw vehicle counts on the right
        if self.stats['vehicle_types']:
            x_pos = width - 200
            y_pos = 25
            cv2.rectangle(overlay, (x_pos - 10, 0), (width, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, "Vehicle Counts:", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += line_height

            for vtype, count in sorted(self.stats['vehicle_types'].items(),
                                      key=lambda x: x[1], reverse=True):
                color = VEHICLE_COLORS.get(vtype, VEHICLE_COLORS['unknown'])
                cv2.putText(frame, f"{vtype}: {count}", (x_pos, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += line_height

        return frame

    def process_video(self, video_path: str):
        """Process video file with visual display"""
        print(f"🎬 Opening video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame delay for playback timing
        frame_delay = int(1000 / video_fps) if video_fps > 0 else 30  # milliseconds

        print(f"📹 Video: {total_frames} frames, {video_fps:.2f} FPS, {width}x{height}")
        print(f"🤖 Starting AI detection (confidence: {self.confidence})")
        print()
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  Arrow Left/Right - Skip backward/forward 1 second")
        print("  PageUp/PageDown - Skip backward/forward 10 seconds")
        print("  Q - Quit")
        print("  +/- - Adjust confidence")
        print()

        window_name = "Road Sentinel - AI Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        last_time = time.time()
        display_fps = 0
        seek_request = 0  # Frames to seek (positive = forward, negative = backward)
        processing_time = 0

        print(f"⚡ Performance mode: Processing AI every {self.process_every_n_frames} frames for smooth playback")
        print()

        while True:
            # Handle seek requests
            if seek_request != 0:
                new_pos = frame_count + seek_request
                new_pos = max(0, min(new_pos, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                frame_count = new_pos
                seek_request = 0
                self.last_result = None  # Clear cached result after seek
                print(f"⏩ Seeking to frame {frame_count}/{total_frames}")

            if not self.paused:
                ret, frame = cap.read()

                if not ret:
                    print("\n✅ Video finished")
                    break

                frame_count += 1
                self.stats['total_frames'] += 1

                # Only run AI detection every N frames for smooth playback
                if frame_count % self.process_every_n_frames == 0:
                    result = self.detect_frame(frame)
                    if result:
                        self.last_result = result
                        self.stats['processed_frames'] += 1
                        processing_time = result.get('processing_time_ms', 0)
                else:
                    # Reuse last result for intermediate frames
                    result = self.last_result

                # Draw detections (using last result if not processing this frame)
                annotated = self.draw_detections(frame, result)

                # Calculate FPS
                current_time = time.time()
                time_diff = current_time - last_time
                if time_diff > 0:
                    display_fps = 1.0 / time_diff
                last_time = current_time

                # Draw info panel with progress
                annotated = self.draw_info_panel(annotated, display_fps, processing_time,
                                                 frame_count, total_frames)

                # Store current frame for pause display
                self.current_frame = frame
                self.current_annotated = annotated

                # Show frame
                cv2.imshow(window_name, annotated)
            else:
                # Display the last frame when paused
                if self.current_annotated is not None:
                    cv2.imshow(window_name, self.current_annotated)

            # Handle keyboard input with proper timing
            key = cv2.waitKey(frame_delay if not self.paused else 30) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                print("\n⏹️  Stopped by user")
                break
            elif key == ord(' '):  # Space
                self.paused = not self.paused
                print("⏸️  Paused" if self.paused else "▶️  Resumed")
            elif key == ord('+') or key == ord('='):
                self.confidence = min(1.0, self.confidence + 0.05)
                print(f"🎯 Confidence: {self.confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.confidence = max(0.1, self.confidence - 0.05)
                print(f"🎯 Confidence: {self.confidence:.2f}")
            elif key == 83:  # Right arrow
                seek_request = int(video_fps)  # Skip 1 second forward
            elif key == 81:  # Left arrow
                seek_request = -int(video_fps)  # Skip 1 second backward
            elif key == 85:  # Page Up
                seek_request = int(video_fps * 10)  # Skip 10 seconds forward
            elif key == 86:  # Page Down
                seek_request = -int(video_fps * 10)  # Skip 10 seconds backward

        cap.release()
        cv2.destroyAllWindows()

        # Print final statistics
        self.print_statistics()

    def process_camera(self, camera_id: int = 0):
        """Process webcam/camera feed with visual display"""
        print(f"📷 Opening camera {camera_id}...")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Camera: {width}x{height}")
        print(f"🤖 Starting AI detection (confidence: {self.confidence})")
        print(f"⚡ Performance mode: Processing AI every {self.process_every_n_frames} frames for smooth playback")
        print()
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  Q - Quit")
        print("  +/- - Adjust confidence")
        print()

        window_name = "Road Sentinel - Live Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_time = time.time()
        display_fps = 0
        frame_count = 0
        processing_time = 0

        while True:
            if not self.paused:
                ret, frame = cap.read()

                if not ret:
                    print("❌ Failed to read from camera")
                    break

                frame_count += 1
                self.stats['total_frames'] += 1

                # Only run AI detection every N frames for smooth playback
                if frame_count % self.process_every_n_frames == 0:
                    result = self.detect_frame(frame)
                    if result:
                        self.last_result = result
                        self.stats['processed_frames'] += 1
                        processing_time = result.get('processing_time_ms', 0)
                else:
                    # Reuse last result for intermediate frames
                    result = self.last_result

                # Draw detections
                annotated = self.draw_detections(frame, result)

                # Calculate FPS
                current_time = time.time()
                time_diff = current_time - last_time
                if time_diff > 0:
                    display_fps = 1.0 / time_diff
                last_time = current_time

                # Draw info panel
                annotated = self.draw_info_panel(annotated, display_fps, processing_time, 0, 0)

                # Store current frame for pause display
                self.current_frame = frame
                self.current_annotated = annotated

                # Show frame
                cv2.imshow(window_name, annotated)
            else:
                # Display the last frame when paused
                if self.current_annotated is not None:
                    cv2.imshow(window_name, self.current_annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("\n⏹️  Stopped by user")
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print("⏸️  Paused" if self.paused else "▶️  Resumed")
            elif key == ord('+') or key == ord('='):
                self.confidence = min(1.0, self.confidence + 0.05)
                print(f"🎯 Confidence: {self.confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.confidence = max(0.1, self.confidence - 0.05)
                print(f"🎯 Confidence: {self.confidence:.2f}")

        cap.release()
        cv2.destroyAllWindows()

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        """Print final statistics"""
        print("\n" + "=" * 70)
        print("📊 SESSION STATISTICS")
        print("=" * 70)
        print(f"Total frames displayed: {self.stats['total_frames']}")
        print(f"Frames processed with AI: {self.stats['processed_frames']}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"Total incidents: {self.stats['incidents']}")

        if self.stats['vehicle_types']:
            print("\n🚗 Vehicle Types:")
            for vtype, count in sorted(self.stats['vehicle_types'].items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"   {vtype}: {count}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Visual Real-Time Testing for Road Sentinel AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with video file
  python test_visual.py video.mp4

  # Use webcam
  python test_visual.py --camera

  # Use specific camera (0, 1, 2, etc.)
  python test_visual.py --camera 1

  # Adjust confidence threshold
  python test_visual.py video.mp4 --confidence 0.3

  # Process every frame (slower but more accurate)
  python test_visual.py video.mp4 --skip 1

  # Process every 5 frames (faster playback)
  python test_visual.py video.mp4 --skip 5
        """
    )

    parser.add_argument('video', nargs='?', help='Path to video file')
    parser.add_argument('--camera', type=int, nargs='?', const=0,
                       help='Use camera (default: 0 for webcam)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--skip', type=int, default=3,
                       help='Process AI every N frames for smooth playback (default: 3, use 1 for every frame)')

    args = parser.parse_args()

    # Check if AI service is running
    tester = VisualTester(confidence=args.confidence, process_every_n_frames=args.skip)

    print("=" * 70)
    print("🚦 Road Sentinel - Visual Testing")
    print("=" * 70)
    print()

    if not tester.check_ai_service():
        print("❌ AI service is not running!")
        print()
        print("Please start the AI service first:")
        print("   cd server/ai-service")
        print("   .\\venv\\Scripts\\Activate.ps1  # Windows")
        print("   python -m app.main")
        print()
        sys.exit(1)

    print("✅ AI service is running")
    print()

    # Determine mode
    if args.camera is not None:
        # Camera mode
        tester.process_camera(args.camera)
    elif args.video:
        # Video file mode
        tester.process_video(args.video)
    else:
        print("❌ Please specify a video file or use --camera for webcam")
        print()
        print("Examples:")
        print("   python test_visual.py video.mp4")
        print("   python test_visual.py --camera")
        sys.exit(1)


if __name__ == "__main__":
    main()
