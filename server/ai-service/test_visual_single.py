#!/usr/bin/env python3
"""
Single-threaded visual tester for ZERO LAG detection
Processes each frame synchronously for immediate, accurate bounding boxes
Best for testing with single video when you need perfect detection accuracy
"""

import cv2
import requests
import numpy as np
import sys
import time
from pathlib import Path
import argparse
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort

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
LINE_COLOR = (0, 255, 255)     # Cyan for counting lines
ID_COLOR = (255, 255, 255)     # White for IDs


class VehicleState:
    """Track individual vehicle state across frames"""
    def __init__(self, track_id: int, vehicle_type: str, first_seen_frame: int):
        self.track_id = track_id
        self.vehicle_type = vehicle_type
        self.first_seen_frame = first_seen_frame
        self.last_seen_frame = first_seen_frame
        self.crossed_entry = False
        self.entry_frame = None
        self.bbox_history = deque(maxlen=30)

    def update_position(self, bbox: Tuple[int, int, int, int], frame_num: int):
        self.last_seen_frame = frame_num
        self.bbox_history.append(bbox)

    def get_center(self) -> Optional[Tuple[int, int]]:
        if not self.bbox_history:
            return None
        x1, y1, x2, y2 = self.bbox_history[-1]
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def is_active(self, current_frame: int, timeout_frames: int = 30) -> bool:
        return (current_frame - self.last_seen_frame) < timeout_frames

    def is_timed_out(self, current_frame: int, timeout_frames: int) -> bool:
        if not self.crossed_entry or not self.entry_frame:
            return False
        return (current_frame - self.entry_frame) >= timeout_frames


class SingleThreadTester:
    """Single-threaded visual tester - ZERO LAG, perfect detection sync"""

    def __init__(self, confidence: float = 0.5, entry_line_y: float = 0.3,
                 timeout_seconds: float = 120):
        self.confidence = confidence
        self.paused = False
        self.entry_line_y = entry_line_y
        self.timeout_seconds = timeout_seconds
        self.timeout_frames = 0

        # DeepSORT tracker with GPU optimization
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            embedder="mobilenet",
            half=True,  # FP16 for speed
            embedder_gpu=True
        )

        # Vehicle tracking state
        self.vehicles = {}
        self.counted_in = set()
        self.timed_out_vehicles = set()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'unique_vehicles': 0,
            'vehicles_entered': 0,
            'vehicles_timed_out': 0,
            'active_vehicles': 0,
            'vehicle_types': defaultdict(int),
            'avg_processing_time': 0,
            'processing_times': []
        }

        self.frame_width = 0
        self.frame_height = 0
        self.video_fps = 0

    def check_ai_service(self) -> bool:
        try:
            response = requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def detect_frame(self, frame: np.ndarray, frame_num: int) -> Optional[Dict[str, Any]]:
        try:
            # Optimized JPEG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)

            files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            data = {
                "camera_id": f"SINGLE-TEST-{frame_num}",
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
                self.stats['processing_times'].append(processing_time)
                return result

            return None

        except Exception as e:
            print(f"Detection error: {e}")
            return None

    def update_tracking(self, detections: List[Dict], frame: np.ndarray, frame_num: int):
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert detections to DeepSORT format
        raw_detections = []
        for det in detections:
            bbox = det['bbox']
            x1, y1, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            raw_detections.append(([x1, y1, w, h], det['confidence'], det['class']))

        # Update tracker
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        # Process tracks
        tracked_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            det_class = track.get_det_class()
            det_conf = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.0

            tracked_det = {
                'track_id': track_id,
                'class': det_class if det_class else 'unknown',
                'confidence': det_conf,
                'bbox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1}
            }

            tracked_detections.append(tracked_det)

            # Update vehicle state
            if track_id not in self.vehicles:
                self.vehicles[track_id] = VehicleState(track_id, tracked_det['class'], frame_num)
                self.stats['unique_vehicles'] += 1
                self.stats['vehicle_types'][tracked_det['class']] += 1

            vehicle = self.vehicles[track_id]
            vehicle.update_position((x1, y1, x2, y2), frame_num)

            # Check line crossing
            self.check_line_crossing(vehicle, frame_num)

        # Update active vehicles count
        self.stats['active_vehicles'] = sum(
            1 for v in self.vehicles.values() if v.is_active(frame_num)
        )

        return tracked_detections

    def check_line_crossing(self, vehicle: VehicleState, frame_num: int):
        center = vehicle.get_center()
        if not center:
            return

        cx, cy = center
        entry_line_px = int(self.frame_height * self.entry_line_y)

        # Check entry line crossing
        if not vehicle.crossed_entry and cy > entry_line_px:
            if len(vehicle.bbox_history) > 1:
                prev_center_y = int((vehicle.bbox_history[-2][1] + vehicle.bbox_history[-2][3]) / 2)
                if prev_center_y <= entry_line_px:
                    vehicle.crossed_entry = True
                    vehicle.entry_frame = frame_num
                    if vehicle.track_id not in self.counted_in:
                        self.counted_in.add(vehicle.track_id)
                        self.stats['vehicles_entered'] += 1
                        print(f"✅ Vehicle #{vehicle.track_id} ({vehicle.vehicle_type}) ENTERED")

        # Check for timeout
        if vehicle.is_timed_out(frame_num, self.timeout_frames):
            if vehicle.track_id not in self.timed_out_vehicles:
                self.timed_out_vehicles.add(vehicle.track_id)
                self.stats['vehicles_timed_out'] += 1
                time_elapsed = (frame_num - vehicle.entry_frame) / self.video_fps
                print(f"⏱️  Vehicle #{vehicle.track_id} ({vehicle.vehicle_type}) TIMED OUT after {time_elapsed:.1f}s")

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = frame.copy()

        if not detections:
            return annotated

        for det in detections:
            track_id = det.get('track_id', None)
            vehicle_type = det['class']
            confidence = det['confidence']
            bbox = det['bbox']

            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

            color = VEHICLE_COLORS.get(vehicle_type, VEHICLE_COLORS['unknown'])

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw unique ID and label
            if track_id is not None and track_id != -1:
                label = f"ID:{track_id} {vehicle_type} {confidence:.2f}"
            else:
                label = f"{vehicle_type} {confidence:.2f}"

            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y - 10, label_size[1] + 10)

            cv2.rectangle(annotated, (x, label_y - label_size[1] - 10),
                         (x + label_size[0], label_y), color, -1)

            cv2.putText(annotated, label, (x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated

    def draw_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        entry_y = int(height * self.entry_line_y)
        cv2.line(frame, (0, entry_y), (width, entry_y), LINE_COLOR, 3)
        cv2.putText(frame, "ENTRY LINE", (10, entry_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINE_COLOR, 2)
        return frame

    def draw_info_panel(self, frame: np.ndarray, fps: float, processing_time: float,
                       current_pos: int = 0, total_frames: int = 0) -> np.ndarray:
        height, width = frame.shape[:2]

        overlay = frame.copy()
        panel_height = 300
        cv2.rectangle(overlay, (0, 0), (450, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_pos = 25
        line_height = 22

        progress_text = ""
        if total_frames > 0:
            progress_pct = (current_pos / total_frames) * 100
            progress_text = f"Progress: {current_pos}/{total_frames} ({progress_pct:.1f}%)"

        timeout_mins = self.timeout_seconds / 60

        info_lines = [
            f"Display FPS: {fps:.1f}",
            f"AI Processing: {processing_time:.1f}ms",
            "Mode: SINGLE-THREAD (ZERO LAG)",
            progress_text if progress_text else f"Frames: {self.stats['total_frames']}",
            "",
            f"🚗 Unique Vehicles: {self.stats['unique_vehicles']}",
            f"📥 Vehicles ENTERED: {self.stats['vehicles_entered']}",
            f"🔄 Active Now: {self.stats['active_vehicles']}",
            f"⏱️  Timed Out ({timeout_mins:.0f}m): {self.stats['vehicles_timed_out']}",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "Q - Quit"
        ]

        for line in info_lines:
            if line:
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += line_height

        # Draw vehicle type counts on the right
        if self.stats['vehicle_types']:
            x_pos = width - 200
            y_pos = 25
            cv2.rectangle(overlay, (x_pos - 10, 0), (width, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, "Vehicle Types:", (x_pos, y_pos),
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
        print(f"🎬 Opening video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.timeout_frames = int(self.timeout_seconds * self.video_fps)

        print(f"📹 Video: {total_frames} frames, {self.video_fps:.2f} FPS, {self.frame_width}x{self.frame_height}")
        print(f"🤖 Single-threaded mode - ZERO LAG detection")
        print(f"🎯 DeepSORT tracking enabled")
        print(f"📏 Entry line: {self.entry_line_y*100:.0f}%")
        print(f"⏱️  Vehicle timeout: {self.timeout_seconds/60:.0f} minutes")
        print()

        window_name = "Road Sentinel - ZERO LAG Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        last_time = time.time()
        display_fps = 0

        while True:
            if not self.paused:
                ret, frame = cap.read()

                if not ret:
                    print("\n✅ Video finished")
                    break

                frame_count += 1
                self.stats['total_frames'] += 1
                self.stats['processed_frames'] += 1

                # Process THIS frame with AI (synchronous - no lag!)
                result = self.detect_frame(frame, frame_count)

                if result:
                    detections = result.get('detections', [])
                    processing_time = result.get('processing_time_ms', 0)

                    # Update tracking with detections
                    tracked_detections = self.update_tracking(detections, frame, frame_count)

                    # Draw detections (perfectly synced!)
                    annotated = self.draw_detections(frame, tracked_detections)
                else:
                    annotated = frame.copy()
                    processing_time = 0

                # Draw counting lines
                annotated = self.draw_counting_lines(annotated)

                # Calculate FPS
                current_time = time.time()
                time_diff = current_time - last_time
                if time_diff > 0:
                    display_fps = 1.0 / time_diff
                last_time = current_time

                # Draw info panel
                annotated = self.draw_info_panel(annotated, display_fps, processing_time,
                                                 frame_count, total_frames)

                # Show frame
                cv2.imshow(window_name, annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("\n⏹️  Stopped by user")
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print("⏸️  Paused" if self.paused else "▶️  Resumed")

        cap.release()
        cv2.destroyAllWindows()

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        print("\n" + "=" * 70)
        print("📊 SESSION STATISTICS")
        print("=" * 70)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"\n🚗 VEHICLE TRACKING:")
        print(f"   Unique vehicles seen: {self.stats['unique_vehicles']}")
        print(f"   Vehicles ENTERED: {self.stats['vehicles_entered']}")
        print(f"   Vehicles timed out ({self.timeout_seconds/60:.0f} min): {self.stats['vehicles_timed_out']}")
        still_active = self.stats['vehicles_entered'] - self.stats['vehicles_timed_out']
        print(f"   Still being tracked: {still_active}")

        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            print(f"\n⚡ Performance:")
            print(f"   Avg AI processing: {avg_time:.1f}ms per frame")

        if self.stats['vehicle_types']:
            print("\n🚙 Vehicle Types:")
            for vtype, count in sorted(self.stats['vehicle_types'].items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"   {vtype}: {count}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Single-Threaded Visual Testing - ZERO LAG',
        epilog="""
Examples:
  # Test with video (default settings)
  python test_visual_single.py video.mp4

  # Adjust confidence threshold
  python test_visual_single.py video.mp4 --confidence 0.3

  # Custom entry line and timeout
  python test_visual_single.py video.mp4 --entry 0.2 --timeout 180
        """
    )

    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--entry', type=float, default=0.3,
                       help='Entry line position (0.0-1.0, default: 0.3)')
    parser.add_argument('--timeout', type=float, default=120,
                       help='Vehicle timeout in seconds (default: 120)')

    args = parser.parse_args()

    tester = SingleThreadTester(
        confidence=args.confidence,
        entry_line_y=args.entry,
        timeout_seconds=args.timeout
    )

    print("=" * 70)
    print("🚦 Road Sentinel - ZERO LAG Visual Testing")
    print("=" * 70)
    print()

    if not tester.check_ai_service():
        print("❌ AI service is not running!")
        print()
        print("Please start the AI service first:")
        print("   python -m app.main")
        print()
        sys.exit(1)

    print("✅ AI service is running")
    print()

    tester.process_video(args.video)


if __name__ == "__main__":
    main()
