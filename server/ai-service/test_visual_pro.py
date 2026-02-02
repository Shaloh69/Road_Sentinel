#!/usr/bin/env python3
"""
Professional Visual Testing with Multi-Threading, DeepSORT Tracking, and Bidirectional Counting
Features:
- Multi-threaded AI processing (30+ FPS)
- DeepSORT vehicle tracking with unique IDs
- Bidirectional counting (IN/OUT)
- Entry/Exit line crossing detection
- GPU optimizations
"""

import cv2
import requests
import numpy as np
import sys
import time
from pathlib import Path
import argparse
from typing import Optional, Dict, List, Any, Tuple
import threading
import queue
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
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
        self.bbox_history = deque(maxlen=30)  # Keep last 30 positions

    def update_position(self, bbox: Tuple[int, int, int, int], frame_num: int):
        """Update vehicle position"""
        self.last_seen_frame = frame_num
        self.bbox_history.append(bbox)

    def get_center(self) -> Optional[Tuple[int, int]]:
        """Get center point of most recent bounding box"""
        if not self.bbox_history:
            return None
        x1, y1, x2, y2 = self.bbox_history[-1]
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def is_active(self, current_frame: int, timeout_frames: int = 30) -> bool:
        """Check if vehicle is still active (seen recently)"""
        return (current_frame - self.last_seen_frame) < timeout_frames

    def is_timed_out(self, current_frame: int, timeout_frames: int) -> bool:
        """Check if vehicle has been in frame too long without exiting (2 min timeout)"""
        if not self.crossed_entry or not self.entry_frame:
            return False
        return (current_frame - self.entry_frame) >= timeout_frames


class ProVisualTester:
    """Professional visual testing with multi-threading and tracking"""

    def __init__(self, confidence: float = 0.5, num_workers: int = 3,
                 entry_line_y: float = 0.3, timeout_seconds: float = 120):
        self.confidence = confidence
        self.num_workers = num_workers
        self.paused = False

        # Counting line (as fraction of frame height)
        self.entry_line_y = entry_line_y  # 30% from top
        self.timeout_seconds = timeout_seconds  # 2 minutes default
        self.timeout_frames = 0  # Will be calculated based on video FPS

        # DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,  # Keep tracks for 30 frames without detection
            n_init=3,    # Confirm track after 3 consecutive detections
            max_iou_distance=0.7,
            embedder="mobilenet",  # Feature extractor
            half=True,  # Use FP16 for speed
            embedder_gpu=True
        )

        # Multi-threading
        self.frame_queue = queue.Queue(maxsize=num_workers * 2)
        self.result_queue = queue.Queue(maxsize=30)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.stop_processing = threading.Event()
        self.workers = []

        # Vehicle tracking state
        self.vehicles = {}  # track_id -> VehicleState
        self.counted_in = set()  # IDs that crossed entry line
        self.timed_out_vehicles = set()  # IDs that timed out

        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'unique_vehicles': 0,
            'vehicles_entered': 0,
            'vehicles_timed_out': 0,
            'active_vehicles': 0,
            'vehicle_types': defaultdict(int),
            'incidents': 0,
            'avg_processing_time': 0,
            'processing_times': []
        }

        self.current_frame = None
        self.current_annotated = None
        self.last_detections = []
        self.frame_width = 0
        self.frame_height = 0

    def check_ai_service(self) -> bool:
        """Check if AI service is running"""
        try:
            response = requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def detect_frame(self, frame: np.ndarray, frame_num: int) -> Optional[Dict[str, Any]]:
        """Send frame to AI service for detection"""
        try:
            # Encode frame as JPEG with quality optimization
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)

            # Send to AI service
            files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            data = {
                "camera_id": f"VISUAL-TEST-{frame_num}",
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
                result['frame_num'] = frame_num
                return result

            return None

        except Exception as e:
            print(f"Detection error: {e}")
            return None

    def _processing_worker(self):
        """Background worker thread for AI processing"""
        while not self.stop_processing.is_set():
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=0.1)
                frame, frame_num = frame_data

                # Process with AI
                result = self.detect_frame(frame, frame_num)

                # Put result in result queue
                if result:
                    try:
                        self.result_queue.put(result, timeout=0.1)
                        self.stats['processed_frames'] += 1
                    except queue.Full:
                        # If result queue is full, discard oldest result
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put(result, timeout=0.1)
                        except:
                            pass

                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def start_async_processing(self):
        """Start multiple background processing threads"""
        self.stop_processing.clear()
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._processing_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        print(f"🚀 Started {self.num_workers} AI processing workers")

    def stop_async_processing(self):
        """Stop all background processing threads"""
        self.stop_processing.set()
        for worker in self.workers:
            worker.join(timeout=2)
        self.workers.clear()

    def update_tracking(self, detections: List[Dict], frame: np.ndarray, frame_num: int):
        """Update DeepSORT tracking with new detections"""
        if not detections:
            # Update tracker with no detections
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert detections to DeepSORT format: [bbox, confidence, class]
        raw_detections = []
        for det in detections:
            bbox = det['bbox']
            # DeepSORT expects [x1, y1, w, h]
            x1 = bbox['x']
            y1 = bbox['y']
            w = bbox['width']
            h = bbox['height']

            raw_detections.append((
                [x1, y1, w, h],
                det['confidence'],
                det['class']
            ))

        # Update tracker
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        # Process tracks
        tracked_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)  # Ensure track_id is integer
            ltrb = track.to_ltrb()  # Get bbox as [left, top, right, bottom]

            # Convert to our format
            x1, y1, x2, y2 = map(int, ltrb)

            # Get the detection that matches this track
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
                self.vehicles[track_id] = VehicleState(
                    track_id,
                    tracked_det['class'],
                    frame_num
                )
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
        """Check if vehicle crossed entry line"""
        center = vehicle.get_center()
        if not center:
            return

        cx, cy = center
        entry_line_px = int(self.frame_height * self.entry_line_y)

        # Check entry line crossing (entering from top)
        if not vehicle.crossed_entry and cy > entry_line_px:
            if len(vehicle.bbox_history) > 1:
                # Check if previously above line
                prev_center_y = int((vehicle.bbox_history[-2][1] + vehicle.bbox_history[-2][3]) / 2)
                if prev_center_y <= entry_line_px:
                    vehicle.crossed_entry = True
                    vehicle.entry_frame = frame_num
                    if vehicle.track_id not in self.counted_in:
                        self.counted_in.add(vehicle.track_id)
                        self.stats['vehicles_entered'] += 1
                        print(f"✅ Vehicle #{vehicle.track_id} ({vehicle.vehicle_type}) ENTERED")

        # Check for timeout (2 minutes after entry)
        if vehicle.is_timed_out(frame_num, self.timeout_frames):
            if vehicle.track_id not in self.timed_out_vehicles:
                self.timed_out_vehicles.add(vehicle.track_id)
                self.stats['vehicles_timed_out'] += 1
                time_elapsed = (frame_num - vehicle.entry_frame) / self.video_fps if hasattr(self, 'video_fps') else 0
                print(f"⏱️  Vehicle #{vehicle.track_id} ({vehicle.vehicle_type}) TIMED OUT after {time_elapsed:.1f}s")

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes with unique IDs and labels on frame"""
        annotated = frame.copy()

        if not detections:
            return annotated

        for det in detections:
            track_id = det.get('track_id', None)
            vehicle_type = det['class']
            confidence = det['confidence']
            bbox = det['bbox']

            x = bbox['x']
            y = bbox['y']
            w = bbox['width']
            h = bbox['height']

            # Get color for vehicle type
            color = VEHICLE_COLORS.get(vehicle_type, VEHICLE_COLORS['unknown'])

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw unique ID and label
            if track_id is not None and track_id != -1:
                label = f"ID:{track_id} {vehicle_type} {confidence:.2f}"
            else:
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

        return annotated

    def draw_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw entry counting line"""
        height, width = frame.shape[:2]

        # Entry line
        entry_y = int(height * self.entry_line_y)
        cv2.line(frame, (0, entry_y), (width, entry_y), LINE_COLOR, 3)
        cv2.putText(frame, "ENTRY LINE", (10, entry_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINE_COLOR, 2)

        return frame

    def draw_info_panel(self, frame: np.ndarray, fps: float, processing_time: float,
                       current_pos: int = 0, total_frames: int = 0) -> np.ndarray:
        """Draw information panel on frame"""
        height, width = frame.shape[:2]

        # Semi-transparent background for info panel
        overlay = frame.copy()
        panel_height = 300
        cv2.rectangle(overlay, (0, 0), (450, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw info text
        y_pos = 25
        line_height = 22

        # Calculate video progress
        progress_text = ""
        if total_frames > 0:
            progress_pct = (current_pos / total_frames) * 100
            progress_text = f"Progress: {current_pos}/{total_frames} ({progress_pct:.1f}%)"

        timeout_mins = self.timeout_seconds / 60

        info_lines = [
            f"Display FPS: {fps:.1f}",
            f"AI Processing: {processing_time:.1f}ms",
            f"Workers: {self.num_workers}",
            progress_text if progress_text else f"Frames: {self.stats['total_frames']}",
            "",
            f"🚗 Unique Vehicles: {self.stats['unique_vehicles']}",
            f"📥 Vehicles ENTERED: {self.stats['vehicles_entered']}",
            f"🔄 Active Now: {self.stats['active_vehicles']}",
            f"⏱️  Timed Out ({timeout_mins:.0f}m): {self.stats['vehicles_timed_out']}",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "Arrow Left/Right - Skip 1s",
            "Q - Quit"
        ]

        for line in info_lines:
            if line:  # Skip empty lines
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
        """Process video file with visual display"""
        print(f"🎬 Opening video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = video_fps  # Store for timeout calculations
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate timeout in frames (e.g., 2 minutes = 120 seconds * FPS)
        self.timeout_frames = int(self.timeout_seconds * video_fps)

        # Calculate frame delay for playback timing
        frame_delay = int(1000 / video_fps) if video_fps > 0 else 30  # milliseconds

        print(f"📹 Video: {total_frames} frames, {video_fps:.2f} FPS, {self.frame_width}x{self.frame_height}")
        print(f"🤖 Starting AI detection (confidence: {self.confidence})")
        print(f"🧵 Multi-threaded with {self.num_workers} workers")
        print(f"🎯 DeepSORT tracking enabled")
        print(f"📏 Entry line: {self.entry_line_y*100:.0f}%")
        print(f"⏱️  Vehicle timeout: {self.timeout_seconds/60:.0f} minutes ({self.timeout_frames} frames)")
        print()

        window_name = "Road Sentinel - Professional AI Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        last_time = time.time()
        display_fps = 0
        seek_request = 0
        processing_time = 0

        # Start background AI processing threads
        self.start_async_processing()

        try:
            while True:
                # Handle seek requests
                if seek_request != 0:
                    new_pos = frame_count + seek_request
                    new_pos = max(0, min(new_pos, total_frames - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_count = new_pos
                    seek_request = 0
                    # Clear queues after seek
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            break
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except:
                            break
                    self.last_detections = []
                    print(f"⏩ Seeking to frame {frame_count}/{total_frames}")

                # Check for new AI results (non-blocking)
                try:
                    result = self.result_queue.get_nowait()
                    if result:
                        detections = result.get('detections', [])
                        processing_time = result.get('processing_time_ms', 0)

                        # Update tracking with new detections
                        if self.current_frame is not None:
                            self.last_detections = self.update_tracking(
                                detections,
                                self.current_frame,
                                frame_count
                            )
                except queue.Empty:
                    pass

                if not self.paused:
                    ret, frame = cap.read()

                    if not ret:
                        print("\n✅ Video finished")
                        break

                    frame_count += 1
                    self.stats['total_frames'] += 1
                    self.current_frame = frame.copy()

                    # Send frame to processing queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait((frame.copy(), frame_count))
                    except queue.Full:
                        # Queue full, skip this frame
                        pass

                    # Draw detections using last tracked results
                    annotated = self.draw_detections(frame, self.last_detections)

                    # Draw counting lines
                    annotated = self.draw_counting_lines(annotated)

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

        finally:
            # Stop background processing threads
            self.stop_async_processing()
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
        print(f"\n🚗 VEHICLE TRACKING:")
        print(f"   Unique vehicles seen: {self.stats['unique_vehicles']}")
        print(f"   Vehicles ENTERED: {self.stats['vehicles_entered']}")
        print(f"   Vehicles timed out ({self.timeout_seconds/60:.0f} min): {self.stats['vehicles_timed_out']}")
        still_active = self.stats['vehicles_entered'] - self.stats['vehicles_timed_out']
        print(f"   Still being tracked: {still_active}")

        if self.stats['vehicle_types']:
            print("\n🚙 Vehicle Types:")
            for vtype, count in sorted(self.stats['vehicle_types'].items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"   {vtype}: {count}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Professional Visual Testing with Multi-Threading and Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with video file (default 3 workers, 2 min timeout)
  python test_visual_pro.py video.mp4

  # Use 5 workers for maximum performance
  python test_visual_pro.py video.mp4 --workers 5

  # Adjust confidence threshold
  python test_visual_pro.py video.mp4 --confidence 0.3

  # Custom entry line and timeout
  python test_visual_pro.py video.mp4 --entry 0.2 --timeout 180
        """
    )

    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of AI processing workers (default: 3, recommended: 3-5)')
    parser.add_argument('--entry', type=float, default=0.3,
                       help='Entry line position (0.0-1.0, default: 0.3 = 30%% from top)')
    parser.add_argument('--timeout', type=float, default=120,
                       help='Vehicle timeout in seconds (default: 120 = 2 minutes)')

    args = parser.parse_args()

    # Check if AI service is running
    tester = ProVisualTester(
        confidence=args.confidence,
        num_workers=args.workers,
        entry_line_y=args.entry,
        timeout_seconds=args.timeout
    )

    print("=" * 70)
    print("🚦 Road Sentinel - Professional Visual Testing")
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

    # Process video
    tester.process_video(args.video)


if __name__ == "__main__":
    main()
