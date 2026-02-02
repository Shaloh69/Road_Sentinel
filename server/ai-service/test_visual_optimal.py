#!/usr/bin/env python3
"""
OPTIMAL Visual Tester - 30 FPS + ZERO LAG
Uses DeepSORT predictive tracking for perfect sync at full speed

How it works:
- Process AI every 2-3 frames (realistic with 40-50ms processing time)
- Use DeepSORT's Kalman filter to PREDICT positions for intermediate frames
- Boxes stay perfectly synced because predictions are very accurate
- 30 FPS playback with zero visual lag!
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
import threading
import queue

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

INCIDENT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 255)


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


class OptimalTester:
    """30 FPS with ZERO LAG using DeepSORT predictive tracking"""

    def __init__(self, confidence: float = 0.5, entry_line_y: float = 0.3,
                 timeout_seconds: float = 120, num_workers: int = 2):
        self.confidence = confidence
        self.paused = False
        self.entry_line_y = entry_line_y
        self.timeout_seconds = timeout_seconds
        self.timeout_frames = 0
        self.num_workers = num_workers

        # DeepSORT tracker with optimized parameters for fewer ID switches
        self.tracker = DeepSort(
            max_age=60,  # Keep tracks longer without detection (reduce ID switches)
            n_init=2,    # Confirm tracks faster (less delay)
            max_iou_distance=0.8,  # More lenient matching (reduce ID switches)
            embedder="mobilenet",
            half=True,
            embedder_gpu=True
        )

        # Async AI processing - larger queues for better worker utilization
        self.frame_queue = queue.Queue(maxsize=max(16, num_workers * 2))
        self.result_queue = queue.Queue(maxsize=max(20, num_workers * 3))
        self.stop_processing = threading.Event()
        self.workers = []

        # Dynamic AI interval tracking
        self.base_ai_frame_interval = 3  # Base interval
        self.current_ai_interval = 3  # Dynamically adjusted
        self.max_ai_interval = 10  # Don't skip more than this
        self.queue_high_watermark = max(8, num_workers)  # When to start skipping
        self.queue_low_watermark = max(2, num_workers // 2)  # When to resume normal rate

        # Vehicle tracking state
        self.vehicles = {}
        self.counted_in = set()
        self.timed_out_vehicles = set()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'ai_processed_frames': 0,
            'predicted_frames': 0,
            'unique_vehicles': 0,
            'vehicles_entered': 0,
            'vehicles_timed_out': 0,
            'active_vehicles': 0,
            'vehicle_types': defaultdict(int),
            'avg_ai_time': 0,
            'ai_times': [],
            'ai_intervals': [],  # Track dynamic interval changes
            'frames_queued': 0,
            'frames_dropped': 0
        }

        self.frame_width = 0
        self.frame_height = 0
        self.video_fps = 0
        self.latest_detections = None
        self.processing_lock = threading.Lock()

    def check_ai_service(self) -> bool:
        try:
            response = requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _processing_worker(self):
        """Background worker for async AI processing"""
        while not self.stop_processing.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                frame, frame_num = frame_data

                # Process with AI (this blocks but it's in background thread)
                result = self.detect_frame(frame, frame_num)

                if result:
                    result['frame_num'] = frame_num  # Track which frame this is for
                    # Use blocking put with timeout - don't discard valid results
                    try:
                        self.result_queue.put(result, timeout=0.5)
                        self.stats['ai_processed_frames'] += 1
                    except queue.Full:
                        # Only discard if truly full after waiting
                        pass

                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def start_async_processing(self):
        """Start background AI workers"""
        self.stop_processing.clear()
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._processing_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        print(f"🚀 Started {self.num_workers} AI workers in background")

    def stop_async_processing(self):
        """Stop all workers"""
        self.stop_processing.set()
        for worker in self.workers:
            worker.join(timeout=2)
        self.workers.clear()

    def detect_frame(self, frame: np.ndarray, frame_num: int) -> Optional[Dict[str, Any]]:
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)

            files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            data = {
                "camera_id": f"OPTIMAL-{frame_num}",
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
                self.stats['ai_times'].append(processing_time)
                return result

            return None

        except Exception as e:
            print(f"Detection error: {e}")
            return None

    def update_tracking(self, detections: Optional[List[Dict]], frame: np.ndarray,
                       frame_num: int, is_ai_frame: bool):
        """
        Update tracking with AI detections OR use predictions

        is_ai_frame=True: New AI detections, update tracker
        is_ai_frame=False: No AI, use tracker predictions only
        """

        if is_ai_frame and detections:
            # AI frame: Update tracker with new detections
            raw_detections = []
            for det in detections:
                bbox = det['bbox']
                x1, y1, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                raw_detections.append(([x1, y1, w, h], det['confidence'], det['class']))

            tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        else:
            # Prediction frame: Update tracker without detections (uses Kalman prediction)
            tracks = self.tracker.update_tracks([], frame=frame)

        # Process tracks (both confirmed and predicted)
        tracked_detections = []
        for track in tracks:
            # Accept both confirmed and tentative tracks for smooth display
            if not track.is_confirmed() and not track.is_tentative():
                continue

            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Ensure bbox stays within frame
            x1 = max(0, min(x1, self.frame_width))
            y1 = max(0, min(y1, self.frame_height))
            x2 = max(0, min(x2, self.frame_width))
            y2 = max(0, min(y2, self.frame_height))

            det_class = track.get_det_class()
            det_conf = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.0

            tracked_det = {
                'track_id': track_id,
                'class': det_class if det_class else 'unknown',
                'confidence': det_conf,
                'bbox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                'predicted': not is_ai_frame  # Mark if this is a prediction
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
            confidence = det.get('confidence', 0.0)
            if confidence is None:
                confidence = 0.0
            bbox = det['bbox']
            is_predicted = det.get('predicted', False)

            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

            color = VEHICLE_COLORS.get(vehicle_type, VEHICLE_COLORS['unknown'])

            # Draw bounding box (dashed if predicted)
            if is_predicted:
                # Dashed box for predictions
                self.draw_dashed_rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            else:
                # Solid box for AI detections
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw unique ID and label
            if track_id is not None and track_id != -1:
                pred_marker = "P" if is_predicted else ""
                label = f"ID:{track_id}{pred_marker} {vehicle_type} {confidence:.2f}"
            else:
                label = f"{vehicle_type} {confidence:.2f}"

            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y - 10, label_size[1] + 10)

            cv2.rectangle(annotated, (x, label_y - label_size[1] - 10),
                         (x + label_size[0], label_y), color, -1)

            cv2.putText(annotated, label, (x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated

    def draw_dashed_rectangle(self, img, pt1, pt2, color, thickness, dash_length=10):
        """Draw dashed rectangle for predicted boxes"""
        x1, y1 = pt1
        x2, y2 = pt2

        # Top line
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        # Bottom line
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
        # Left line
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        # Right line
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

    def draw_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        entry_y = int(height * self.entry_line_y)
        cv2.line(frame, (0, entry_y), (width, entry_y), LINE_COLOR, 3)
        cv2.putText(frame, "ENTRY LINE", (10, entry_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINE_COLOR, 2)
        return frame

    def draw_info_panel(self, frame: np.ndarray, fps: float, ai_time: float,
                       current_pos: int = 0, total_frames: int = 0,
                       is_ai_frame: bool = False, queue_size: int = 0) -> np.ndarray:
        height, width = frame.shape[:2]

        overlay = frame.copy()
        panel_height = 360
        cv2.rectangle(overlay, (0, 0), (450, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_pos = 25
        line_height = 22

        progress_text = ""
        if total_frames > 0:
            progress_pct = (current_pos / total_frames) * 100
            progress_text = f"Progress: {current_pos}/{total_frames} ({progress_pct:.1f}%)"

        timeout_mins = self.timeout_seconds / 60
        mode_text = "AI FRAME" if is_ai_frame else "PREDICTED"

        # Calculate effective AI FPS
        effective_ai_fps = 30.0 / self.current_ai_interval if self.current_ai_interval > 0 else 0

        info_lines = [
            f"Display FPS: {fps:.1f} (Target: 30 FPS)",
            f"Mode: {mode_text}",
            f"AI Processing: {ai_time:.1f}ms",
            f"AI Interval: {self.current_ai_interval} frames (~{effective_ai_fps:.1f} AI FPS)",
            f"Queue: {queue_size}/{self.frame_queue.maxsize} | Workers: {self.num_workers}",
            progress_text if progress_text else f"Frames: {self.stats['total_frames']}",
            "",
            f"Unique Vehicles: {self.stats['unique_vehicles']}",
            f"Vehicles ENTERED: {self.stats['vehicles_entered']}",
            f"Active Now: {self.stats['active_vehicles']}",
            f"Timed Out ({timeout_mins:.0f}m): {self.stats['vehicles_timed_out']}",
            "",
            "Legend:",
            "Solid box = AI Detection",
            "Dashed box = Predicted",
            "",
            "Controls: SPACE=Pause Q=Quit"
        ]

        for line in info_lines:
            if line:
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += line_height

        # Draw vehicle type counts
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

        # ALWAYS use 30 FPS playback regardless of video's native FPS
        target_fps = 30
        frame_delay = int(1000 / target_fps)  # 33ms for 30 FPS

        print(f"📹 Video: {total_frames} frames, {self.video_fps:.2f} FPS (native), {self.frame_width}x{self.frame_height}")
        print(f"🎯 OPTIMAL MODE - Playing at {target_fps} FPS with ZERO LAG")
        print(f"⚡ DeepSORT predictive tracking + {self.num_workers} async AI workers")
        print(f"📦 Queue sizes: frame={self.frame_queue.maxsize}, result={self.result_queue.maxsize}")
        print(f"🔄 Dynamic AI interval: {self.base_ai_frame_interval}-{self.max_ai_interval} frames (auto-adjusts)")
        print(f"📏 Entry line: {self.entry_line_y*100:.0f}%")
        print()

        window_name = "Road Sentinel - OPTIMAL (30 FPS + Zero Lag)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        last_time = time.time()
        display_fps = 0
        last_ai_time = 0

        # Start async AI workers
        self.start_async_processing()

        try:
            while True:
                # Check for AI results (non-blocking)
                try:
                    result = self.result_queue.get_nowait()
                    if result:
                        with self.processing_lock:
                            self.latest_detections = result.get('detections', [])
                            last_ai_time = result.get('processing_time_ms', 0)
                except queue.Empty:
                    pass

                if not self.paused:
                    ret, frame = cap.read()

                    if not ret:
                        print("\n✅ Video finished")
                        break

                    frame_count += 1
                    self.stats['total_frames'] += 1

                    # Dynamic AI interval adjustment based on queue backlog
                    current_queue_size = self.frame_queue.qsize()
                    if current_queue_size >= self.queue_high_watermark:
                        # Queue backing up - skip more frames
                        self.current_ai_interval = min(self.current_ai_interval + 1, self.max_ai_interval)
                    elif current_queue_size <= self.queue_low_watermark:
                        # Queue draining - process more frames
                        self.current_ai_interval = max(self.current_ai_interval - 1, self.base_ai_frame_interval)

                    # Send frame to AI workers (non-blocking, async)
                    if frame_count % self.current_ai_interval == 1:
                        try:
                            self.frame_queue.put_nowait((frame.copy(), frame_count))
                            self.stats['frames_queued'] += 1
                            self.stats['ai_intervals'].append(self.current_ai_interval)
                        except queue.Full:
                            self.stats['frames_dropped'] += 1  # Track dropped frames

                    # ALWAYS use predictions for display (NEVER wait for AI)
                    with self.processing_lock:
                        current_detections = self.latest_detections

                    # Update tracking (uses predictions if no new AI data)
                    has_new_ai = current_detections is not None
                    tracked_detections = self.update_tracking(
                        current_detections if has_new_ai else None,
                        frame,
                        frame_count,
                        has_new_ai
                    )

                    if not has_new_ai:
                        self.stats['predicted_frames'] += 1

                    # Draw detections (always from predictions - no lag!)
                    annotated = self.draw_detections(frame, tracked_detections)

                    # Draw counting lines
                    annotated = self.draw_counting_lines(annotated)

                    # Calculate display FPS
                    current_time = time.time()
                    time_diff = current_time - last_time
                    if time_diff > 0:
                        display_fps = 1.0 / time_diff
                    last_time = current_time

                    # Draw info panel
                    is_ai_frame = (frame_count % self.current_ai_interval == 1)
                    annotated = self.draw_info_panel(annotated, display_fps, last_ai_time,
                                                     frame_count, total_frames, is_ai_frame,
                                                     current_queue_size)

                    # Show frame
                    cv2.imshow(window_name, annotated)

                # Handle keyboard input
                key = cv2.waitKey(frame_delay) & 0xFF

                if key == ord('q') or key == 27:
                    print("\n⏹️  Stopped by user")
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                    print("⏸️  Paused" if self.paused else "▶️  Resumed")

        finally:
            self.stop_async_processing()
            cap.release()
            cv2.destroyAllWindows()

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        print(f"Total frames: {self.stats['total_frames']}")
        print(f"AI processed: {self.stats['ai_processed_frames']}")
        print(f"Predicted: {self.stats['predicted_frames']}")
        print(f"Frames queued: {self.stats['frames_queued']}")
        print(f"Frames dropped (queue full): {self.stats['frames_dropped']}")

        print(f"\nVEHICLE TRACKING:")
        print(f"   Unique vehicles seen: {self.stats['unique_vehicles']}")
        print(f"   Vehicles ENTERED: {self.stats['vehicles_entered']}")
        print(f"   Vehicles timed out ({self.timeout_seconds/60:.0f} min): {self.stats['vehicles_timed_out']}")

        if self.stats['ai_times']:
            avg_time = sum(self.stats['ai_times']) / len(self.stats['ai_times'])
            min_time = min(self.stats['ai_times'])
            max_time = max(self.stats['ai_times'])
            print(f"\nPERFORMANCE:")
            print(f"   Workers: {self.num_workers}")
            print(f"   Avg AI processing: {avg_time:.1f}ms per frame")
            print(f"   Min/Max AI time: {min_time:.1f}ms / {max_time:.1f}ms")
            print(f"   Theoretical max (1 worker): {1000/avg_time:.1f} FPS")
            print(f"   Theoretical max ({self.num_workers} workers): {self.num_workers * 1000/avg_time:.1f} FPS")

        if self.stats['ai_intervals']:
            avg_interval = sum(self.stats['ai_intervals']) / len(self.stats['ai_intervals'])
            min_interval = min(self.stats['ai_intervals'])
            max_interval = max(self.stats['ai_intervals'])
            print(f"\nDYNAMIC AI INTERVAL:")
            print(f"   Avg interval: {avg_interval:.1f} frames")
            print(f"   Range: {min_interval} - {max_interval} frames")
            print(f"   Effective AI FPS: ~{30/avg_interval:.1f} FPS")

        if self.stats['vehicle_types']:
            print("\nVehicle Types:")
            for vtype, count in sorted(self.stats['vehicle_types'].items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"   {vtype}: {count}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='OPTIMAL Visual Testing - 30 FPS + Zero Lag',
        epilog="""
Examples:
  # Test with video (2 async workers - default)
  python test_visual_optimal.py video.mp4

  # Use 3 workers for more AI throughput
  python test_visual_optimal.py video.mp4 --workers 3

  # Adjust confidence
  python test_visual_optimal.py video.mp4 --confidence 0.3
        """
    )

    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--entry', type=float, default=0.3,
                       help='Entry line position (default: 0.3)')
    parser.add_argument('--timeout', type=float, default=120,
                       help='Vehicle timeout in seconds (default: 120)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of async AI workers (default: 2)')

    args = parser.parse_args()

    tester = OptimalTester(
        confidence=args.confidence,
        entry_line_y=args.entry,
        timeout_seconds=args.timeout,
        num_workers=args.workers
    )

    print("=" * 70)
    print("🚦 Road Sentinel - OPTIMAL Testing (30 FPS + Zero Lag)")
    print("=" * 70)
    print()

    if not tester.check_ai_service():
        print("❌ AI service is not running!")
        print("Please start: python -m app.main")
        sys.exit(1)

    print("✅ AI service is running")
    print()

    tester.process_video(args.video)


if __name__ == "__main__":
    main()
