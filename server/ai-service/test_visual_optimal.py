#!/usr/bin/env python3
"""
OPTIMAL Visual Tester - TRUE 30+ FPS with Decoupled Architecture

Architecture:
- MAIN THREAD: Display only (never blocks) - reads video, draws cached boxes
- TRACKING THREAD: Runs DeepSORT (the slow part) in background
- AI WORKERS: HTTP requests to AI service in parallel

This ensures smooth video playback regardless of AI/tracking speed.
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
from dataclasses import dataclass, field
from copy import deepcopy

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


@dataclass
class TrackedBox:
    """A single tracked bounding box with interpolation support"""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    vehicle_type: str
    confidence: float
    frame_num: int
    # Velocity for interpolation (pixels per frame)
    vx: float = 0.0
    vy: float = 0.0

    def interpolate(self, target_frame: int, max_extrapolate_frames: int = 30) -> 'TrackedBox':
        """Interpolate box position to target frame"""
        frames_diff = target_frame - self.frame_num
        # Limit extrapolation to prevent boxes flying off screen
        clamped_diff = max(-max_extrapolate_frames, min(frames_diff, max_extrapolate_frames))
        return TrackedBox(
            track_id=self.track_id,
            x1=self.x1 + self.vx * clamped_diff,
            y1=self.y1 + self.vy * clamped_diff,
            x2=self.x2 + self.vx * clamped_diff,
            y2=self.y2 + self.vy * clamped_diff,
            vehicle_type=self.vehicle_type,
            confidence=self.confidence,
            frame_num=target_frame,
            vx=self.vx,
            vy=self.vy
        )


class BoxCache:
    """Thread-safe cache of tracked boxes for display"""

    def __init__(self):
        self._lock = threading.Lock()
        self._boxes: Dict[int, TrackedBox] = {}  # track_id -> TrackedBox
        self._last_update_frame = 0
        self._history: Dict[int, deque] = {}  # track_id -> history of boxes

    def update(self, boxes: List[TrackedBox], frame_num: int):
        """Update cache with new tracked boxes (called by tracking thread)"""
        with self._lock:
            # Calculate velocities from history
            new_boxes = {}
            for box in boxes:
                if box.track_id in self._history:
                    history = self._history[box.track_id]
                    if len(history) > 0:
                        prev = history[-1]
                        dt = box.frame_num - prev.frame_num
                        if dt > 0:
                            box.vx = (box.x1 - prev.x1) / dt
                            box.vy = (box.y1 - prev.y1) / dt
                else:
                    self._history[box.track_id] = deque(maxlen=10)

                self._history[box.track_id].append(box)
                new_boxes[box.track_id] = box

            self._boxes = new_boxes
            self._last_update_frame = frame_num

    def get_interpolated(self, frame_num: int, max_age: int = 9999) -> List[TrackedBox]:
        """Get boxes interpolated to current frame (called by main thread)"""
        with self._lock:
            result = []
            for track_id, box in self._boxes.items():
                age = frame_num - box.frame_num
                # Always show boxes - interpolate to current position
                # Don't filter by age since detection can lag significantly
                if age <= max_age and age >= 0:
                    interpolated = box.interpolate(frame_num)
                    result.append(interpolated)
            return result

    def get_last_update_frame(self) -> int:
        with self._lock:
            return self._last_update_frame

    def get_box_count(self) -> int:
        with self._lock:
            return len(self._boxes)


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


class OptimalTester:
    """TRUE 30+ FPS with fully decoupled architecture"""

    def __init__(self, confidence: float = 0.5, entry_line_y: float = 0.3,
                 timeout_seconds: float = 120, num_workers: int = 2,
                 detection_mode: str = 'traffic'):
        self.confidence = confidence
        self.paused = False
        self.entry_line_y = entry_line_y
        self.timeout_seconds = timeout_seconds
        self.timeout_frames = 0
        self.num_workers = num_workers
        self.detection_mode = detection_mode

        # Thread-safe box cache for display
        self.box_cache = BoxCache()

        # Queues for inter-thread communication
        self.frame_queue = queue.Queue(maxsize=max(16, num_workers * 2))
        self.detection_queue = queue.Queue(maxsize=max(20, num_workers * 3))

        # Thread control
        self.stop_event = threading.Event()
        self.ai_workers = []
        self.tracking_thread = None

        # Vehicle tracking state (updated by tracking thread)
        self.vehicles = {}
        self.counted_in = set()
        self.timed_out_vehicles = set()
        self.vehicles_lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'ai_processed_frames': 0,
            'display_fps_samples': deque(maxlen=60),
            'unique_vehicles': 0,
            'vehicles_entered': 0,
            'vehicles_timed_out': 0,
            'active_vehicles': 0,
            'vehicle_types': defaultdict(int),
            'ai_times': [],
            'tracking_times': [],
        }
        self.stats_lock = threading.Lock()

        self.frame_width = 0
        self.frame_height = 0
        self.video_fps = 0
        self.current_frame_num = 0

    def check_ai_service(self) -> bool:
        try:
            response = requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    # ==================== AI WORKER THREAD ====================
    def _ai_worker(self):
        """Background worker: sends frames to AI service"""
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue

                frame, frame_num = frame_data

                # Encode and send to AI service
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)

                files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                data = {
                    "camera_id": f"OPTIMAL-{frame_num}",
                    "confidence_threshold": str(self.confidence)
                }

                # Choose endpoint
                if self.detection_mode == 'traffic':
                    endpoint = f"{AI_SERVICE_URL}/api/detect/traffic"
                elif self.detection_mode == 'incidents':
                    endpoint = f"{AI_SERVICE_URL}/api/detect/incidents"
                else:
                    endpoint = f"{AI_SERVICE_URL}/api/detect"

                start_time = time.time()
                response = requests.post(endpoint, files=files, data=data, timeout=10)
                processing_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    result = response.json()
                    result['frame_num'] = frame_num
                    result['frame'] = frame  # Include frame for DeepSORT embeddings
                    result['processing_time_ms'] = processing_time

                    with self.stats_lock:
                        self.stats['ai_times'].append(processing_time)
                        self.stats['ai_processed_frames'] += 1

                    # Send to tracking thread
                    try:
                        self.detection_queue.put_nowait(result)
                    except queue.Full:
                        pass  # Drop if tracking can't keep up

                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"AI worker error: {e}")

    # ==================== TRACKING THREAD ====================
    def _tracking_worker(self):
        """Background worker: runs DeepSORT tracking (the slow part)"""
        # Initialize DeepSORT in this thread
        tracker = DeepSort(
            max_age=60,
            n_init=2,
            max_iou_distance=0.8,
            embedder="mobilenet",
            half=True,
            embedder_gpu=True
        )

        last_frame_num = 0

        while not self.stop_event.is_set():
            try:
                result = self.detection_queue.get(timeout=0.1)
                if result is None:
                    continue

                frame_num = result['frame_num']
                frame = result['frame']
                detections = result.get('detections', [])

                # Debug: print when we get detections
                if detections:
                    print(f"[Track] Frame {frame_num}: {len(detections)} detections received")

                start_time = time.time()

                # Convert detections to DeepSORT format
                raw_detections = []
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    conf = det.get('confidence', 0.5)
                    if conf is None:
                        conf = 0.5
                    raw_detections.append(([x1, y1, w, h], conf, det['class']))

                # Run DeepSORT (this is the slow operation)
                tracks = tracker.update_tracks(raw_detections, frame=frame)

                tracking_time = (time.time() - start_time) * 1000
                with self.stats_lock:
                    self.stats['tracking_times'].append(tracking_time)

                # Convert tracks to TrackedBox objects
                tracked_boxes = []
                for track in tracks:
                    if not track.is_confirmed() and not track.is_tentative():
                        continue

                    track_id = int(track.track_id)
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = ltrb

                    # Clamp to frame bounds
                    x1 = max(0, min(x1, self.frame_width))
                    y1 = max(0, min(y1, self.frame_height))
                    x2 = max(0, min(x2, self.frame_width))
                    y2 = max(0, min(y2, self.frame_height))

                    det_class = track.get_det_class() or 'unknown'
                    det_conf = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.0
                    if det_conf is None:
                        det_conf = 0.0

                    tracked_boxes.append(TrackedBox(
                        track_id=track_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        vehicle_type=det_class,
                        confidence=det_conf,
                        frame_num=frame_num
                    ))

                    # Update vehicle state
                    self._update_vehicle_state(track_id, det_class, (x1, y1, x2, y2), frame_num)

                # Update the box cache (main thread will read from this)
                self.box_cache.update(tracked_boxes, frame_num)

                # Debug: print cached boxes
                if tracked_boxes:
                    print(f"[Track] Frame {frame_num}: {len(tracked_boxes)} boxes cached")

                last_frame_num = frame_num
                self.detection_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Tracking worker error: {e}")
                import traceback
                traceback.print_exc()

    def _update_vehicle_state(self, track_id: int, vehicle_type: str,
                              bbox: Tuple[float, float, float, float], frame_num: int):
        """Update vehicle state and check line crossing (called by tracking thread)"""
        with self.vehicles_lock:
            if track_id not in self.vehicles:
                self.vehicles[track_id] = VehicleState(track_id, vehicle_type, frame_num)
                with self.stats_lock:
                    self.stats['unique_vehicles'] += 1
                    self.stats['vehicle_types'][vehicle_type] += 1

            vehicle = self.vehicles[track_id]
            vehicle.update_position(bbox, frame_num)

            # Check line crossing
            center = vehicle.get_center()
            if center:
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
                                with self.stats_lock:
                                    self.stats['vehicles_entered'] += 1
                                print(f"Vehicle #{vehicle.track_id} ({vehicle.vehicle_type}) ENTERED")

    # ==================== THREAD MANAGEMENT ====================
    def start_workers(self):
        """Start all background workers"""
        self.stop_event.clear()

        # Start AI workers
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._ai_worker, daemon=True)
            worker.start()
            self.ai_workers.append(worker)

        # Start single tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_worker, daemon=True)
        self.tracking_thread.start()

        print(f"Started {self.num_workers} AI workers + 1 tracking thread")

    def stop_workers(self):
        """Stop all workers"""
        self.stop_event.set()
        for worker in self.ai_workers:
            worker.join(timeout=2)
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2)
        self.ai_workers.clear()

    # ==================== DRAWING (Main Thread) ====================
    def draw_boxes(self, frame: np.ndarray, boxes: List[TrackedBox],
                   is_interpolated: bool = False) -> np.ndarray:
        """Draw tracked boxes on frame (called by main thread)"""
        annotated = frame.copy()

        for box in boxes:
            color = VEHICLE_COLORS.get(box.vehicle_type, VEHICLE_COLORS['unknown'])

            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)

            # Clamp to frame
            x1 = max(0, min(x1, self.frame_width - 1))
            y1 = max(0, min(y1, self.frame_height - 1))
            x2 = max(0, min(x2, self.frame_width - 1))
            y2 = max(0, min(y2, self.frame_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{box.track_id} {box.vehicle_type}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y1 - 10, label_size[1] + 10)

            cv2.rectangle(annotated, (x1, label_y - label_size[1] - 10),
                         (x1 + label_size[0], label_y), color, -1)
            cv2.putText(annotated, label, (x1, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated

    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        entry_y = int(self.frame_height * self.entry_line_y)
        cv2.line(frame, (0, entry_y), (self.frame_width, entry_y), LINE_COLOR, 3)
        cv2.putText(frame, "ENTRY LINE", (10, entry_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINE_COLOR, 2)
        return frame

    def draw_info_panel(self, frame: np.ndarray, display_fps: float,
                       current_pos: int, total_frames: int) -> np.ndarray:
        overlay = frame.copy()
        panel_height = 360
        cv2.rectangle(overlay, (0, 0), (420, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_pos = 25
        line_height = 22

        # Calculate stats
        with self.stats_lock:
            ai_times = self.stats['ai_times'][-100:] if self.stats['ai_times'] else [0]
            tracking_times = self.stats['tracking_times'][-100:] if self.stats['tracking_times'] else [0]
            ai_processed = self.stats['ai_processed_frames']
            unique_vehicles = self.stats['unique_vehicles']
            vehicles_entered = self.stats['vehicles_entered']

        avg_ai_time = sum(ai_times) / len(ai_times) if ai_times else 0
        avg_tracking_time = sum(tracking_times) / len(tracking_times) if tracking_times else 0

        progress_pct = (current_pos / total_frames) * 100 if total_frames > 0 else 0
        last_update = self.box_cache.get_last_update_frame()
        detection_lag = current_pos - last_update
        cached_boxes = self.box_cache.get_box_count()

        info_lines = [
            f"Display FPS: {display_fps:.1f}",
            f"Video: {current_pos}/{total_frames} ({progress_pct:.1f}%)",
            f"",
            f"AI Time: {avg_ai_time:.1f}ms ({1000/avg_ai_time:.1f} FPS)" if avg_ai_time > 0 else "AI Time: --",
            f"Track Time: {avg_tracking_time:.1f}ms" if avg_tracking_time > 0 else "Track Time: --",
            f"Detection Lag: {detection_lag} frames",
            f"Cached Boxes: {cached_boxes}",
            f"AI Processed: {ai_processed}",
            f"",
            f"Unique Vehicles: {unique_vehicles}",
            f"Vehicles ENTERED: {vehicles_entered}",
            f"",
            f"Mode: {self.detection_mode.upper()}",
            f"Workers: {self.num_workers}",
            f"",
            f"SPACE=Pause Q=Quit"
        ]

        for line in info_lines:
            if line:
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += line_height

        return frame

    # ==================== MAIN LOOP ====================
    def process_video(self, video_path: str):
        print(f"Opening video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.timeout_frames = int(self.timeout_seconds * self.video_fps)

        # Use video's native FPS or cap at 60
        target_fps = min(self.video_fps, 60) if self.video_fps > 0 else 30
        frame_delay = max(1, int(1000 / target_fps))

        print(f"Video: {total_frames} frames, {self.video_fps:.2f} FPS, {self.frame_width}x{self.frame_height}")
        print(f"Target display: {target_fps:.1f} FPS ({frame_delay}ms delay)")
        print(f"Mode: {self.detection_mode.upper()}")
        print()

        window_name = "Road Sentinel - Decoupled Architecture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Start background workers
        self.start_workers()

        frame_count = 0
        last_time = time.time()
        display_fps = 0.0
        fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
        last_fps_update = time.time()
        frame_times = deque(maxlen=30)

        # AI frame interval (send every N frames to AI)
        ai_interval = 3

        try:
            while True:
                loop_start = time.time()

                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\nVideo finished")
                        break

                    frame_count += 1
                    self.current_frame_num = frame_count

                    with self.stats_lock:
                        self.stats['total_frames'] = frame_count

                    # Send frame to AI workers (every N frames)
                    if frame_count % ai_interval == 1:
                        try:
                            self.frame_queue.put_nowait((frame.copy(), frame_count))
                        except queue.Full:
                            pass  # Skip if queue full

                    # Get interpolated boxes from cache (INSTANT - no blocking!)
                    # Use large max_age since detection can lag significantly behind display
                    boxes = self.box_cache.get_interpolated(frame_count, max_age=9999)

                    # Draw on frame
                    annotated = self.draw_boxes(frame, boxes)
                    annotated = self.draw_counting_line(annotated)

                    # Calculate display FPS
                    current_time = time.time()
                    frame_times.append(current_time - loop_start)

                    if current_time - last_fps_update >= fps_update_interval:
                        if frame_times:
                            avg_frame_time = sum(frame_times) / len(frame_times)
                            display_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                        last_fps_update = current_time

                    # Draw info panel
                    annotated = self.draw_info_panel(annotated, display_fps, frame_count, total_frames)

                    # Show frame
                    cv2.imshow(window_name, annotated)

                # Handle input - use remaining time for waitKey
                elapsed = (time.time() - loop_start) * 1000
                wait_time = max(1, frame_delay - int(elapsed))
                key = cv2.waitKey(wait_time) & 0xFF

                if key == ord('q') or key == 27:
                    print("\nStopped by user")
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")

        finally:
            self.stop_workers()
            cap.release()
            cv2.destroyAllWindows()

        self.print_statistics()

    def print_statistics(self):
        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)

        with self.stats_lock:
            print(f"Total frames displayed: {self.stats['total_frames']}")
            print(f"AI frames processed: {self.stats['ai_processed_frames']}")

            print(f"\nVEHICLE TRACKING:")
            print(f"   Unique vehicles: {self.stats['unique_vehicles']}")
            print(f"   Vehicles entered: {self.stats['vehicles_entered']}")

            if self.stats['ai_times']:
                ai_times = self.stats['ai_times']
                print(f"\nAI PERFORMANCE:")
                print(f"   Avg: {sum(ai_times)/len(ai_times):.1f}ms")
                print(f"   Min: {min(ai_times):.1f}ms")
                print(f"   Max: {max(ai_times):.1f}ms")

            if self.stats['tracking_times']:
                track_times = self.stats['tracking_times']
                print(f"\nTRACKING PERFORMANCE:")
                print(f"   Avg: {sum(track_times)/len(track_times):.1f}ms")
                print(f"   Min: {min(track_times):.1f}ms")
                print(f"   Max: {max(track_times):.1f}ms")

            if self.stats['vehicle_types']:
                print("\nVEHICLE TYPES:")
                for vtype, count in sorted(self.stats['vehicle_types'].items(),
                                          key=lambda x: x[1], reverse=True):
                    print(f"   {vtype}: {count}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Road Sentinel - Decoupled Architecture (TRUE 30+ FPS)',
        epilog="""
Examples:
  python test_visual_optimal.py video.mp4 --workers 4
  python test_visual_optimal.py video.mp4 --workers 8 --mode all
        """
    )

    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--entry', type=float, default=0.3,
                       help='Entry line position (default: 0.3)')
    parser.add_argument('--timeout', type=float, default=120,
                       help='Vehicle timeout in seconds (default: 120)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of AI workers (default: 4)')
    parser.add_argument('--mode', type=str, default='traffic',
                       choices=['traffic', 'incidents', 'all'],
                       help='Detection mode (default: traffic)')

    args = parser.parse_args()

    tester = OptimalTester(
        confidence=args.confidence,
        entry_line_y=args.entry,
        timeout_seconds=args.timeout,
        num_workers=args.workers,
        detection_mode=args.mode
    )

    print("=" * 70)
    print("Road Sentinel - Decoupled Architecture")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  - Main Thread: Display only (never blocks)")
    print("  - AI Workers: HTTP requests to AI service")
    print("  - Tracking Thread: DeepSORT (runs in background)")
    print()

    if not tester.check_ai_service():
        print("AI service is not running!")
        print("Please start: python -m app.main")
        sys.exit(1)

    print("AI service is running")
    print()

    tester.process_video(args.video)


if __name__ == "__main__":
    main()
