#!/usr/bin/env python3
"""
Camera Connection Test Script for Road Sentinel
Automatically detects and connects to any available camera
Optionally runs YOLO26 inference if AI service is available
"""

import cv2
import numpy as np
import sys
import time
import argparse
import requests
import threading
import queue
from typing import Optional, Tuple, List, Dict

# AI Service configuration
AI_SERVICE_URL = "http://localhost:8000"

# Colors for visualization (BGR format)
VEHICLE_COLORS = {
    'car': (0, 255, 0),        # Green
    'truck': (255, 165, 0),    # Orange
    'bus': (0, 255, 255),      # Yellow
    'motorcycle': (255, 0, 255),  # Magenta
    'bicycle': (255, 255, 0),  # Cyan
    'unknown': (128, 128, 128) # Gray
}
INCIDENT_COLOR = (0, 0, 255)  # Red


def find_available_cameras(max_cameras: int = 5) -> List[int]:
    """
    Scan for available cameras by trying to open camera indices 0 to max_cameras-1

    Args:
        max_cameras: Maximum number of camera indices to check

    Returns:
        List of available camera indices
    """
    available_cameras = []

    print(f"Scanning for available cameras (checking indices 0-{max_cameras-1})...")

    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Try to read a frame to verify camera is working
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                print(f"✓ Camera {camera_id} found: {width}x{height} @ {fps}fps")
                available_cameras.append(camera_id)
            cap.release()

    return available_cameras


def check_ai_service() -> bool:
    """Check if AI service is running and accessible"""
    try:
        response = requests.get(f"{AI_SERVICE_URL}/health", timeout=2)
        if response.status_code == 200:
            print(f"✓ AI Service is running at {AI_SERVICE_URL}")
            return True
    except:
        pass

    print(f"✗ AI Service not available at {AI_SERVICE_URL}")
    print("  Running in camera-only mode (no AI detection)")
    return False


def detect_with_ai(frame: np.ndarray, confidence: float = 0.5) -> Optional[Dict]:
    """
    Send frame to AI service for detection

    Args:
        frame: Image frame (BGR format)
        confidence: Confidence threshold for detections

    Returns:
        Detection results or None if request fails
    """
    try:
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        # Prepare request
        files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        data = {
            "camera_id": "test_camera",
            "confidence_threshold": str(confidence)
        }

        # Send request to AI service with shorter timeout
        response = requests.post(
            f"{AI_SERVICE_URL}/api/detect",
            files=files,
            data=data,
            timeout=2  # Reduced timeout for faster response
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"AI service error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Detection error: {e}")
        return None


def draw_detections(frame: np.ndarray, result: Dict) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame

    Args:
        frame: Image frame to draw on
        result: Detection results from AI service

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Draw vehicle detections
    if 'detections' in result:
        for detection in result['detections']:
            bbox = detection['bbox']
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']

            vehicle_type = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)

            # Get color for vehicle type
            color = VEHICLE_COLORS.get(vehicle_type, VEHICLE_COLORS['unknown'])

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label with background
            label = f"{vehicle_type}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw incident detections
    if 'incidents' in result and isinstance(result['incidents'], list):
        for incident in result['incidents']:
            # Check if incident has bbox (some incidents may not have location)
            if 'bbox' not in incident:
                continue

            bbox = incident['bbox']
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']

            incident_type = incident.get('type', 'unknown')
            confidence = incident.get('confidence', 0)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), INCIDENT_COLOR, 3)

            # Draw label with background
            label = f"INCIDENT: {incident_type} ({confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - label_h - 10), (x + label_w, y), INCIDENT_COLOR, -1)
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def add_info_overlay(frame: np.ndarray, camera_id: int, fps: float,
                     detection_count: int = 0, ai_enabled: bool = False) -> np.ndarray:
    """
    Add information overlay to frame

    Args:
        frame: Image frame
        camera_id: Camera ID
        fps: Current FPS
        detection_count: Number of detections
        ai_enabled: Whether AI detection is enabled

    Returns:
        Frame with overlay
    """
    overlay = frame.copy()

    # Create semi-transparent background
    info_height = 120
    cv2.rectangle(overlay, (0, 0), (400, info_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Add text information
    y_pos = 25
    line_height = 25

    cv2.putText(frame, f"Camera ID: {camera_id}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += line_height

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += line_height

    ai_status = "ON" if ai_enabled else "OFF"
    ai_color = (0, 255, 0) if ai_enabled else (0, 0, 255)
    cv2.putText(frame, f"AI Detection: {ai_status}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ai_color, 2)
    y_pos += line_height

    if ai_enabled:
        cv2.putText(frame, f"Detections: {detection_count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add controls help
    help_y = frame.shape[0] - 70
    cv2.putText(frame, "Controls:", (10, help_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Q - Quit  |  SPACE - Toggle AI  |  C - Change Camera",
                (10, help_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def detection_worker(frame_queue: queue.Queue, result_queue: queue.Queue, confidence: float, stop_event: threading.Event):
    """
    Background worker thread for AI detection
    Processes frames asynchronously without blocking the main video feed
    """
    while not stop_event.is_set():
        try:
            # Get frame from queue (non-blocking with timeout)
            frame = frame_queue.get(timeout=0.1)

            # Run detection
            result = detect_with_ai(frame, confidence)

            if result:
                # Put result in queue (discard old results if queue is full)
                try:
                    result_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result and add new one
                    try:
                        result_queue.get_nowait()
                        result_queue.put_nowait(result)
                    except:
                        pass

            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            # Silent error handling - don't spam console
            pass


def run_camera_test(camera_id: int, use_ai: bool = True, confidence: float = 0.5):
    """
    Run camera test with optional AI detection (ASYNC - Full 30 FPS)

    Args:
        camera_id: Camera index to use
        use_ai: Whether to use AI detection
        confidence: Confidence threshold for AI detection
    """
    # Open camera with DirectShow backend (more reliable on Windows)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS

    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\n{'='*60}")
    print(f"Camera {camera_id} opened successfully!")
    print(f"Resolution: {width}x{height}")
    print(f"Camera FPS: {fps_cap}")
    print(f"AI Detection: {'Enabled (ASYNC - Full FPS)' if use_ai else 'Disabled'}")
    if use_ai:
        print(f"Confidence Threshold: {confidence}")
    print(f"{'='*60}\n")

    # Create window
    window_name = f"Road Sentinel - Camera {camera_id} Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    # Async processing queues and thread
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    detection_thread = None

    # Processing control
    ai_enabled = use_ai
    process_every_n_frames = 5  # Send every 5th frame for detection (still get 30 FPS display)
    frame_count = 0
    last_result = None
    detection_count = 0

    # Start detection thread if AI is enabled
    if ai_enabled:
        detection_thread = threading.Thread(
            target=detection_worker,
            args=(frame_queue, result_queue, confidence, stop_event),
            daemon=True
        )
        detection_thread.start()

    print("Camera feed started. Press 'Q' to quit, SPACE to toggle AI, 'C' to change camera")

    try:
        while True:
            # Read frame at full FPS
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame from camera")
                break

            frame_count += 1
            annotated_frame = frame.copy()

            # Send frame for AI detection (non-blocking)
            if ai_enabled and frame_count % process_every_n_frames == 0:
                try:
                    # Non-blocking put - if queue is full, skip this frame
                    frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip this frame if queue is full

            # Check for new detection results (non-blocking)
            if ai_enabled:
                try:
                    # Get latest result without blocking
                    new_result = result_queue.get_nowait()
                    last_result = new_result
                    detection_count = len(last_result.get('detections', []))
                except queue.Empty:
                    pass  # No new results, use last_result

            # Draw detections if we have results
            if ai_enabled and last_result:
                annotated_frame = draw_detections(annotated_frame, last_result)

            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # Add info overlay
            annotated_frame = add_info_overlay(
                annotated_frame,
                camera_id,
                current_fps,
                detection_count,
                ai_enabled
            )

            # Display frame
            cv2.imshow(window_name, annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):  # Space bar - toggle AI
                ai_enabled = not ai_enabled
                print(f"AI Detection: {'Enabled' if ai_enabled else 'Disabled'}")

                # Start or stop detection thread
                if ai_enabled and (detection_thread is None or not detection_thread.is_alive()):
                    stop_event.clear()
                    detection_thread = threading.Thread(
                        target=detection_worker,
                        args=(frame_queue, result_queue, confidence, stop_event),
                        daemon=True
                    )
                    detection_thread.start()
                elif not ai_enabled:
                    stop_event.set()

                last_result = None
                detection_count = 0
            elif key == ord('c') or key == ord('C'):  # C - change camera
                print("\nChanging camera...")
                stop_event.set()
                cap.release()
                return 'change_camera'

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        stop_event.set()
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()

    return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test camera connection for Road Sentinel')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera ID to use (default: auto-detect)')
    parser.add_argument('--no-ai', action='store_true',
                        help='Disable AI detection')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for AI detection (default: 0.5)')
    parser.add_argument('--list', action='store_true',
                        help='List available cameras and exit')

    args = parser.parse_args()

    print("=" * 60)
    print("Road Sentinel - Camera Connection Test")
    print("=" * 60)

    # Find available cameras
    available_cameras = find_available_cameras()

    if not available_cameras:
        print("\n✗ No cameras found!")
        print("  Please check:")
        print("  1. Camera is connected")
        print("  2. Camera permissions are set correctly")
        print("  3. No other application is using the camera")
        return 1

    print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")

    # If --list flag, just show cameras and exit
    if args.list:
        return 0

    # Check AI service availability (unless --no-ai)
    ai_available = False
    if not args.no_ai:
        ai_available = check_ai_service()

    use_ai = ai_available and not args.no_ai

    # Select camera
    if args.camera is not None:
        if args.camera in available_cameras:
            camera_id = args.camera
        else:
            print(f"\n✗ Camera {args.camera} not found!")
            print(f"  Available cameras: {available_cameras}")
            return 1
    else:
        # Use first available camera
        camera_id = available_cameras[0]

    # Run camera test (with camera switching support)
    current_camera_idx = available_cameras.index(camera_id)

    while True:
        result = run_camera_test(camera_id, use_ai, args.confidence)

        if result == 'change_camera':
            # Switch to next camera
            current_camera_idx = (current_camera_idx + 1) % len(available_cameras)
            camera_id = available_cameras[current_camera_idx]
            print(f"Switching to camera {camera_id}...")
            time.sleep(0.5)  # Brief pause before opening new camera
        else:
            # Normal exit
            break

    print("\nCamera test completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
