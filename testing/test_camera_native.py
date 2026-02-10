#!/usr/bin/env python3
"""
Camera test using native resolution (no resolution change)
For troubleshooting Windows camera issues
"""
import cv2
import sys

# Try camera 0 with DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Failed to open camera")
    sys.exit(1)

# Get native resolution (don't change it)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"✅ Camera opened successfully")
print(f"Resolution: {width}x{height} @ {fps}fps")
print(f"Press 'Q' to quit")

cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print(f"❌ Failed to read frame {frame_count}")
        break

    frame_count += 1

    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Test completed - {frame_count} frames captured")
